"""HuggingFace token: load / validate / atomic save.

Token lives at `~/.imgen/hf_token` (chmod 600). v0.2.x and earlier kept it
at `~/.hf_token`; we still read that legacy path as a fallback and
auto-migrate to the new location on first load so colleagues who upgrade
don't have to do anything manual.

Precedence in `load_token()`:
    1. $HF_TOKEN env var (no file touched, no migration).
    2. ~/.imgen/hf_token (new path).
    3. ~/.hf_token (legacy) → moved to new path on read, value returned.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .colors import ok, warn
from .paths import HF_CLI_TOKEN_FILE, LEGACY_TOKEN_FILE, TOKEN_FILE, ensure_state_dir

__all__ = [
    "TOKEN_MAX_BYTES",
    "TokenValidation",
    "ValidationError",
    "active_token_path",
    "check_token_perms",
    "load_token",
    "safe_display_username",
    "save_token_atomic",
    "sync_token_to_hf_cli_store",
    "validate_token",
]

# The closed set of failure kinds :func:`validate_token` reports.
# Typed as ``Literal`` so mypy --strict catches typos and exhaustive
# match statements at call sites. (v0.3.6 architect IMP-2.)
ValidationError = Literal["auth", "network", "parse"]


def safe_display_username(name: str) -> str:
    """v0.7.2 security NIT: strip non-printable chars from an HF
    account name before printing.

    HF account names are user-controlled (the HF user picks them).
    A maliciously crafted account could embed control bytes — ANSI
    escapes (``\\x1b[2J`` clears the terminal), C0 controls, DEL
    (``\\x7f``), C1 controls (``\\x80``–``\\x9f``). The risk profile
    is small (single-user Mac, user's own HF account), but the
    repr-or-strip pattern is the established defence at every other
    user-supplied-string-reaches-terminal surface (v0.4 security
    IMP-2 on alias paths, v0.3.6 style filename safety, etc.).

    Strategy: keep printable Unicode; replace control chars with
    a single ``?``. Empty result (e.g. all-control input) falls
    back to ``"?"``. Length cap 80 chars — no HF username is that
    long; if one is, truncate with ``…`` so the display line stays
    one row.

    Pure: no I/O.

    Accepted residual risk (v0.7.2 security NIT, documented):
    ``str.isprintable()`` PASSES bidirectional control codepoints
    (U+202E RTL OVERRIDE, U+200B ZERO-WIDTH SPACE, U+200D ZERO-WIDTH
    JOINER, U+2066–U+2069 isolates) and confusable / homoglyph
    characters. On a capable terminal these can visually reorder or
    hide text. Blocking all non-ASCII is hostile to legitimate
    international usernames (Cyrillic, CJK, emoji ARE valid HF
    names), so we accept the residual risk for the single-user-Mac
    threat profile. Tighten only if a use case appears that demands
    strict ASCII-or-bust display.
    """
    safe = "".join(c if c.isprintable() else "?" for c in name)
    if not safe:
        return "?"
    if len(safe) > 80:
        safe = safe[:79] + "…"
    return safe


@dataclass(frozen=True, slots=True)
class TokenValidation:
    """Result of :func:`validate_token`.

    Either ``username`` is set (success) or ``error`` is set (failure) —
    never both, never neither. The invariant is enforced at construction
    via :meth:`__post_init__`; ``ValueError`` on misuse beats "silent
    fall-through into the wrong branch at the caller" (which is what
    the pre-v0.3.6 ``str | None`` shape allowed).

    Frozen + slots: immutable like a NamedTuple, no per-instance dict.
    A NamedTuple subclass would be a more obvious fit, but ``typing.
    NamedTuple`` (Python 3.12) refuses ``__new__`` / ``__init__``
    overrides and we need to validate at construction. Surface is
    attribute-access only (``r.username`` / ``r.error``), matching the
    NamedTuple ergonomics callers already used in v0.3.5.

    error:
        ``None``       — success; ``username`` holds the HF whoami name.
        ``"auth"``     — HF rejected the token (401). Token is invalid,
                         revoked, or doesn't carry the needed scopes.
        ``"network"``  — couldn't reach HF (offline, DNS fail, timeout,
                         non-401 HTTP error including 5xx). User's
                         token may still be fine — try later.
        ``"parse"``    — got a 200 but the response wasn't valid JSON
                         with a recognizable name/fullname field.
                         Captive portals and transparent proxies serve
                         this kind of response.
    """
    username: str | None
    error: ValidationError | None

    def __post_init__(self) -> None:
        # XOR check: exactly one of (username, error) must be set.
        # Both-None means a programming bug at the construction site
        # (e.g. a forgotten return value); both-set means the caller
        # is trying to express a contradictory state. Either way,
        # raising at construction is strictly more useful than letting
        # setup.py's if/elif chain dispatch to the wrong message.
        # (v0.3.6 architect IMP-2 + python-reviewer IMPORTANT.)
        if (self.username is None) == (self.error is None):
            raise ValueError(
                "TokenValidation requires exactly one of username/error "
                f"to be set; got username={self.username!r}, "
                f"error={self.error!r}"
            )

# Cap on token file size. Real HF tokens are ~70 chars (`hf_` + 37-char
# secret + room to grow). 4 KB is several orders above realistic use; a
# larger file means something's wrong — refuse rather than slurp into
# memory and pass to mflux.
TOKEN_MAX_BYTES = 4096

# Per-process guard so a failing legacy migration (e.g. read-only home)
# only warns once per CLI run, not on every load_token() call.
_migrate_attempted = False


def load_token() -> str | None:
    """Return the HF token, or None if no source provided.

    Side effect: if only the legacy `~/.hf_token` is present, attempts to
    move it to `~/.imgen/hf_token` (atomic rename, chmod 600). If the
    migration fails, the legacy file is still read so the user isn't
    blocked, but a warning explains how to move it manually.
    """
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()

    if TOKEN_FILE.exists():
        return _read_token_file(TOKEN_FILE)

    if LEGACY_TOKEN_FILE.exists():
        if _try_migrate_legacy():
            return _read_token_file(TOKEN_FILE)
        return _read_token_file(LEGACY_TOKEN_FILE)

    return None


def active_token_path() -> Path | None:
    """The path `load_token()` would read from (None if no file exists).

    Ignores $HF_TOKEN — this is for reporting which on-disk file backs
    the token, e.g. for permission checks or doctor output.
    """
    if TOKEN_FILE.exists():
        return TOKEN_FILE
    if LEGACY_TOKEN_FILE.exists():
        return LEGACY_TOKEN_FILE
    return None


def check_token_perms() -> bool:
    """Return True if the active token file has 0o600 perms (or no file)."""
    active = active_token_path()
    if active is None:
        return True
    mode = active.stat().st_mode & 0o777
    return mode == 0o600


def save_token_atomic(tok: str) -> None:
    """Write token to TOKEN_FILE with 0600 perms atomically.

    O_CREAT|O_EXCL ensures no world-readable window between write and chmod.
    Caller must delete the existing file first if updating. STATE_DIR is
    created if missing so cmd_setup works on a fresh install before any
    other state-dir-touching command has run.
    """
    ensure_state_dir()
    fd = os.open(str(TOKEN_FILE),
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)


SyncStatus = Literal["written", "matched", "diverged_overwritten", "error"]


def sync_token_to_hf_cli_store(tok: str) -> SyncStatus:
    """Sync ``tok`` into HF CLI's token store (``~/.cache/huggingface/token``).

    v0.7.12 (gap 9): close the silent-drift surface where
    ``~/.imgen/hf_token`` and ``~/.cache/huggingface/token`` carried
    different tokens — imgen kept working off its own copy while
    ``hf download`` / diffusers failed with "Invalid user token". The
    standalone ``hf`` CLI normally writes 0o644; we tighten to 0o600
    on each sync (best-effort — ``hf`` may re-widen on its own next
    write, but a defence-in-depth chmod here costs nothing).

    Returns one of:
        ``"written"``               — file didn't exist; new file created.
        ``"matched"``               — file already had exact same content,
                                      no-op.
        ``"diverged_overwritten"``  — file existed with DIFFERENT content
                                      (the drift case); overwritten with
                                      ``tok``. Caller should warn the
                                      user about the divergence so they
                                      know the prior HF CLI token is
                                      gone.
        ``"error"``                 — write failed (filesystem-level OS
                                      error). Caller should report but
                                      not abort — imgen itself still has
                                      its token via ``save_token_atomic``.

    Pure-ish: no env reads, no network. Touches HF_CLI_TOKEN_FILE +
    parent dir (auto-created with mode 0o700).

    Single-source-of-truth at setup time: imgen setup is treated as the
    authority for the user's HF token because the user JUST interactively
    pasted it. A previously-set HF CLI token (e.g. from a stale
    ``hf auth login`` months ago) is informed-consent-overwritten via the
    ``diverged_overwritten`` return value, not silently blown away.
    """
    try:
        parent = HF_CLI_TOKEN_FILE.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        if HF_CLI_TOKEN_FILE.exists():
            try:
                existing = HF_CLI_TOKEN_FILE.read_text()
            except OSError:
                # Unreadable but exists — treat as divergence and try
                # to overwrite. If the overwrite also fails the outer
                # try/except returns "error".
                existing = None
            if existing == tok:
                # Tighten perms even on no-content-change in case the
                # HF CLI last wrote 0o644 — best-effort, don't fail if
                # chmod is rejected on a weird filesystem.
                try:
                    HF_CLI_TOKEN_FILE.chmod(0o600)
                except OSError:
                    pass
                return "matched"
            # Diverged — overwrite. Use O_TRUNC because the file
            # already exists; perms preserved via explicit chmod after
            # write so the new content lands at 0o600 even if the file
            # had wider perms before.
            HF_CLI_TOKEN_FILE.write_text(tok)
            HF_CLI_TOKEN_FILE.chmod(0o600)
            return "diverged_overwritten"

        # Fresh write — use O_CREAT|O_EXCL + 0o600 from creation to
        # avoid any world-readable window (same discipline as
        # save_token_atomic for TOKEN_FILE).
        fd = os.open(str(HF_CLI_TOKEN_FILE),
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(tok)
        return "written"
    except OSError:
        return "error"


def validate_token(token: str) -> TokenValidation:
    """Hit HF whoami; distinguish auth-fail, network-down, and parse errors.

    Pre-v0.3.6 this returned ``str | None`` and the caller couldn't tell
    "your token is wrong" (actionable: replace token) from "you're
    offline" (actionable: try later) from "captive portal" (actionable:
    log in to the wifi). The lumped warn confused users. Now returns
    :class:`TokenValidation` so the caller phrases the right message.
    (python #7 from v0.1.x review.)
    """
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            # Cap response size — defends against DNS hijack / captive portal
            # serving arbitrary bytes.
            raw = resp.read(64_000)
            if len(raw) >= 64_000:
                return TokenValidation(None, "parse")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return TokenValidation(None, "parse")
            name = data.get("name") or data.get("fullname")
            if name:
                # v0.7.2 security NIT: HF account names are user-
                # controlled. A maliciously crafted account could
                # embed ANSI escapes (e.g. `\x1b[2J` to clear the
                # terminal) that fire when imgen setup / doctor
                # prints the name. Strip non-printable chars before
                # returning — defence-in-depth matching the v0.4
                # security IMP-2 pattern on user-supplied strings.
                name = safe_display_username(name)
                return TokenValidation(name, None)
            return TokenValidation(None, "parse")
    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError — must match first so the
        # status code is available. 401 means HF saw the token and
        # rejected it; everything else (403/404/5xx) is HF being weird
        # or down — not actionable on the user's token side.
        if e.code == 401:
            return TokenValidation(None, "auth")
        return TokenValidation(None, "network")
    except (urllib.error.URLError, TimeoutError, OSError):
        return TokenValidation(None, "network")


# ── internal helpers ────────────────────────────────────────────────────

def _read_token_file(path: Path) -> str | None:
    """Read a token file with size cap. Warns + returns None on issues."""
    try:
        size = path.stat().st_size
    except OSError as e:
        warn(f"Couldn't stat {path}: {e}")
        return None
    if size > TOKEN_MAX_BYTES:
        warn(f"{path} too large ({size} bytes; cap {TOKEN_MAX_BYTES}) "
             "— refusing to load. Replace the file with a valid token.")
        return None
    try:
        return path.read_text().strip()
    except OSError as e:
        warn(f"Couldn't read {path}: {e}")
        return None


def _try_migrate_legacy() -> bool:
    """Promote LEGACY_TOKEN_FILE → TOKEN_FILE with 0o600 perms from creation.

    Uses save_token_atomic (O_CREAT|O_EXCL 0o600) + unlink rather than
    os.replace + os.chmod. The latter has a window where the new file
    inherits the legacy file's possibly wider perms (huggingface-cli
    commonly writes 0o644) until the chmod call completes — exploitable
    on a shared Mac. save_token_atomic creates with 0o600 from the
    syscall, eliminating the window.

    Oversized / unreadable legacy files refuse migration silently and let
    load_token's fallback _read_token_file(LEGACY) emit the single warn.

    Per-process: a failing migration only logs once. Returns True on
    success or sibling-won race; False if migration was skipped or
    failed.
    """
    global _migrate_attempted
    if _migrate_attempted:
        return False
    _migrate_attempted = True

    # Probe size before reading — we don't want to slurp megabytes of
    # garbage just to refuse to migrate it. Mirrors _read_token_file's
    # check; the load_token fallback will warn about the bad file once.
    try:
        size = LEGACY_TOKEN_FILE.stat().st_size
    except OSError:
        return False
    if size > TOKEN_MAX_BYTES:
        return False

    try:
        content = LEGACY_TOKEN_FILE.read_text()
    except OSError:
        return False

    try:
        ensure_state_dir()
        save_token_atomic(content)
    except FileExistsError:
        # Sibling process beat us — TOKEN_FILE already has correct perms
        # from the sibling's save_token_atomic. Clean up the legacy file
        # we no longer need; suppress unlink failure (sibling may have
        # already removed it).
        try:
            LEGACY_TOKEN_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return True
    except OSError as e:
        warn(f"Couldn't migrate {LEGACY_TOKEN_FILE} → {TOKEN_FILE}: {e}. "
             f"Move it manually: mv {LEGACY_TOKEN_FILE} {TOKEN_FILE}")
        return False

    try:
        LEGACY_TOKEN_FILE.unlink()
    except OSError as e:
        warn(f"Saved new HF token but couldn't remove legacy "
             f"{LEGACY_TOKEN_FILE}: {e}")
    ok(f"Migrated HF token: {LEGACY_TOKEN_FILE} → {TOKEN_FILE}")
    return True

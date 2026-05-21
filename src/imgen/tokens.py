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
from pathlib import Path

from .colors import ok, warn
from .paths import LEGACY_TOKEN_FILE, TOKEN_FILE, ensure_state_dir

__all__ = [
    "TOKEN_MAX_BYTES",
    "active_token_path",
    "check_token_perms",
    "load_token",
    "save_token_atomic",
    "validate_token",
]

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


def validate_token(token: str) -> str | None:
    """Hit HF whoami; return username on success, None on failure."""
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
                return None
            data = json.loads(raw)
            return data.get("name") or data.get("fullname")
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


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

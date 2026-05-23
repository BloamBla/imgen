"""HF token loader bounds + behaviour.

The token file lives at `~/.imgen/hf_token` (chmod 600). For users who
upgraded from v0.2.x and earlier we still read `~/.hf_token` as a legacy
fallback, auto-migrating to the new path on first load.

A malicious or buggy producer could leave a multi-megabyte file there;
we cap reads so the value never becomes a memory-exhaustion vector.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from imgen.tokens import TOKEN_MAX_BYTES, active_token_path, load_token


@pytest.fixture
def tmp_token(tmp_path, monkeypatch):
    """Redirect both token paths + STATE_DIR to a per-test tmp location.

    Mutates module-level constants — restored by monkeypatch on test exit.
    Also wipes any $HF_TOKEN env var so file paths are the only source,
    and resets the per-process auto-migrate guard so each test gets a
    fresh attempt.

    Returns SimpleNamespace(new=Path, legacy=Path, state_dir=Path).
    """
    state_dir = tmp_path / ".imgen"
    state_dir.mkdir(mode=0o700)
    new_token = state_dir / "hf_token"
    legacy_token = tmp_path / ".hf_token"

    monkeypatch.delenv("HF_TOKEN", raising=False)

    import imgen.paths as paths_mod
    import imgen.tokens as tokens_mod
    monkeypatch.setattr(paths_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(tokens_mod, "TOKEN_FILE", new_token)
    monkeypatch.setattr(tokens_mod, "LEGACY_TOKEN_FILE", legacy_token)
    monkeypatch.setattr(tokens_mod, "_migrate_attempted", False)

    return SimpleNamespace(new=new_token, legacy=legacy_token,
                           state_dir=state_dir)


def test_load_token_no_file_returns_none(tmp_token):
    assert load_token() is None


def test_load_token_normal_size_returns_stripped_content(tmp_token):
    tmp_token.new.write_text("hf_abc123" + "x" * 50 + "\n")
    result = load_token()
    assert result == "hf_abc123" + "x" * 50  # trailing \n stripped


def test_load_token_oversized_file_returns_none_with_warning(tmp_token, capsys):
    """A rogue token file with megabytes of junk shouldn't be slurped
    into memory and passed to mflux. Refuse + warn. (security I4 / v0.1.x
    audit security #17)"""
    payload = "hf_" + "x" * (TOKEN_MAX_BYTES + 100)
    tmp_token.new.write_text(payload)
    result = load_token()
    assert result is None
    captured = capsys.readouterr()
    assert "too large" in (captured.out + captured.err).lower()


def test_load_token_at_cap_is_accepted(tmp_token):
    """Boundary: file exactly TOKEN_MAX_BYTES is OK."""
    payload = "h" * TOKEN_MAX_BYTES
    tmp_token.new.write_text(payload)
    result = load_token()
    assert result == payload


def test_load_token_env_var_overrides_file(tmp_token, monkeypatch):
    """$HF_TOKEN wins over file (existing v0.1.x behaviour — pin)."""
    tmp_token.new.write_text("hf_from_file")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert load_token() == "hf_from_env"


# ── Migration: ~/.hf_token → ~/.imgen/hf_token ──────────────────────────

def test_load_token_legacy_only_auto_migrates(tmp_token, capsys):
    """Only legacy file exists → contents readable + file moved to new path."""
    tmp_token.legacy.write_text("hf_legacy_token_value")
    tmp_token.legacy.chmod(0o600)

    result = load_token()

    assert result == "hf_legacy_token_value"
    assert tmp_token.new.exists(), "legacy should be migrated to new path"
    assert not tmp_token.legacy.exists(), "legacy file should be gone"
    captured = capsys.readouterr()
    assert "migrated" in (captured.out + captured.err).lower()


def test_load_token_legacy_migration_preserves_0600_perms(tmp_token):
    """Migrated file must still be 0o600 even if rename preserved a wider mode."""
    tmp_token.legacy.write_text("hf_legacy")
    tmp_token.legacy.chmod(0o644)  # intentionally wrong

    load_token()

    mode = tmp_token.new.stat().st_mode & 0o777
    assert mode == 0o600


def test_load_token_new_path_wins_over_legacy(tmp_token):
    """If both exist, new path wins; legacy is left untouched."""
    tmp_token.new.write_text("hf_new_value")
    tmp_token.legacy.write_text("hf_legacy_value")

    result = load_token()

    assert result == "hf_new_value"
    assert tmp_token.legacy.exists(), "legacy must not be touched when new exists"


def test_load_token_env_var_overrides_legacy_file(tmp_token, monkeypatch):
    """$HF_TOKEN beats legacy too, and does not trigger migration."""
    tmp_token.legacy.write_text("hf_from_legacy")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    assert load_token() == "hf_from_env"
    assert tmp_token.legacy.exists(), "env-var path must not migrate the file"


def test_load_token_creates_state_dir_during_migration(tmp_token):
    """Migration must create ~/.imgen if the user upgraded before it existed.

    Simulate by removing the state_dir the fixture pre-created.
    """
    # blow away state_dir to mimic a fresh-install path
    tmp_token.new.parent.rmdir()
    tmp_token.legacy.write_text("hf_legacy")

    result = load_token()

    assert result == "hf_legacy"
    assert tmp_token.new.exists()
    assert (tmp_token.new.parent.stat().st_mode & 0o777) == 0o700


# ── active_token_path() ─────────────────────────────────────────────────

def test_active_token_path_returns_none_when_no_file(tmp_token):
    assert active_token_path() is None


def test_active_token_path_returns_new_when_only_new_exists(tmp_token):
    tmp_token.new.write_text("hf_x")
    assert active_token_path() == tmp_token.new


def test_active_token_path_returns_legacy_when_only_legacy_exists(tmp_token):
    tmp_token.legacy.write_text("hf_x")
    assert active_token_path() == tmp_token.legacy


def test_active_token_path_prefers_new_when_both_exist(tmp_token):
    tmp_token.new.write_text("hf_new")
    tmp_token.legacy.write_text("hf_legacy")
    assert active_token_path() == tmp_token.new


# ── Migration race + edge cases (post-v0.2.2 review) ────────────────────

def test_load_token_legacy_oversized_does_not_migrate(tmp_token, capsys):
    """A bloated legacy file (huggingface-cli writing garbage etc.) must
    not be migrated into the new path — that would just promote the
    garbage. _read_token_file's oversize warn fires; no new file is
    created."""
    payload = "h" * (TOKEN_MAX_BYTES + 100)
    tmp_token.legacy.write_text(payload)

    result = load_token()

    assert result is None
    assert not tmp_token.new.exists(), "oversize legacy must not be migrated"
    assert tmp_token.legacy.exists(), "legacy stays for manual cleanup"
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "too large" in out.lower()
    # The 'too large' warn must fire at most twice (once during migration
    # probe + once during fallback read) — never more.
    assert out.lower().count("too large") <= 2


def test_try_migrate_legacy_handles_sibling_already_wrote(tmp_token):
    """If TOKEN_FILE exists when _try_migrate_legacy tries to save (i.e. a
    sibling imgen process beat us between our `not TOKEN_FILE.exists()`
    check in load_token and our save), save_token_atomic raises
    FileExistsError. We swallow it, clean up the legacy file, and report
    success — the sibling's value is what subsequent reads will see.

    Tests _try_migrate_legacy directly since load_token() short-circuits
    at TOKEN_FILE.exists() in production; the race window only opens
    once you're already inside the migration function.
    """
    from imgen.tokens import _try_migrate_legacy

    tmp_token.legacy.write_text("hf_legacy_value")
    tmp_token.new.write_text("hf_sibling_value")
    tmp_token.new.chmod(0o600)

    result = _try_migrate_legacy()

    assert result is True
    assert not tmp_token.legacy.exists(), "legacy must be cleaned after sibling won"
    # Sibling's content preserved — we didn't overwrite.
    assert tmp_token.new.read_text() == "hf_sibling_value"


def test_load_token_legacy_migration_never_widens_perms(tmp_token):
    """Pin the no-chmod-window guarantee: even when legacy is world-readable
    AND we observe the new file mid-creation (which we can't, but we can
    pin the post-condition), the new file's mode is 0o600 from creation.
    save_token_atomic uses O_CREAT|O_EXCL 0o600, so there is no window
    where the new file inherits 0o644."""
    tmp_token.legacy.write_text("hf_legacy")
    tmp_token.legacy.chmod(0o644)

    load_token()

    # Belt-and-braces: assert final mode, and that the legacy file's
    # mode never propagated. (The real race-window guard is in the
    # implementation choice — save_token_atomic vs os.replace+chmod.)
    new_mode = tmp_token.new.stat().st_mode & 0o777
    assert new_mode == 0o600
    assert not tmp_token.legacy.exists()


# ── save_token_atomic auto-creates state dir ────────────────────────────

def test_save_token_atomic_creates_state_dir(tmp_token):
    """save_token_atomic must work on a fresh install where ~/.imgen
    doesn't exist yet — cmd_setup writes the token before it creates
    state dirs."""
    from imgen.tokens import save_token_atomic
    tmp_token.new.parent.rmdir()  # mimic fresh install

    save_token_atomic("hf_freshly_set")

    assert tmp_token.new.exists()
    assert tmp_token.new.read_text() == "hf_freshly_set"
    assert (tmp_token.new.stat().st_mode & 0o777) == 0o600


# ── validate_token: auth / network / parse distinction (python #7) ──────


class _FakeResponse:
    """Minimal urlopen-context-manager stand-in for validate_token tests."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, size: int | None = None) -> bytes:
        return self._body[:size] if size else self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


def _patch_urlopen(monkeypatch, handler) -> None:
    """Patch the symbol validate_token actually calls. tokens.py uses
    `urllib.request.urlopen` via the imported module reference, so
    patching the module attribute is the right surface."""
    import imgen.tokens as tokens_mod
    monkeypatch.setattr(tokens_mod.urllib.request, "urlopen", handler)


def test_validate_token_success_returns_username(monkeypatch):
    """200 + JSON body with a `name` field → success."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(b'{"name": "alice"}'),
    )
    result = validate_token("hf_abc")
    assert result.username == "alice"
    assert result.error is None


def test_validate_token_success_with_fullname_field(monkeypatch):
    """HF whoami may return `fullname` instead of `name` for some accounts."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(b'{"fullname": "Alice Q."}'),
    )
    result = validate_token("hf_abc")
    assert result.username == "Alice Q."
    assert result.error is None


def test_validate_token_401_returns_auth_error(monkeypatch):
    """HTTP 401 from HF → token rejected. Distinct from "couldn't reach"
    so the caller can phrase an actionable message (rotate token vs
    check network)."""
    import urllib.error
    from imgen.tokens import validate_token

    def fake(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, None
        )
    _patch_urlopen(monkeypatch, fake)
    result = validate_token("hf_revoked")
    assert result.username is None
    assert result.error == "auth"


def test_validate_token_500_returns_network_error(monkeypatch):
    """5xx is HF being down/flaky, not a token problem — bucket as
    "network" (caller messages: try later, token may still be fine)."""
    import urllib.error
    from imgen.tokens import validate_token

    def fake(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Server Error", {}, None
        )
    _patch_urlopen(monkeypatch, fake)
    result = validate_token("hf_abc")
    assert result.error == "network"


def test_validate_token_urlerror_returns_network_error(monkeypatch):
    """Offline / DNS fail / connection refused → network bucket."""
    import urllib.error
    from imgen.tokens import validate_token

    def fake(req, timeout=None):
        raise urllib.error.URLError("offline")
    _patch_urlopen(monkeypatch, fake)
    result = validate_token("hf_abc")
    assert result.error == "network"


def test_validate_token_timeout_returns_network_error(monkeypatch):
    """urlopen timeout → network bucket. validate_token caps at 10s."""
    from imgen.tokens import validate_token

    def fake(req, timeout=None):
        raise TimeoutError("slow")
    _patch_urlopen(monkeypatch, fake)
    result = validate_token("hf_abc")
    assert result.error == "network"


def test_validate_token_non_json_returns_parse_error(monkeypatch):
    """200 + HTML body (captive portal / proxy login page) — JSON parse
    fails, distinct from auth/network so the user knows to log in to
    the wifi rather than rotating their token."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(b"<html>captive portal</html>"),
    )
    result = validate_token("hf_abc")
    assert result.error == "parse"


def test_validate_token_json_without_name_returns_parse_error(monkeypatch):
    """200 + JSON but no `name`/`fullname` — HF API shape changed or
    response is from a non-HF endpoint. Bucket as "parse"."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(b'{"unrelated": "x"}'),
    )
    result = validate_token("hf_abc")
    assert result.error == "parse"


def test_validate_token_oversized_response_returns_parse_error(monkeypatch):
    """Response body >= 64 KB cap → treated as parse failure. Defends
    against DNS hijack serving infinite bytes."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(b"x" * 80_000),
    )
    result = validate_token("hf_abc")
    assert result.error == "parse"


# ── TokenValidation invariant (architect IMP-2 + python IMPORTANT-3) ────


def test_token_validation_rejects_both_fields_none():
    """The "either username or error, never both, never neither"
    invariant is enforced at construction. Both-None means a programming
    bug at the call site (e.g. forgotten return value path); raising
    at __new__ beats letting setup.py dispatch to the wrong elif branch
    with a misleading message."""
    from imgen.tokens import TokenValidation
    with pytest.raises(ValueError, match="exactly one"):
        TokenValidation(None, None)


def test_token_validation_rejects_both_fields_set():
    """Both-set is contradictory (a successful validation can't also
    be a failure). Raising at __new__ catches misuse at the source."""
    from imgen.tokens import TokenValidation
    with pytest.raises(ValueError, match="exactly one"):
        TokenValidation("alice", "auth")


def test_token_validation_accepts_success_state():
    """username set, error None — the documented success shape."""
    from imgen.tokens import TokenValidation
    result = TokenValidation("alice", None)
    assert result.username == "alice"
    assert result.error is None


def test_token_validation_accepts_each_failure_kind():
    """All three documented error kinds construct cleanly."""
    from imgen.tokens import TokenValidation
    for kind in ("auth", "network", "parse"):
        result = TokenValidation(None, kind)  # type: ignore[arg-type]
        assert result.error == kind


# ── v0.7.2: safe_display_username — control-byte sanitisation ────────


class TestSafeDisplayUsername:
    """v0.7.2 security NIT: HF account names are user-controlled.
    A maliciously crafted account name with ANSI escapes could clear
    the terminal when imgen setup / doctor prints it. Defence-in-
    depth strip matching the v0.4 IMP-2 pattern for any user-supplied
    string reaching the terminal."""

    def test_plain_ascii_unchanged(self):
        from imgen.tokens import safe_display_username
        assert safe_display_username("alice") == "alice"

    def test_unicode_printable_kept(self):
        """Cyrillic / CJK / emoji — printable Unicode survives."""
        from imgen.tokens import safe_display_username
        assert safe_display_username("Станислав") == "Станислав"
        assert safe_display_username("用户名") == "用户名"

    def test_ansi_escape_replaced(self):
        """\\x1b (ESC) starts an ANSI sequence — would clear the
        terminal if printed verbatim. Stripped to ``?``."""
        from imgen.tokens import safe_display_username
        out = safe_display_username("alice\x1b[2Jevil")
        assert "\x1b" not in out
        assert "alice" in out
        assert "evil" in out
        assert "?" in out

    def test_c0_controls_replaced(self):
        """C0 range (\\x00–\\x1f including null, bell, backspace)."""
        from imgen.tokens import safe_display_username
        out = safe_display_username("alice\x00\x07\x08bob")
        assert "\x00" not in out
        assert "\x07" not in out
        assert "\x08" not in out

    def test_del_replaced(self):
        """\\x7f (DEL) is non-printable, must be stripped."""
        from imgen.tokens import safe_display_username
        assert "\x7f" not in safe_display_username("foo\x7fbar")

    def test_c1_controls_replaced(self):
        """C1 range (\\x80–\\x9f) — terminal escape lead-ins."""
        from imgen.tokens import safe_display_username
        out = safe_display_username("alice\x9b[2Jbob")
        assert "\x9b" not in out

    def test_all_control_input_falls_back_to_question_mark(self):
        from imgen.tokens import safe_display_username
        assert safe_display_username("\x00\x01\x02") == "???"

    def test_empty_input_returns_question_mark(self):
        from imgen.tokens import safe_display_username
        assert safe_display_username("") == "?"

    def test_long_input_truncated(self):
        from imgen.tokens import safe_display_username
        long_name = "a" * 200
        out = safe_display_username(long_name)
        assert len(out) <= 80


def test_validate_token_sanitises_username(monkeypatch):
    """v0.7.2: end-to-end — a malicious account name comes back from
    HF, validate_token strips control bytes before returning. Caller
    in setup.py prints the username verbatim, so the sanitisation
    must happen INSIDE validate_token."""
    from imgen.tokens import validate_token
    _patch_urlopen(
        monkeypatch,
        lambda req, timeout=None: _FakeResponse(
            b'{"name": "alice\\u001b[2Jevil"}',
        ),
    )
    result = validate_token("hf_abc")
    assert result.username is not None
    assert "\x1b" not in result.username

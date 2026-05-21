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

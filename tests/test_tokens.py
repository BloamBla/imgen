"""HF token loader bounds + behaviour.

The token file is `~/.hf_token` (chmod 600). A malicious or buggy producer
could leave a multi-megabyte file there; we cap reads so the value never
becomes a memory-exhaustion vector.
"""
from __future__ import annotations

import os

import pytest

from imgen.tokens import TOKEN_MAX_BYTES, load_token


@pytest.fixture
def tmp_token(tmp_path, monkeypatch):
    """Redirect TOKEN_FILE to a per-test tmp location.

    Mutates a module-level constant — restored by monkeypatch on test exit.
    Also wipes any $HF_TOKEN env var so the file path is the only source.
    """
    token_file = tmp_path / ".hf_token"
    monkeypatch.delenv("HF_TOKEN", raising=False)

    import imgen.tokens as tokens_mod
    monkeypatch.setattr(tokens_mod, "TOKEN_FILE", token_file)
    return token_file


def test_load_token_no_file_returns_none(tmp_token):
    assert load_token() is None


def test_load_token_normal_size_returns_stripped_content(tmp_token):
    tmp_token.write_text("hf_abc123" + "x" * 50 + "\n")
    result = load_token()
    assert result == "hf_abc123" + "x" * 50  # trailing \n stripped


def test_load_token_oversized_file_returns_none_with_warning(tmp_token, capsys):
    """A rogue ~/.hf_token with megabytes of junk shouldn't be slurped
    into memory and passed to mflux. Refuse + warn. (security I4 / v0.1.x
    audit security #17)"""
    payload = "hf_" + "x" * (TOKEN_MAX_BYTES + 100)
    tmp_token.write_text(payload)
    result = load_token()
    assert result is None
    captured = capsys.readouterr()
    assert "too large" in (captured.out + captured.err).lower()


def test_load_token_at_cap_is_accepted(tmp_token):
    """Boundary: file exactly TOKEN_MAX_BYTES is OK."""
    payload = "h" * TOKEN_MAX_BYTES
    tmp_token.write_text(payload)
    result = load_token()
    assert result == payload


def test_load_token_env_var_overrides_file(tmp_token, monkeypatch):
    """$HF_TOKEN wins over file (existing v0.1.x behaviour — pin)."""
    tmp_token.write_text("hf_from_file")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert load_token() == "hf_from_env"

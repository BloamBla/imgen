"""resolve_prompt — picks the effective prompt from --custom-prompt /
--prompt-file / stdin without ever putting it in argv.

The security goal: a sensitive prompt should never reach `ps auxww`.
This pure function handles the precedence + edge cases; cmd_generate
just consumes the result.
"""
from __future__ import annotations

import io

import pytest

from imgen.prompt_input import (
    PROMPT_MAX_BYTES,
    PromptInputError,
    resolve_prompt,
)


# ── None / pass-through ──────────────────────────────────────────────────

def test_returns_none_when_nothing_supplied():
    assert resolve_prompt(custom_prompt=None, prompt_file=None) is None


def test_returns_custom_prompt_string_as_is():
    assert resolve_prompt(custom_prompt="hello world", prompt_file=None) \
        == "hello world"


# ── --custom-prompt - reads stdin ────────────────────────────────────────

def test_dash_reads_stdin():
    stdin = io.StringIO("piped prompt text\n")
    result = resolve_prompt(custom_prompt="-", prompt_file=None, stdin=stdin)
    assert result == "piped prompt text"


def test_dash_strips_trailing_whitespace():
    stdin = io.StringIO("piped\n\n  \n")
    result = resolve_prompt(custom_prompt="-", prompt_file=None, stdin=stdin)
    assert result == "piped"


def test_dash_with_empty_stdin_raises():
    with pytest.raises(PromptInputError) as exc_info:
        resolve_prompt(custom_prompt="-", prompt_file=None, stdin=io.StringIO(""))
    assert "empty" in str(exc_info.value).lower()


def test_dash_with_whitespace_only_stdin_raises():
    with pytest.raises(PromptInputError):
        resolve_prompt(
            custom_prompt="-", prompt_file=None, stdin=io.StringIO("   \n\n")
        )


# ── --prompt-file PATH reads file ────────────────────────────────────────

def test_prompt_file_reads_content(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("from file")
    assert resolve_prompt(custom_prompt=None, prompt_file=p) == "from file"


def test_prompt_file_strips_trailing_whitespace(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("text\n\n  \n")
    assert resolve_prompt(custom_prompt=None, prompt_file=p) == "text"


def test_prompt_file_empty_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(PromptInputError) as exc_info:
        resolve_prompt(custom_prompt=None, prompt_file=p)
    assert "empty" in str(exc_info.value).lower()


def test_prompt_file_whitespace_only_raises(tmp_path):
    p = tmp_path / "ws.txt"
    p.write_text("   \n\n   ")
    with pytest.raises(PromptInputError):
        resolve_prompt(custom_prompt=None, prompt_file=p)


def test_prompt_file_missing_raises(tmp_path):
    p = tmp_path / "nope.txt"
    with pytest.raises(PromptInputError) as exc_info:
        resolve_prompt(custom_prompt=None, prompt_file=p)
    assert "not found" in str(exc_info.value).lower()


def test_prompt_file_directory_path_raises(tmp_path):
    # tmp_path is a directory; passing it as prompt-file should fail.
    with pytest.raises(PromptInputError):
        resolve_prompt(custom_prompt=None, prompt_file=tmp_path)


def test_prompt_file_size_cap_enforced(tmp_path):
    """Defends against a multi-GB file getting slurped into RAM."""
    p = tmp_path / "huge.txt"
    p.write_bytes(b"x" * (PROMPT_MAX_BYTES + 1))
    with pytest.raises(PromptInputError) as exc_info:
        resolve_prompt(custom_prompt=None, prompt_file=p)
    assert "too large" in str(exc_info.value).lower()


def test_prompt_file_at_cap_is_accepted(tmp_path):
    """Boundary: exactly PROMPT_MAX_BYTES is OK (after strip)."""
    p = tmp_path / "exact.txt"
    p.write_bytes(b"x" * PROMPT_MAX_BYTES)
    result = resolve_prompt(custom_prompt=None, prompt_file=p)
    assert len(result) == PROMPT_MAX_BYTES


# ── Mutex: --custom-prompt + --prompt-file ───────────────────────────────

def test_both_custom_prompt_and_prompt_file_raises(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("from file")
    with pytest.raises(PromptInputError) as exc_info:
        resolve_prompt(custom_prompt="from CLI", prompt_file=p)
    assert "mutually exclusive" in str(exc_info.value).lower()


def test_both_dash_and_prompt_file_also_raises(tmp_path):
    """--custom-prompt - and --prompt-file are also mutex (dash is still
    a --custom-prompt value)."""
    p = tmp_path / "prompt.txt"
    p.write_text("from file")
    with pytest.raises(PromptInputError):
        resolve_prompt(
            custom_prompt="-", prompt_file=p, stdin=io.StringIO("piped")
        )

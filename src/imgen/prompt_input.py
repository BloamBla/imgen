"""Source the effective custom prompt from --custom-prompt / --prompt-file.

Both channels exist for the same reason: keep prompt text out of `argv`,
so other local users can't read it via `ps auxww`. v0.1.x put the prompt
text directly on the command line — fine for "anime style please", risky
for anything sensitive.

Caller (cmd_generate) passes the two argparse values; this function
returns the effective prompt string or None (no prompt source given).
Raises PromptInputError on:
  - both flags supplied (ambiguous)
  - missing / not-a-file / oversized / empty prompt file
  - empty stdin when --custom-prompt - is used
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import IO

PROMPT_MAX_BYTES = 64 * 1024  # 64 KB — way more than any real prompt

__all__ = ["PromptInputError", "PROMPT_MAX_BYTES", "resolve_prompt"]


class PromptInputError(Exception):
    """Raised when --custom-prompt / --prompt-file can't yield a usable prompt."""


def resolve_prompt(
    *,
    custom_prompt: str | None,
    prompt_file: Path | None,
    stdin: IO[str] = sys.stdin,
) -> str | None:
    """Resolve the effective prompt text from CLI inputs.

    Returns None when neither input is supplied (caller falls back to a
    style preset). `stdin` is injectable for testing — defaults to the
    process stdin so production code doesn't need to thread it.
    """
    if custom_prompt is not None and prompt_file is not None:
        raise PromptInputError(
            "--custom-prompt and --prompt-file are mutually exclusive — "
            "pick one source for the prompt"
        )

    if prompt_file is not None:
        return _read_prompt_file(prompt_file)

    if custom_prompt == "-":
        return _read_stdin(stdin)

    return custom_prompt  # a literal string, or None


def _read_prompt_file(path: Path) -> str:
    if not path.exists():
        raise PromptInputError(f"--prompt-file not found: {path}")
    if not path.is_file():
        raise PromptInputError(f"--prompt-file is not a file: {path}")
    size = path.stat().st_size
    if size > PROMPT_MAX_BYTES:
        raise PromptInputError(
            f"--prompt-file too large: {size} bytes "
            f"(cap {PROMPT_MAX_BYTES})"
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise PromptInputError(f"--prompt-file is empty: {path}")
    return content


def _read_stdin(stdin: IO[str]) -> str:
    content = stdin.read().strip()
    if not content:
        raise PromptInputError(
            "stdin is empty (--custom-prompt - requires piped input)"
        )
    return content

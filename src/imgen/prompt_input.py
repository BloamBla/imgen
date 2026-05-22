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

from .colors import warn

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

    # Empty / whitespace-only literal prompt: fail fast rather than
    # silently fall through. Pre-v0.3.5 the falsy empty string would
    # treat `effective_custom_prompt` as "no custom prompt" in the
    # dispatch downstream — the preset prompt would be used instead,
    # and the user wouldn't know their --custom-prompt was ignored.
    # v0.3.5 made `imgen photo.jpg --custom-prompt "..."` a primary
    # UX path (no-style augmentation lift), so this gap is newly
    # reachable. (v0.3.5 reviewer HIGH.)
    if custom_prompt is not None and not custom_prompt.strip():
        raise PromptInputError(
            "--custom-prompt is empty — pass actual prompt text, "
            "or use '-' to read from stdin"
        )

    return custom_prompt  # a literal string, or None


def _read_prompt_file(path: Path) -> str:
    if not path.exists():
        raise PromptInputError(f"--prompt-file not found: {path}")
    if not path.is_file():
        raise PromptInputError(f"--prompt-file is not a file: {path}")
    # Single open + bounded read avoids the stat-then-read TOCTOU and the
    # extra syscall. PROMPT_MAX_BYTES + 1 lets us tell "exactly at cap"
    # from "over cap".
    try:
        st = path.stat()
        with path.open("rb") as f:
            raw = f.read(PROMPT_MAX_BYTES + 1)
    except OSError as e:
        raise PromptInputError(f"--prompt-file read failed: {path}: {e}") from e
    if len(raw) > PROMPT_MAX_BYTES:
        raise PromptInputError(
            f"--prompt-file too large: > {PROMPT_MAX_BYTES} bytes"
        )
    try:
        content = raw.decode("utf-8").strip()
    except UnicodeDecodeError as e:
        raise PromptInputError(
            f"--prompt-file is not UTF-8: {path}: {e}"
        ) from e
    if not content:
        raise PromptInputError(f"--prompt-file is empty: {path}")
    # Loose-perms warning: the user passed --prompt-file specifically to
    # keep the prompt out of `ps auxww`, so a world-readable file
    # undercuts the threat model. Don't refuse — just warn.
    mode = st.st_mode & 0o777
    if mode != 0o600:
        warn(f"--prompt-file {path} has mode {oct(mode)} — "
             f"chmod 600 if it contains anything sensitive")
    return content


def _read_stdin(stdin: IO[str]) -> str:
    # Bounded read so `cat /dev/zero | imgen --custom-prompt -` can't OOM.
    # Symmetric with --prompt-file's cap. read(N+1) lets us distinguish
    # "exactly at cap" from "over cap" without slurping unbounded.
    raw = stdin.read(PROMPT_MAX_BYTES + 1)
    if len(raw) > PROMPT_MAX_BYTES:
        raise PromptInputError(
            f"stdin input too large: > {PROMPT_MAX_BYTES} bytes "
            "(--custom-prompt - cap)"
        )
    content = raw.strip()
    if not content:
        raise PromptInputError(
            "stdin is empty (--custom-prompt - requires piped input)"
        )
    return content

"""Subprocess wrapper that scrubs HuggingFace tokens from stderr in real time.

mflux / huggingface_hub can include Authorization headers (`Bearer hf_xxx`) in
HTTP error tracebacks written to stderr. Streaming-redact those before they
hit the user's terminal while preserving real-time tqdm progress (which uses
carriage-return overwrites, not newlines).
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import BinaryIO

__all__ = ["build_mflux_env", "format_cmd", "run_with_stderr_redaction"]


# Single source of truth for the env allow-list reaching the mflux
# subprocess. Forwarding the FULL parent environment would leak any
# secret the user's shell carries (other tokens, AWS creds, ssh-agent
# vars) into the child's tracebacks and crash reports. The allow-list
# captures only the env keys mflux + huggingface_hub + MLX genuinely
# consume. (HF_TOKEN is added on top when the backend needs it.)
#
# COLUMNS / LINES are appended at build time from `shutil.get_terminal_size`
# so tqdm renders full-width progress bars instead of a wrapped 80-col
# fallback.
_MFLUX_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    "MLX_METAL_PRECOMPILE_PATH",
)


def build_mflux_env(token: str | None) -> dict[str, str]:
    """Minimal environment for the mflux subprocess.

    Allow-listed keys from :data:`_MFLUX_ENV_ALLOWLIST` are copied from
    the parent environment; ``HF_TOKEN`` is added when the backend
    needs gated-model access; ``COLUMNS`` / ``LINES`` are forwarded
    from the host terminal so tqdm renders at the user's actual width.

    Shared by ``cmd_generate`` and ``cmd_batch`` (v0.3.0 IMP-5 — the
    two call sites used to inline this block separately, risking the
    allow-list drifting between them on the next edit).
    """
    env: dict[str, str] = {
        k: os.environ[k] for k in _MFLUX_ENV_ALLOWLIST if k in os.environ
    }
    if token:
        env["HF_TOKEN"] = token
    term = shutil.get_terminal_size(fallback=(80, 24))
    env["COLUMNS"] = str(term.columns)
    env["LINES"] = str(term.lines)
    return env

# Minimum 36 chars after `hf_` so a truncated prefix at a buffer boundary
# (e.g. `hf_AbC\n` flushed via the last-`\r`-or-`\n` rule before the rest
# of the token arrives) can't sneak through as plaintext. Real HF tokens
# are 36+ chars; anything shorter that looks like `hf_` is harmless.
_TOKEN_LEAK_RE = re.compile(rb"hf_[A-Za-z0-9_\-]{36,}")


def run_with_stderr_redaction(
    cmd: list[str],
    env: dict,
    log_file: BinaryIO | None = None,
) -> int:
    """Run subprocess streaming stderr to terminal with HF token patterns
    redacted on the fly.

    Flushes up to the last `\\n` or `\\r` in the chunk, keeping the tail
    buffered so multi-byte UTF-8 sequences (e.g. tqdm's unicode block chars)
    don't get split mid-character.

    log_file (v0.2.5+): if given, the SAME redacted bytes are appended to
    this file object in real time. Same byte stream as the terminal, so
    the on-disk log is also token-safe. The caller owns the lifecycle —
    typically a BatchLogger's borrowed fd; this helper writes + flushes
    but does NOT close. (Was `log_path: Path | None` in v0.2.3-v0.2.4;
    architect FWD-6 from v0.2.4 review unified ownership under
    BatchLogger.)
    """
    proc = subprocess.Popen(
        cmd, env=env, stderr=subprocess.PIPE, bufsize=0,
    )
    buffer = b""
    try:
        # Explicit guard rather than `assert`: asserts are stripped under
        # `python -O` / PYTHONOPTIMIZE=1, and a None.read(...) call eight
        # lines down would otherwise crash with an opaque AttributeError.
        # (python C3 from v0.2.3 review)
        if proc.stderr is None:
            proc.kill()
            raise RuntimeError(
                "subprocess_helpers: stderr pipe missing — "
                "this is a bug (stderr=PIPE not honoured)"
            )
        while True:
            chunk = proc.stderr.read(256)
            if not chunk:
                if buffer:
                    redacted = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", buffer)
                    sys.stderr.buffer.write(redacted)
                    sys.stderr.buffer.flush()
                    if log_file is not None:
                        log_file.write(redacted)
                        log_file.flush()
                break
            buffer += chunk
            last = max(buffer.rfind(b"\n"), buffer.rfind(b"\r"))
            if last >= 0:
                to_flush, buffer = buffer[:last + 1], buffer[last + 1:]
                redacted = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", to_flush)
                sys.stderr.buffer.write(redacted)
                sys.stderr.buffer.flush()
                if log_file is not None:
                    log_file.write(redacted)
                    log_file.flush()
        # wait() must be inside the same try so a hang here is still
        # interruptible via Ctrl-C — previously the wait() was outside
        # and a wedged mflux child made the shell unresponsive.
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc.returncode


def format_cmd(cmd: list[str]) -> str:
    """Pretty-print a command, keeping --flag value pairs on the same line.

    Quoting uses ``shlex.quote`` so values containing spaces, ``$``,
    backticks, single quotes, or newlines are wrapped/escaped correctly
    — output is structurally safe to read AND to copy-paste back into a
    POSIX shell (zsh/bash). README still recommends re-invoking imgen
    rather than pasting, because the displayed argv reflects what mflux
    will see, not necessarily what the user originally typed (e.g.
    --custom-prompt - from stdin is shown as the resolved text).
    (python #12 from v0.1.x review.)
    """
    parts = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if (token.startswith("--")
                and i + 1 < len(cmd)
                and not cmd[i + 1].startswith("--")):
            parts.append(f"{token} {shlex.quote(cmd[i + 1])}")
            i += 2
        else:
            parts.append(shlex.quote(token))
            i += 1
    return " \\\n  ".join(parts)

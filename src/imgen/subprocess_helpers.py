"""Subprocess wrapper that scrubs HuggingFace tokens from stderr in real time.

mflux / huggingface_hub can include Authorization headers (`Bearer hf_xxx`) in
HTTP error tracebacks written to stderr. Streaming-redact those before they
hit the user's terminal while preserving real-time tqdm progress (which uses
carriage-return overwrites, not newlines).
"""
from __future__ import annotations

import re
import subprocess
import sys

_TOKEN_LEAK_RE = re.compile(rb"hf_[A-Za-z0-9_\-]{8,}")


def run_with_stderr_redaction(cmd: list[str], env: dict) -> int:
    """Run subprocess streaming stderr to terminal with HF token patterns
    redacted on the fly.

    Flushes up to the last `\\n` or `\\r` in the chunk, keeping the tail
    buffered so multi-byte UTF-8 sequences (e.g. tqdm's unicode block chars)
    don't get split mid-character.
    """
    proc = subprocess.Popen(
        cmd, env=env, stderr=subprocess.PIPE, bufsize=0,
    )
    buffer = b""
    try:
        assert proc.stderr is not None
        while True:
            chunk = proc.stderr.read(256)
            if not chunk:
                if buffer:
                    sys.stderr.buffer.write(
                        _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", buffer))
                    sys.stderr.buffer.flush()
                break
            buffer += chunk
            last = max(buffer.rfind(b"\n"), buffer.rfind(b"\r"))
            if last >= 0:
                to_flush, buffer = buffer[:last + 1], buffer[last + 1:]
                sys.stderr.buffer.write(
                    _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", to_flush))
                sys.stderr.buffer.flush()
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

    For human display only — do NOT paste the output into a shell. The
    quoting here is intentionally naive (only escapes `"`) and won't handle
    `$`, backticks, or newlines safely. README warns about this.
    """
    parts = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if (token.startswith("--")
                and i + 1 < len(cmd)
                and not cmd[i + 1].startswith("--")):
            value = cmd[i + 1]
            if " " in value or any(c in value for c in '"\''):
                value = '"' + value.replace('"', '\\"') + '"'
            parts.append(f"{token} {value}")
            i += 2
        else:
            parts.append(token)
            i += 1
    return " \\\n  ".join(parts)

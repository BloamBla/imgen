"""Stderr token redactor regex coverage + end-to-end run_with_stderr_redaction.

`_TOKEN_LEAK_RE` is the safety net that strips `hf_*` tokens from mflux
stderr before it reaches the user's terminal. The regex tests lock the
pattern's behaviour so a future "tighten" or "loosen" is intentional.

v0.2.6 added integration tests at the bottom that spawn a real Python
subprocess (via `sys.executable -c`) to exercise the full tee+redact
flow end-to-end: chunk loop, buffer-up-to-last-newline, dual-write to
terminal stderr + optional log_file, returncode pass-through.

Cost: ~100-150ms per spawned subprocess (Python startup). Suite stays
well under the project's <2s soft target. The previous "lower-coverage
carve-out" in CLAUDE.md for this function was lifted in v0.2.6 — the
reviewer-flagged gap (twice, v0.2.4 and v0.2.5 reviews) wasn't worth
keeping just to save 300ms of suite time on load-bearing security code.
"""
from __future__ import annotations

import io
import re
import sys

import pytest

from imgen.subprocess_helpers import _TOKEN_LEAK_RE, run_with_stderr_redaction


def test_redacts_full_length_token():
    """Real HF tokens are 36+ chars; the redactor must catch them."""
    line = b"Authorization: Bearer hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdef\n"
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", line)
    assert b"hf_AbCd" not in result
    assert b"hf_***REDACTED***" in result


def test_does_not_redact_short_prefix_hf_x():
    """`hf_` followed by < 36 chars must NOT match. Without this lower
    bound, a buffer-boundary split could flush `hf_` + a few chars as
    plaintext (rest of the token would land in next chunk, never
    redacted because that chunk doesn't have an `hf_` prefix).
    (security N1)"""
    line = b"random output mentioning hf_short here\n"
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", line)
    # `hf_short` is 8 chars after the prefix — must NOT be redacted
    # because it CAN'T be a real token.
    assert b"hf_short" in result


def test_does_not_match_words_starting_with_hf():
    """Defensive: don't redact e.g. "hf_huggingface_xyz" (8+ chars but
    in a normal sentence) ... actually it would be redacted because it
    looks like a token. The 36+ floor is what protects normal English
    words like "hf_xyz" from false matches."""
    short_word = b"hf_quick" + b"abc"  # 11 chars after hf_
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", short_word)
    assert short_word in result  # too short to match {36,}


def test_pattern_minimum_length_is_36():
    """Pin the minimum-match length. If the regex changes from {36,} to
    something looser, this test loudly fails to force the reviewer to
    re-justify the choice."""
    pattern_src = _TOKEN_LEAK_RE.pattern
    assert b"{36,}" in pattern_src or "{36,}" in str(pattern_src), \
        f"redactor pattern {pattern_src!r} missing {{36,}} min-length"


# ── end-to-end run_with_stderr_redaction (v0.2.6 — closes review NIT) ──


# A 40-char hf_-prefixed token; >36-char minimum the regex requires.
_FAKE_TOKEN = "hf_" + "A" * 40


def _python_cmd(code: str) -> list[str]:
    """Build a subprocess argv that runs Python code via -c.

    Uses sys.executable so tests pick up the same interpreter that's
    running the suite (matters in tox/multi-venv setups; here mostly
    .venv/bin/python)."""
    return [sys.executable, "-c", code]


def test_run_with_stderr_redaction_redacts_to_terminal(capfdbinary):
    """Token written to subprocess stderr → redacted in parent stderr.

    Uses capfdbinary (not capsys) because subprocess_helpers writes via
    `sys.stderr.buffer.write(redacted_bytes)` — the binary buffer-level
    interface. capsys intercepts sys.stdout/sys.stderr at the text
    level and can miss writes that go directly to fd 2."""
    code = (
        "import sys; "
        f"sys.stderr.write('Authorization: Bearer {_FAKE_TOKEN}\\n')"
    )

    rc = run_with_stderr_redaction(_python_cmd(code), env={})

    assert rc == 0
    err = capfdbinary.readouterr().err
    # Token's tail bytes should not survive in any form.
    assert _FAKE_TOKEN.encode() not in err
    # Redaction placeholder lands instead.
    assert b"hf_***REDACTED***" in err


def test_run_with_stderr_redaction_redacts_to_log_file(capfdbinary):
    """log_file= receives the SAME redacted byte stream as terminal.

    This is the load-bearing invariant for BatchLogger.borrow_fd(): the
    on-disk per-batch log mustn't contain tokens even though terminal
    output is also redacted."""
    code = (
        "import sys; "
        f"sys.stderr.write('Authorization: Bearer {_FAKE_TOKEN}\\n')"
    )
    log = io.BytesIO()

    run_with_stderr_redaction(_python_cmd(code), env={}, log_file=log)

    log_content = log.getvalue()
    assert _FAKE_TOKEN.encode() not in log_content
    assert b"hf_***REDACTED***" in log_content
    # Sanity: both sinks got the redacted version.
    assert b"hf_***REDACTED***" in capfdbinary.readouterr().err


def test_run_with_stderr_redaction_passes_through_short_hf_prefixes(
    capfdbinary,
):
    """Strings like `hf_short` (< 36 chars after the prefix) are NOT
    real tokens — must reach the terminal unchanged. End-to-end
    counterpart to test_does_not_redact_short_prefix_hf_x."""
    code = "import sys; sys.stderr.write('build flag hf_quick rev3\\n')"
    log = io.BytesIO()

    run_with_stderr_redaction(_python_cmd(code), env={}, log_file=log)

    assert b"hf_quick" in capfdbinary.readouterr().err
    assert b"hf_quick" in log.getvalue()


def test_run_with_stderr_redaction_returns_subprocess_returncode(
    capfdbinary,
):
    """The helper's return value is the subprocess's exit code.
    cmd_generate's exit-code map and history `status` field both
    depend on this round-trip."""
    code = "import sys; sys.stderr.write('done\\n'); sys.exit(42)"

    rc = run_with_stderr_redaction(_python_cmd(code), env={})

    assert rc == 42
    # Drain capture so other tests don't see this stderr.
    capfdbinary.readouterr()


def test_run_with_stderr_redaction_handles_multi_chunk_output(
    capfdbinary,
):
    """The chunk loop reads 256 bytes at a time and flushes up to the
    last `\\n` or `\\r`. Verify a payload larger than one chunk gets
    fully redacted (no token fragment slips out as buffer tail)."""
    # Build a payload > 256 bytes so the read loop iterates at least
    # twice. Place the token at a position straddling a 256-byte boundary.
    prefix = "x" * 240
    code = (
        "import sys; "
        f"sys.stderr.write({prefix!r} + 'AUTH: ' + {_FAKE_TOKEN!r} + '\\n')"
    )
    log = io.BytesIO()

    run_with_stderr_redaction(_python_cmd(code), env={}, log_file=log)

    log_content = log.getvalue()
    assert _FAKE_TOKEN.encode() not in log_content
    assert b"hf_***REDACTED***" in log_content
    # Drain.
    capfdbinary.readouterr()

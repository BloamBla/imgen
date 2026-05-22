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

from imgen.subprocess_helpers import (
    _TOKEN_LEAK_RE,
    format_cmd,
    run_with_stderr_redaction,
)


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
# Mixed-case + digits + underscore — closer to real HF token shape, so a
# future regex tightening (e.g. requiring character-class diversity)
# would still match what we test against.
_FAKE_TOKEN = "hf_AbCdEf0123_GhIjKlMnOpQrStUvWxYz" + "AbCd1234"


def _python_cmd(code: str) -> list[str]:
    """Build a subprocess argv that runs Python code via -c.

    Uses sys.executable so tests pick up the same interpreter that's
    running the suite (matters in tox/multi-venv setups; here mostly
    .venv/bin/python)."""
    return [sys.executable, "-c", code]


def test_run_with_stderr_redaction_redacts_to_terminal(capfdbinary):
    """Token written to subprocess stderr → redacted in parent stderr.

    Uses capfdbinary (not capsys) because subprocess_helpers writes via
    `sys.stderr.buffer.write(redacted_bytes)`. capsys replaces
    `sys.stderr` with a text-only stand-in that has no `.buffer`
    attribute — `sys.stderr.buffer.write` would AttributeError under
    capsys. capfdbinary captures fd 2 directly via os.dup2, transparent
    to the production code path."""
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


def test_run_with_stderr_redaction_handles_multi_chunk_output(
    capfdbinary,
):
    """The chunk loop reads 256 bytes at a time and flushes up to the
    last `\\n` or `\\r`. Verify a token that straddles a read boundary
    AFTER a newline-driven flush still gets fully redacted (no token
    fragment slips out as the buffer tail held between reads).

    Force determinism: subprocess writes a first chunk WITH a `\\n` and
    flushes, then writes the token in a second write+flush. This
    ensures the parent's first `read(256)` returns just the first
    chunk (the kernel pipe is drained), and the token lives across
    the second-read boundary.

    Without the deterministic flush, subprocess buffering can deliver
    the entire payload in one syscall write, the parent's one
    `read(256)` returns it all at once, and the multi-chunk path
    isn't exercised. (v0.2.6 review NIT — double-flagged by security
    + architect.)"""
    code = (
        "import sys; "
        "sys.stderr.write('first chunk done\\n'); "
        "sys.stderr.flush(); "
        f"sys.stderr.write('AUTH: ' + {_FAKE_TOKEN!r} + '\\n'); "
        "sys.stderr.flush()"
    )
    log = io.BytesIO()

    run_with_stderr_redaction(_python_cmd(code), env={}, log_file=log)

    log_content = log.getvalue()
    # First chunk delivered intact.
    assert b"first chunk done" in log_content
    # Token in the second chunk fully redacted.
    assert _FAKE_TOKEN.encode() not in log_content
    assert b"hf_***REDACTED***" in log_content


# ── build_mflux_env (v0.3.0 IMP-5: single source of truth) ─────────────


from imgen.subprocess_helpers import build_mflux_env


def test_build_mflux_env_includes_path_when_set(monkeypatch):
    """PATH is in the allow-list — child process needs it to find
    any subprocess it spawns (mflux shelling out to git etc.)."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = build_mflux_env(token=None)
    assert env["PATH"] == "/usr/bin:/bin"


def test_build_mflux_env_omits_keys_not_in_parent(monkeypatch):
    """Allow-list keys that AREN'T set in the parent stay absent —
    no fabricated empties leaking into the child."""
    for k in ("TMPDIR", "MLX_METAL_PRECOMPILE_PATH", "HF_HUB_CACHE"):
        monkeypatch.delenv(k, raising=False)
    env = build_mflux_env(token=None)
    assert "TMPDIR" not in env
    assert "MLX_METAL_PRECOMPILE_PATH" not in env
    assert "HF_HUB_CACHE" not in env


def test_build_mflux_env_excludes_unrelated_parent_vars(monkeypatch):
    """Allow-list is a positive list — anything not enumerated stays
    out, even when set in parent. This is the security guarantee:
    AWS creds, ssh-agent socket, random shell aliases never reach
    the mflux subprocess."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-not-for-mflux")
    monkeypatch.setenv("MY_RANDOM_VAR", "leak-me")
    env = build_mflux_env(token=None)
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "MY_RANDOM_VAR" not in env


def test_build_mflux_env_token_set_when_passed(monkeypatch):
    env = build_mflux_env(token="hf_realtoken123")
    assert env["HF_TOKEN"] == "hf_realtoken123"


def test_build_mflux_env_token_omitted_when_none(monkeypatch):
    """qwen (open backend) → no HF_TOKEN forwarded. Defence-in-depth
    against accidentally injecting credentials into open-backend runs."""
    env = build_mflux_env(token=None)
    assert "HF_TOKEN" not in env


def test_build_mflux_env_forwards_terminal_size(monkeypatch):
    """COLUMNS/LINES from shutil.get_terminal_size — tqdm reads these
    to render a full-width progress bar instead of the 80-col fallback
    when running detached from a tty."""
    env = build_mflux_env(token=None)
    assert "COLUMNS" in env
    assert "LINES" in env
    assert env["COLUMNS"].isdigit()
    assert env["LINES"].isdigit()


# ── format_cmd (python #12 from v0.1.x review — shlex-quoted output) ────


def test_format_cmd_keeps_flag_value_pairs_on_same_line():
    """`--flag value` stays on one line; positional tokens get their own."""
    out = format_cmd(["mflux", "--prompt", "hello", "--steps", "20"])
    lines = out.split(" \\\n  ")
    assert lines[0] == "mflux"
    assert lines[1] == "--prompt hello"
    assert lines[2] == "--steps 20"


def test_format_cmd_quotes_spaces():
    """Values with spaces get shlex.quote — paste-safe single-quote wrap."""
    out = format_cmd(["mflux", "--prompt", "a cat sitting"])
    assert "--prompt 'a cat sitting'" in out


def test_format_cmd_quotes_shell_metacharacters():
    """`$`, backticks, semicolons, newlines must be neutralized — naive
    `"`-only escaping (the pre-v0.3.6 implementation) leaked these
    through verbatim, so pasted output could re-interpret as shell."""
    out = format_cmd(["mflux", "--prompt", "$(rm -rf ~)"])
    # shlex.quote wraps in single quotes — $() is inert inside ''.
    assert "--prompt '$(rm -rf ~)'" in out

    out = format_cmd(["mflux", "--prompt", "`whoami`"])
    assert "--prompt '`whoami`'" in out

    out = format_cmd(["mflux", "--prompt", "a; rm -rf ~"])
    assert "--prompt 'a; rm -rf ~'" in out


def test_format_cmd_handles_embedded_single_quote():
    """shlex.quote escapes `'` itself via the `'\\''` idiom — value
    containing apostrophes still produces a paste-safe single string."""
    out = format_cmd(["mflux", "--prompt", "it's fine"])
    # Standard shlex.quote rendering: 'it'"'"'s fine'
    assert "--prompt 'it'\"'\"'s fine'" in out


def test_format_cmd_does_not_quote_safe_values():
    """Bare alphanumeric / dash / dot values render unchanged — shlex.quote
    is no-op for shell-safe strings, so the common case stays readable."""
    out = format_cmd(["mflux", "--steps", "20", "--seed", "42"])
    # No surrounding quotes added when value is already safe.
    assert "--steps 20" in out
    assert "--seed 42" in out


def test_format_cmd_quotes_positional_tokens_with_spaces():
    """Positional tokens (no leading --) also go through shlex.quote so
    a binary name like `/path with spaces/mflux` is paste-safe too."""
    out = format_cmd(["/path with spaces/mflux", "--steps", "20"])
    assert "'/path with spaces/mflux'" in out

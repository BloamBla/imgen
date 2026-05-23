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


# ── v0.6.2 security NIT-3: $HOME → ~ in rendered cmd + stderr ───────────


def test_format_cmd_rewrites_home_to_tilde(monkeypatch):
    """A local-path argv token containing $HOME renders as ~ in the
    pretty-printed cmd. Defence-in-depth so dry-run output + confirm-
    gate transcripts don't disclose the user's home layout if shared.
    """
    monkeypatch.setenv("HOME", "/Users/imgen-test")
    out = format_cmd([
        "mflux", "--lora-paths",
        "/Users/imgen-test/loras/style.safetensors",
        "--steps", "20",
    ])
    assert "~/loras/style.safetensors" in out
    assert "/Users/imgen-test" not in out


def test_format_cmd_does_not_rewrite_when_home_unset(monkeypatch):
    """An empty $HOME (rare but valid) leaves the cmd unchanged — better
    no rewrite than rewriting random paths."""
    monkeypatch.setenv("HOME", "")
    out = format_cmd([
        "mflux", "--lora-paths", "/some/abs/path/foo.safetensors",
    ])
    assert "/some/abs/path/foo.safetensors" in out


def test_format_cmd_does_not_rewrite_when_home_is_root(monkeypatch):
    """``HOME=/`` would rewrite every absolute path to ``~`` — bypass
    the rewrite entirely in that edge case."""
    monkeypatch.setenv("HOME", "/")
    out = format_cmd(["mflux", "--lora-paths", "/abs/foo.safetensors"])
    assert "/abs/foo.safetensors" in out


def test_stderr_redaction_rewrites_home_to_tilde(monkeypatch, capfdbinary):
    """When mflux happens to log a local-path lora.ref to stderr,
    ``$HOME`` is rewritten to ``~`` in both the terminal output and the
    optional log_file. HF token redaction stays in effect."""
    monkeypatch.setenv("HOME", "/Users/imgen-test")
    leak_line = b"loading lora from /Users/imgen-test/loras/style.safetensors\n"
    run_with_stderr_redaction(
        ["python3", "-c", f"import sys; sys.stderr.buffer.write({leak_line!r}); sys.stderr.buffer.flush()"],
        env={"HOME": "/Users/imgen-test"},
    )
    err = capfdbinary.readouterr().err
    assert b"~/loras/style.safetensors" in err
    assert b"/Users/imgen-test" not in err


# ── v0.4: build_mflux_env(backend_secret=...) ───────────────────────────


def test_build_mflux_env_forwards_backend_secret():
    """Custom backends declare a single env var name; build_mflux_env
    injects it under that name. Distinct from HF_TOKEN so the two
    don't collide on the same FLUX-or-custom invocation."""
    env = build_mflux_env(
        token=None,
        backend_secret=("REPLICATE_API_TOKEN", "r8_abc123"),
    )
    assert env["REPLICATE_API_TOKEN"] == "r8_abc123"
    assert "HF_TOKEN" not in env


def test_build_mflux_env_no_backend_secret_when_none():
    """Without a backend_secret tuple, no extra env var is injected."""
    env = build_mflux_env(token=None, backend_secret=None)
    # No FLUX_TOKEN / REPLICATE_API_TOKEN / etc — only the allowlist + tty.
    assert "REPLICATE_API_TOKEN" not in env
    assert "HF_TOKEN" not in env


def test_build_mflux_env_token_and_secret_coexist():
    """FLUX (token) + a custom backend's secret could theoretically be
    set at once if the caller composes wrong — but the env should
    contain BOTH variables without one overwriting the other, since
    they live in different env keys."""
    env = build_mflux_env(
        token="hf_realtoken123",
        backend_secret=("MY_KEY", "value1"),
    )
    assert env["HF_TOKEN"] == "hf_realtoken123"
    assert env["MY_KEY"] == "value1"


def test_build_mflux_env_backward_compatible_no_kwargs():
    """Pre-v0.4 call sites used `build_mflux_env(token=...)`. With the
    new signature having defaults for both kwargs, the old call style
    still works unchanged. Locked in by every existing caller in
    cmd_helpers / commands."""
    env_with_token = build_mflux_env(token="hf_x")
    env_without_token = build_mflux_env(token=None)
    assert env_with_token["HF_TOKEN"] == "hf_x"
    assert "HF_TOKEN" not in env_without_token


# ── build_enhance_env (v0.5 — minimal env for the enhancer subprocess) ─

from imgen.subprocess_helpers import build_enhance_env


def test_build_enhance_env_includes_hf_cache_keys(monkeypatch):
    """Runner needs HF cache redirection so it finds the already-
    downloaded model. Without these, mlx_lm would silently re-download
    to its own default location."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HF_HOME", "/Volumes/ssd/hf")
    monkeypatch.setenv("HF_HUB_CACHE", "/Volumes/ssd/hf/hub")
    env = build_enhance_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HF_HOME"] == "/Volumes/ssd/hf"
    assert env["HF_HUB_CACHE"] == "/Volumes/ssd/hf/hub"


def test_build_enhance_env_does_not_forward_hf_token(monkeypatch):
    """Key v0.5 security gate (IMP-1): HF_TOKEN must NOT cross into
    the runner subprocess. The default Qwen model is open-license,
    and propagating the user's HF token to a separate subprocess
    that downloads arbitrary models would leak the token via
    huggingface_hub's HTTP error tracebacks on any failure."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_should_not_leak")
    env = build_enhance_env()
    assert "HF_TOKEN" not in env, (
        f"HF_TOKEN leaked into enhance_runner env: {env.get('HF_TOKEN')!r}"
    )


def test_build_enhance_env_does_not_forward_arbitrary_secrets(monkeypatch):
    """Sibling secrets (AWS, GH, etc.) the user's shell may carry are
    also denied. Same allow-list discipline as build_mflux_env."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "AKIA...secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    env = build_enhance_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_build_enhance_env_does_not_forward_terminal_size(monkeypatch):
    """The runner has no TUI output (only structured JSON). COLUMNS /
    LINES are mflux-specific (tqdm renders progress bars there); the
    runner doesn't need them. Don't forward."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "40")
    env = build_enhance_env()
    assert "COLUMNS" not in env
    assert "LINES" not in env


def test_build_enhance_env_omits_keys_not_in_parent(monkeypatch):
    """Missing-from-parent keys aren't synthesised. Returns only what
    exists in the parent's environment, filtered by the allowlist."""
    # Clear every allowlisted key but PATH.
    for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
              "MLX_METAL_PRECOMPILE_PATH", "HOME", "USER", "LANG",
              "LC_ALL", "TMPDIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = build_enhance_env()
    assert env == {"PATH": "/usr/bin"}

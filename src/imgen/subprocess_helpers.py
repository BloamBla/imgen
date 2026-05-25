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

__all__ = [
    "InsufficientRAMError",
    "build_enhance_env",
    "build_mflux_env",
    "format_cmd",
    "run_with_stderr_redaction",
]


# v0.8.2 safety net — hard-floor RAM check below which we refuse to
# spawn ML subprocesses (mflux / diffusers / enhance). Defence-in-depth
# against preflight bypass scenarios (architect HIGH-3 + v0.8.2 ops
# post-mortem):
#
# * ``--force`` flag at CLI entry skips preflight_resources entirely.
# * System state changed between preflight (at CLI entry) and the
#   spawn (later in the iteration loop) — user opened Chrome between.
# * Race between parallel ``imgen`` invocations: ``find_running_mflux``
#   only catches PIDs ALREADY running, not the one about to start.
# * User TOML lying about ``ram_baseline_gb`` post-v0.8.1 schema.
# * External code calling ``Engine.run`` directly (post-M-1B real
#   impl) bypasses the entire CLI layer.
# * Test infrastructure bugs that miss the engine.run path (the
#   v0.8.2 M-1C scare on 2026-05-26 — multiple parallel mflux Popens
#   loaded FLUX weights into memory before the user could Ctrl-C).
#
# 4.0 GB floor: even FLUX Q4 wants ~11 GB peak and ~9 GB baseline; with
# less than 4 GB free, the spawn is guaranteed to swap-thrash or OOM
# the parent shell. Hard refuse > silent attempt.
#
# Escape hatch: ``IMGEN_BYPASS_RAM_FLOOR=1`` skips the check. For CI /
# power users who knowingly accept OOM risk; documented in the error
# message itself so an end user always knows how to opt out.
_MIN_SAFE_AVAILABLE_RAM_GB: float = 4.0


class InsufficientRAMError(RuntimeError):
    """Raised by ``run_with_stderr_redaction`` when available RAM falls
    below ``_MIN_SAFE_AVAILABLE_RAM_GB``. Caller (``run_one_iteration``
    via the ``except`` block) catches and writes a failure history
    entry — no real subprocess ever spawns into a dangerous state.
    """


def _assert_safe_ram_or_raise(
    min_available_gb: float = _MIN_SAFE_AVAILABLE_RAM_GB,
) -> None:
    """Hard-floor RAM check fired BEFORE any ``subprocess.Popen``.

    Pure check + raise pattern. No side effects when the floor is
    satisfied. ``get_memory_gb()`` returning ``(0, 0)`` means parse
    failure (non-Darwin, sysctl unavailable, etc.) — we skip the check
    in that case rather than false-positive on legit CI / Linux smoke
    runs.

    Escape hatch: ``IMGEN_BYPASS_RAM_FLOOR=1`` env var bypasses
    unconditionally. Documented in the error message so an end user
    knows how to override.
    """
    if os.environ.get("IMGEN_BYPASS_RAM_FLOOR") == "1":
        return
    # Local import to dodge the circular: checks.py imports
    # subprocess_helpers's allowlist for its own subprocess work
    # (find_running_mflux uses pgrep via subprocess directly, not
    # through this wrapper, so no cycle in practice — but the late
    # import keeps the dependency graph one-directional in the
    # source).
    from .checks import get_memory_gb
    total_gb, available_gb = get_memory_gb()
    if total_gb == 0.0:
        # parse failure / non-Darwin — let the call proceed without
        # the floor check. Production target is Apple Silicon Macs
        # where get_memory_gb is well-tested; the (0, 0) branch is
        # a defence-in-depth fallback, not an expected path.
        return
    if available_gb < min_available_gb:
        raise InsufficientRAMError(
            f"Refusing to spawn ML subprocess: only "
            f"{available_gb:.1f} GB RAM available "
            f"(safety floor: {min_available_gb:.1f} GB). "
            f"mflux needs ~11 GB peak (Q4) or ~18 GB (Q8) for "
            f"FLUX-class models — spawning now would swap-thrash or "
            f"OOM your machine. Close apps and retry, OR set "
            f"IMGEN_BYPASS_RAM_FLOOR=1 to bypass this check at your "
            f"own OOM risk."
        )


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


def build_mflux_env(
    token: str | None = None,
    backend_secret: tuple[str, str] | None = None,
) -> dict[str, str]:
    """Minimal environment for the mflux subprocess.

    Allow-listed keys from :data:`_MFLUX_ENV_ALLOWLIST` are copied from
    the parent environment; ``HF_TOKEN`` is added when the (FLUX-built-
    in) backend needs gated-model access; ``COLUMNS`` / ``LINES`` are
    forwarded from the host terminal so tqdm renders at the user's
    actual width.

    Shared by ``cmd_generate`` and ``cmd_batch`` (v0.3.0 IMP-5 — the
    two call sites used to inline this block separately, risking the
    allow-list drifting between them on the next edit).

    v0.4: ``backend_secret`` is a ``(env_var_name, value)`` tuple for
    custom backends from ``~/.imgen/backends.d/`` that declared a
    ``[secret] env_var = ...`` field. The pair gets injected under the
    declared name (e.g. ``REPLICATE_API_TOKEN``). Distinct slot from
    ``token`` so an HF-token-bearing FLUX run can't accidentally
    overwrite a custom backend's env var (or vice versa). Caller —
    typically ``cmd_helpers._load_backend_and_token`` — resolves the
    backend's ``secret_env_var`` against ``os.environ`` BEFORE calling
    this, including the required-but-missing die path. This helper
    only knows how to inject a pre-resolved pair.
    """
    env: dict[str, str] = {
        k: os.environ[k] for k in _MFLUX_ENV_ALLOWLIST if k in os.environ
    }
    if token:
        env["HF_TOKEN"] = token
    if backend_secret is not None:
        env_name, env_value = backend_secret
        env[env_name] = env_value
    term = shutil.get_terminal_size(fallback=(80, 24))
    env["COLUMNS"] = str(term.columns)
    env["LINES"] = str(term.lines)
    return env


# Minimal environment for the v0.5 LLM prompt-enhancer subprocess
# (``python -m imgen.enhance_runner``). Same allow-list discipline as
# the mflux env above: explicitly enumerate everything the subprocess
# needs, deny everything else by default. Notably HF_TOKEN is NOT
# forwarded — the default Qwen2.5-7B-Instruct-4bit model is open-
# license. Custom enhance models that need auth should be pre-cached
# out-of-band (``huggingface-cli download``) with HF_HOME pointing at
# the cache. Keeps the runner's network surface to "fetch open models
# only", matching the runner's own minimal-permissions design.
# (v0.5 security-reviewer IMP-1.)
_ENHANCE_ENV_ALLOWLIST: tuple[str, ...] = (
    # Filesystem + locale plumbing the Python interpreter needs.
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
    # HuggingFace cache redirection. The runner needs to find the
    # already-downloaded model; without these the parent's HF cache
    # config doesn't cross the subprocess boundary, so the runner
    # would silently re-download to its own default location.
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    # MLX kernel cache (shared with mflux above).
    "MLX_METAL_PRECOMPILE_PATH",
)


def build_enhance_env() -> dict[str, str]:
    """Minimal environment for the enhance_runner subprocess. Mirrors
    :func:`build_mflux_env`'s allow-list discipline; specifically does
    NOT forward ``HF_TOKEN`` or anything else the user's shell may
    carry (AWS creds, GH tokens, SSH agents, etc.). The runner does
    subprocess JSON I/O + mlx_lm inference, nothing else.

    Terminal size is also NOT forwarded — the runner has no TUI
    output, only structured JSON on stdout."""
    return {
        k: os.environ[k]
        for k in _ENHANCE_ENV_ALLOWLIST
        if k in os.environ
    }


# v0.8.0 commit 6 — env allowlist for the diffusers_mps engine subprocess.
# Per architect commit-6 pre-vet M1: a SIBLING of build_enhance_env (NOT a
# generalised build_engine_env) — matches the project precedent of one
# narrow allowlist per subprocess class. Includes HF_TOKEN because
# DiffusionPipeline.from_pretrained may need auth for gated repos (mflux
# uses ~/.imgen/hf_token via build_mflux_env; diffusers_mps inherits the
# HF_TOKEN env var if the user set it, matching standard HuggingFace
# library expectations). PyTorch MPS fallback flag is set inside the
# runner itself (architect pre-vet M4) BEFORE torch is imported, not
# forwarded from parent env.
_DIFFUSERS_ENV_ALLOWLIST: tuple[str, ...] = (
    # Filesystem + locale plumbing the Python interpreter needs.
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
    # HuggingFace cache redirection — runner must find pre-downloaded
    # weights without re-downloading to its own default location.
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    # HF auth — diffusers' from_pretrained supports gated repos via
    # HF_TOKEN; forward through if the user's shell carries one.
    "HF_TOKEN",
)


def build_diffusers_env() -> dict[str, str]:
    """Minimal environment for the diffusers_mps engine subprocess.

    Sibling of :func:`build_enhance_env` per architect commit-6 pre-vet
    M1 — narrow allowlist, deny-by-default. Adds ``HF_TOKEN`` to the
    enhance allowlist because diffusers' ``from_pretrained`` is the
    canonical gated-repo path on the diffusers side; mflux keeps its
    own ``~/.imgen/hf_token`` plumbing via :func:`build_mflux_env`,
    so the two stay separate.
    """
    return {
        k: os.environ[k]
        for k in _DIFFUSERS_ENV_ALLOWLIST
        if k in os.environ
    }


# Minimum 36 chars after `hf_` so a truncated prefix at a buffer boundary
# (e.g. `hf_AbC\n` flushed via the last-`\r`-or-`\n` rule before the rest
# of the token arrives) can't sneak through as plaintext. Real HF tokens
# are 36+ chars; anything shorter that looks like `hf_` is harmless.
_TOKEN_LEAK_RE = re.compile(rb"hf_[A-Za-z0-9_\-]{36,}")


def _home_path_replacer() -> tuple[bytes, bytes] | None:
    """Return ``(needle_bytes, replacement_bytes)`` for $HOME → ~
    rewriting, or None when $HOME is unsuitable.

    We re-read ``HOME`` per call so tests can monkeypatch it cleanly;
    module-import-time capture would freeze the value. Returns
    bytes pre-encoded so the inner loop can use plain ``bytes.replace``
    on UTF-8 stderr without per-call utf-8 encoding.

    Skipped when:
      * ``$HOME`` unset or empty (nothing to rewrite);
      * ``$HOME = "/"`` (every absolute path would collapse to ``~``);
      * ``$HOME`` is NOT absolute (e.g. ``HOME=tmp`` in a stripped
        container env — naive ``bytes.replace(b"tmp", b"~")`` would
        corrupt every chunk containing the substring ``tmp``, e.g.
        ``tmpdir=/var/tmp`` → ``~dir=/var/~``). (v0.6.2 security IMP-1.)

    Prefix safety: needle includes a trailing ``/`` (and replacement
    is ``~/``) so HOME=``/Users/stan`` does NOT rewrite paths under
    ``/Users/stanislav`` (which would otherwise corrupt to ``~islav``).
    Real macOS / Linux home dirs are always followed by ``/`` before
    the next path component, so the trailing-slash needle still hits
    every realistic case while avoiding the prefix collision.
    (v0.6.2 security IMP-2.)
    """
    home = os.environ.get("HOME", "")
    if not home or home == "/" or not os.path.isabs(home):
        return None
    # Append trailing slash to both needle and replacement so we only
    # match path components that fully end at the home boundary.
    return (home.rstrip("/") + "/").encode("utf-8"), b"~/"


def _redact_chunk(buf: bytes, home_pair: tuple[bytes, bytes] | None) -> bytes:
    """Apply HF-token + optional $HOME → ~ redaction to one stderr chunk.

    Order matters: token redaction first (so a token whose surrounding
    bytes happen to include $HOME doesn't accidentally lose its prefix).
    Both passes are byte-level — no decode/encode roundtrip.
    """
    out = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", buf)
    if home_pair is not None:
        home_bytes, tilde_bytes = home_pair
        out = out.replace(home_bytes, tilde_bytes)
    return out


def run_with_stderr_redaction(
    cmd: list[str],
    env: dict,
    log_file: BinaryIO | None = None,
    *,
    stdin_data: bytes | None = None,
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

    stdin_data (v0.8.0 commit 6): if non-None, bytes to write to the
    child's stdin BEFORE entering the stderr-read loop. Used by the
    diffusers_mps Engine to pass JSON-serialised GenParams + Model
    fields to its static runner module (see
    [[project-v080-design]] §E.1 — locked security-critical pattern;
    static argv, all user data crosses the process boundary as a
    bounded JSON blob, never via -c "<string>" or .format()).

    The write happens BEFORE the stderr-read loop and `proc.stdin` is
    closed immediately after — for ≤64 KB payloads (the design-locked
    upper bound for the diffusers runner) the kernel pipe buffer
    holds the entire write without blocking, so no full-duplex
    deadlock risk. Keyword-only so the legacy positional/log_file
    call sites stay binary-compatible (architect commit-6 pre-vet
    CRITICAL C3 lock-in).

    v0.8.2 safety net: hard-floor RAM check fires FIRST, BEFORE any
    ``subprocess.Popen``. If the system has less than
    ``_MIN_SAFE_AVAILABLE_RAM_GB`` available, raises
    :class:`InsufficientRAMError` — caller (typically
    ``run_one_iteration``) catches and writes a failure history entry.
    See ``_assert_safe_ram_or_raise`` for the rationale + opt-out
    escape hatch.
    """
    # Defence-in-depth pre-spawn RAM check. Catches 6+ preflight-bypass
    # scenarios (see ``_assert_safe_ram_or_raise`` docstring). Raises
    # BEFORE Popen so no ML weights get loaded into a dangerous memory
    # state.
    _assert_safe_ram_or_raise()
    popen_kwargs = dict(env=env, stderr=subprocess.PIPE, bufsize=0)
    if stdin_data is not None:
        popen_kwargs["stdin"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, **popen_kwargs)
    if stdin_data is not None:
        # Write before the stderr-read loop. ≤64KB payloads fit the
        # kernel pipe buffer in one syscall on macOS / Linux (default
        # PIPE_BUF cap is 16 KB but the OS-buffer is 64 KB+), so the
        # write completes synchronously and we can close cleanly.
        # Larger writes would risk full-duplex deadlock — but the
        # runner side rejects oversize stdin with EX_USAGE before
        # reading anything, and the diffusers payload is design-
        # capped at 64 KB (security commit-6 pre-vet bounded-stdin).
        try:
            assert proc.stdin is not None
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        except (OSError, BrokenPipeError):
            # Child died before reading stdin — fall through to the
            # stderr-read loop which will surface its exit code.
            pass
    buffer = b""
    # v0.6.x backlog security NIT-3 defence-in-depth: also rewrite
    # ``$HOME`` → ``~`` in the streamed stderr so any local-path lora.ref
    # (or ``--prompt-file PATH``, or any other path mflux happens to log)
    # doesn't disclose ``$HOME`` to anyone the user later shares the
    # batch log with. Token redaction stays the primary defence; this is
    # a cheap secondary scrub on the same byte path.
    home_pair = _home_path_replacer()
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
                    redacted = _redact_chunk(buffer, home_pair)
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
                redacted = _redact_chunk(to_flush, home_pair)
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

    v0.6.2: ``$HOME`` is rewritten to ``~`` in the rendered string —
    matches the same defence-in-depth scrub applied to mflux stderr.
    A local-path ``--lora /Users/me/loras/foo.safetensors`` renders as
    ``~/loras/foo.safetensors`` so dry-run output + confirm-gate
    transcripts don't disclose the user's home layout when shared.

    Privacy-vs-discoverability trade-off (architect v0.6.2 NIT-3): the
    rewrite means a recipient of a shared dry-run transcript sees
    ``~/loras/foo.safetensors`` and may expand ``~`` to their own
    ``$HOME``, leading to "file not found" if they try to literally
    re-run the command. The rewrite optimises for "don't disclose
    paths" over "command is literally re-runnable on another machine".
    Tokens use the same trade-off (``hf_***REDACTED***`` is not
    re-runnable either) and that's the right side of the curve when
    the alternative is leaking secrets / paths into a chat transcript.
    The README LoRA section spells this out for users sharing logs.
    (v0.6.x backlog security NIT-3.)
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
    rendered = " \\\n  ".join(parts)
    # v0.6.2 security IMP-1+IMP-2: mirror _home_path_replacer's
    # guards (absolute-path required + trailing-slash needle to avoid
    # prefix collisions like HOME=/Users/stan rewriting /Users/stanislav).
    home = os.environ.get("HOME", "")
    if home and home != "/" and os.path.isabs(home):
        needle = home.rstrip("/") + "/"
        if needle in rendered:
            rendered = rendered.replace(needle, "~/")
    return rendered

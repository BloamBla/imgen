"""LLM prompt enhancer — subprocess runner + iteration orchestrator.

v0.7.16 split: this module owns the **impure runtime layer** of the
enhancer (`RunnerError`, wire-protocol JSON encode/decode, mlx_lm
subprocess spawn, argv prompt patching, batch orchestration). The
pure decision layer lives in `enhance_decide.py`; the public
`imgen.enhance` namespace re-exports both for backward-compat.

The runtime/decision split keeps the decision logic 100% testable
without GPU, model weights, or even mlx_lm installed — see
`enhance_decide.py` for the pure functions. This file covers the
seams that DO touch subprocess + filesystem.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .enhance_decide import (
    DEFAULT_ENHANCE_TIMEOUT_S,
    EnhanceResult,
    decide_final_prompt,
    is_enhanceable,
)

__all__ = [
    "RunnerError",
    "build_runner_payload",
    "enhance_iteration_prompts",
    "parse_runner_response",
    "replace_prompt_in_cmd",
    "run_with_mlx_lm",
]


# ── Subprocess runner — wire protocol + spawn-and-read wrapper ──────────


class RunnerError(Exception):
    """Raised by :func:`run_with_mlx_lm` when the enhance_runner subprocess
    cannot fulfil a batch — non-zero exit, crash, timeout, malformed
    JSON, or count mismatch. Caller is expected to catch and fall back
    to text-only generation for the entire batch (per-iteration fallback
    is handled inside ``decide_final_prompt`` for individual LLM outputs,
    not here — this exception means "the runner itself failed").
    """


def build_runner_payload(
    *,
    items: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Construct the JSON payload fed to enhance_runner via stdin.

    ``items`` is a list of ``{"system": "...", "user": "..."}`` dicts
    aligned with the prompts that need enhancement. The runner returns
    a list of equal length in the same order — callers must keep that
    alignment to splice results back into the iteration plan.

    Returns a plain dict, JSON-serialisable. Lock-in test guards
    against accidental non-JSON types creeping in (a Path, an Enum)
    that would crash json.dumps at the wire boundary.
    """
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "items": items,
    }


def parse_runner_response(raw: str, *, expected_count: int) -> list[str]:
    """Decode the runner's stdout JSON into a list of output strings.

    Raises :class:`RunnerError` on:
      * malformed JSON (runner crashed mid-write / mixed stdout streams)
      * top-level ``{"error": "..."}`` field (runner reported a clean
        failure — model not found, mlx_lm OOM, etc.)
      * missing ``results`` key (wire protocol violation)
      * item in ``results`` missing ``output`` key
      * results count != ``expected_count`` (mlx_lm dropped or
        duplicated outputs — abort the whole batch rather than
        produce mis-aligned enhancements)
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RunnerError(f"invalid JSON from enhance_runner: {e}") from e
    if "error" in data:
        raise RunnerError(f"enhance_runner reported error: {data['error']}")
    if "results" not in data:
        raise RunnerError(
            "enhance_runner response missing 'results' top-level key"
        )
    results = data["results"]
    if not isinstance(results, list):
        raise RunnerError(
            f"enhance_runner 'results' must be a list, got {type(results).__name__}"
        )
    if len(results) != expected_count:
        raise RunnerError(
            f"enhance_runner expected {expected_count} items, got {len(results)} "
            "(refusing to mis-align outputs to iterations)"
        )
    outputs: list[str] = []
    for i, item in enumerate(results):
        if not isinstance(item, dict) or "output" not in item:
            raise RunnerError(
                f"enhance_runner result item {i} missing 'output' field"
            )
        outputs.append(str(item["output"]))
    return outputs


def run_with_mlx_lm(
    *,
    items: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float = DEFAULT_ENHANCE_TIMEOUT_S,
    python_executable: str | None = None,
    runner_script: str | Path | None = None,
) -> list[str]:
    """Spawn ``python -m imgen.enhance_runner``, send ``items`` as JSON
    stdin, return ``[output_str_for_item_0, ...]``.

    One subprocess invocation handles the WHOLE batch — mlx_lm.load
    happens once, all generations share the loaded model, weights are
    freed when the runner exits. This amortises the ~5-10 s cold-cache
    load over N iterations.

    ``timeout`` is wall-clock seconds; on expiry the subprocess is
    killed and :class:`RunnerError` raised. Default 120 s is generous
    for a small batch but still fires if mlx_lm hangs.

    Test seam: ``python_executable`` + ``runner_script`` can be
    overridden to point at a fake runner script. Production callers
    pass nothing → defaults to ``sys.executable`` + the real
    ``imgen.enhance_runner`` module via ``-m``.

    Not user-controllable by design (v0.5 security NIT-3): neither
    kwarg is exposed via CLI or config. Both values flow into a
    ``subprocess.run`` call that uses ``cmd`` as a list (no shell) and
    a constrained env dict (no PATH interpolation of relative names).
    Test code that overrides these passes paths that already exist on
    disk; we never quote / format them into a shell command. Don't add
    a CLI flag or config field that lets external input set either
    without re-auditing the subprocess call shape.

    Empty ``items`` list short-circuits — no subprocess at all. This
    matters because an enhance-enabled run with all iterations having
    empty prompts would otherwise pay the LLM-load cost for zero work.
    """
    if not items:
        return []

    py = python_executable or sys.executable
    if runner_script is not None:
        cmd = [py, str(runner_script)]
    else:
        cmd = [py, "-m", "imgen.enhance_runner"]

    payload = build_runner_payload(
        items=items,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    payload_json = json.dumps(payload, ensure_ascii=False)

    # v0.5 security-reviewer IMP-1: explicit minimal env. Without this
    # subprocess.run inherits the full parent environment — including
    # HF_TOKEN, AWS credentials, GH tokens, etc. that the runner has no
    # business seeing. Mirrors the build_mflux_env discipline applied to
    # mflux subprocess calls. build_enhance_env() forwards only PATH,
    # HOME, USER, locale, TMPDIR, and HF cache paths — and explicitly
    # does NOT forward HF_TOKEN.
    from .subprocess_helpers import (
        _assert_safe_ram_or_raise,
        build_enhance_env,
    )
    runner_env = build_enhance_env()

    # v0.8.2 safety net (M-NEW-A from §R.4 v0.8.2 review): the enhance
    # subprocess loads Qwen2.5-7B (~4 GB). It's the smaller of the
    # project's ML children, but loading any 4+ GB model into <4 GB
    # available RAM still swap-thrashes. Run the same hard-floor check
    # the mflux + diffusers subprocesses go through via
    # ``run_with_stderr_redaction`` — we use ``subprocess.run`` here
    # (synchronous + small payload), so the assert lives at the call
    # site, NOT inside the run_with_stderr_redaction wrapper.
    # ``InsufficientRAMError`` propagates up; the orchestrator
    # (cmd_helpers.maybe_enhance_prompts) wraps it as a RunnerError
    # via EnhanceResult.fallback_reason="runner_error" so the user
    # sees the cause + the fallback runs with the original prompt.
    _assert_safe_ram_or_raise()

    try:
        result = subprocess.run(
            cmd,
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # we inspect returncode ourselves for diagnostics
            env=runner_env,
        )
    except subprocess.TimeoutExpired as e:
        raise RunnerError(
            f"enhance_runner timed out after {timeout}s "
            "(LLM may be stuck or model load is taking unusually long)"
        ) from e
    except OSError as e:
        raise RunnerError(f"enhance_runner spawn failed: {e}") from e

    if result.returncode != 0 and not result.stdout.strip():
        # Hard crash without JSON output (segfault, OOM-kill, exec
        # failure, etc.) — no parse possible, surface what we have.
        stderr_tail = (result.stderr or "").strip()[-500:]
        raise RunnerError(
            f"enhance_runner exited {result.returncode} without producing "
            f"output. stderr tail: {stderr_tail!r}"
        )

    if result.returncode == 0 and not result.stdout.strip():
        # Defensive (v0.5 python I-3): clean exit but empty stdout.
        # Shouldn't be reachable — enhance_runner always emits JSON on
        # success — but a future bug there would otherwise tumble into
        # parse_runner_response → JSONDecodeError with a confusing
        # "Expecting value" trace. Surface a clean RunnerError instead
        # so the orchestrator can fall back per-item with
        # ``fallback_reason="runner_error"``.
        stderr_tail = (result.stderr or "").strip()[-500:]
        raise RunnerError(
            "enhance_runner exited 0 with no stdout — runner contract "
            "violated. stderr tail: " + repr(stderr_tail)
        )

    return parse_runner_response(result.stdout, expected_count=len(items))


# ── Iteration-level orchestrator (used by cmd_generate / cmd_batch) ─────


def replace_prompt_in_cmd(cmd: list[str], new_prompt: str) -> list[str]:
    """Return a new mflux argv with the ``--prompt`` value swapped.

    Iteration.cmd is built in :func:`backends.build_mflux_cmd` with
    ``--prompt`` followed by the prompt string at a fixed position.
    When the enhancer modifies the prompt post-construction, we patch
    the argv rather than re-running build_mflux_cmd (which would
    require carrying every keyword arg through the Iteration). Pure
    function — never mutates ``cmd``.

    If ``cmd`` does not contain ``--prompt`` (which would be a bug in
    build_mflux_cmd, not a normal path), the input is returned
    unchanged. Defensive: enhance should never crash the generation
    flow, just degrade silently to no-op.
    """
    out = list(cmd)
    try:
        i = out.index("--prompt")
    except ValueError:
        return out  # no --prompt to replace; defensive no-op
    if i + 1 >= len(out):
        return out  # malformed argv; defensive no-op
    out[i + 1] = new_prompt
    return out


def enhance_iteration_prompts(
    *,
    iteration_prompts: list[str],
    system_prompt: str | None,
    invariants: tuple[str, ...],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float = DEFAULT_ENHANCE_TIMEOUT_S,
    run_llm: Callable[..., list[str]] = run_with_mlx_lm,
) -> list[EnhanceResult]:
    """Top-level orchestrator: enhance a batch of iteration prompts.

    One subprocess launch handles the whole batch — mlx_lm.load runs
    ONCE, all N prompts share the loaded model. This amortises the
    cold-cache load over an N-style or N×M-batch invocation.

    Per-prompt fallback paths (all return aligned ``EnhanceResult``
    entries; the returned list always has ``len(iteration_prompts)``):

    * ``system_prompt is None`` (backend has no enhancer config) →
      every result is ``fallback_reason="not_supported_by_backend"``
    * a prompt fails :func:`should_enhance` (empty / too long) →
      that one gets ``"empty_input"`` or ``"input_too_long"``,
      others continue through the LLM
    * the runner subprocess raises :class:`RunnerError` (model load
      failed, timeout, crash) → ALL results get
      ``fallback_reason="runner_error"`` with the error preserved
      in ``fallback_detail`` (v0.6.5; v0.6.0–v0.6.4 stuffed it into
      ``raw_llm_output``, but that field's contract is "what the LLM
      emitted" — a runner crash means no LLM output at all). We don't
      want a partial enhancement across a multi-style run.

    ``run_llm`` is an injection seam for tests — pass a callable with
    the same signature as :func:`run_with_mlx_lm` to skip the real
    subprocess.

    Note on the ``timeout_s`` parameter (v0.5 python N-2): this
    function's public kwarg is ``timeout_s`` (seconds, float). Internally
    it's forwarded to :func:`run_with_mlx_lm` (or the injected ``run_llm``)
    via the ``timeout=`` kwarg (the underlying signature uses ``timeout``,
    not ``timeout_s``). The orchestrator's ``_s`` suffix makes the unit
    obvious at the API boundary; the runner's plain ``timeout`` matches
    Python's stdlib subprocess timeout convention.
    """
    n = len(iteration_prompts)

    # Backend has no system prompt → enhancer not wired for this
    # backend. Fail-safe: don't try to call LLM with a generic
    # instruction that might shape the prompt wrong.
    if system_prompt is None:
        return [
            EnhanceResult(
                final_prompt=p,
                original_prompt=p,
                was_enhanced=False,
                fallback_reason="not_supported_by_backend",
                was_truncated=False,
                raw_llm_output=None,
            )
            for p in iteration_prompts
        ]

    # Decide per-prompt whether it's enhancement-eligible. We're
    # already inside the "enhance is on" code path so the high-level
    # should_enhance gate is redundant — use is_enhanceable directly
    # for the per-prompt content check. (v0.5 architect NIT #1.)
    #
    # v0.6.4 python N-4: ``pre_results`` is a ``dict[int, EnhanceResult]``
    # keyed by iteration index. The v0.5 shape was
    # ``list[EnhanceResult | None]`` filled lazily, which forced two
    # ``# type: ignore[misc]`` filter-comprehensions to drop the None
    # sentinels at return time. The dict shape encodes "slots assigned
    # so far" precisely and lets the final return be a clean ``[d[i]
    # for i in range(n)]`` — no Optional in the type, no type-ignore.
    enhanceable: list[int] = []  # indices of prompts to pass to LLM
    pre_results: dict[int, EnhanceResult] = {}
    for i, p in enumerate(iteration_prompts):
        if not is_enhanceable(p):
            # Stamp the right reason based on which gate tripped.
            if not p.strip():
                reason = "empty_input"
            else:
                reason = "input_too_long"
            pre_results[i] = EnhanceResult(
                final_prompt=p,
                original_prompt=p,
                was_enhanced=False,
                fallback_reason=reason,
                was_truncated=False,
                raw_llm_output=None,
            )
        else:
            enhanceable.append(i)

    # If nothing to enhance, skip the subprocess entirely.
    if not enhanceable:
        # Every iteration index was filled by the pre-result loop above
        # (since none were enhanceable, all hit one of the fallback
        # branches). Iterate in index order to preserve caller alignment.
        return [pre_results[i] for i in range(n)]

    items = [
        {"system": system_prompt, "user": iteration_prompts[i]}
        for i in enhanceable
    ]

    try:
        llm_outputs = run_llm(
            items=items,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
    except RunnerError as e:
        # All-or-nothing fallback: a runner-level failure (model load,
        # crash, timeout) returns ``runner_error`` for every iteration
        # so the user sees consistent behaviour across the batch.
        #
        # v0.6.5 (architect IMP-1): the error message lands in
        # ``fallback_detail``, not ``raw_llm_output``. ``raw_llm_output``
        # means "what the LLM emitted" by contract; the runner crashing
        # before producing output means there is no LLM output. The
        # diagnostic string (model-load trace, timeout, etc.) belongs in
        # the verbose-detail field alongside the ``invariant_violated``
        # reason — symmetric handling across the two paths whose coarse
        # ``fallback_reason`` token loses information without the detail.
        msg = str(e)
        return [
            EnhanceResult(
                final_prompt=p,
                original_prompt=p,
                was_enhanced=False,
                fallback_reason="runner_error",
                was_truncated=False,
                raw_llm_output=None,
                fallback_detail=msg,
            )
            for p in iteration_prompts
        ]

    # Stitch LLM outputs back into the result dict at the correct slots.
    for slot_idx, llm_output in zip(enhanceable, llm_outputs):
        pre_results[slot_idx] = decide_final_prompt(
            original=iteration_prompts[slot_idx],
            enhanced_or_none=llm_output,
            invariants=invariants,
        )

    # Every slot 0..n-1 is filled by now (the pre-loop covered the
    # non-enhanceable ones, the LLM loop just covered the rest). Caller-
    # alignment depends on index order, not dict iteration order.
    return [pre_results[i] for i in range(n)]

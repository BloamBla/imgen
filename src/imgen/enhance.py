"""LLM prompt enhancer (v0.5) — pure-function decision layer.

The enhancer takes the already-constructed Kontext / Qwen-Edit prompt
that imgen would have sent to mflux, asks a local MLX LLM (Qwen2.5-7B
by default) to expand it into a richer, model-tuned version, and feeds
the result to mflux instead. Opt-in via ``--enhance-prompt``.

This module owns the **pure decision logic**:

* When NOT to enhance (disabled, empty input, oversized input).
* How to format messages for the LLM.
* How to sanitise the LLM's response (strip quote-wrapping, trim).
* How to validate the response against per-backend invariants (must
  keep ``preserving …`` clauses for Kontext / Qwen-Edit so we don't
  silently lose identity anchoring).
* How to cap output length without producing invalid UTF-8.
* How to choose the final prompt (enhanced or fallback to original).

The **impure runner** that actually loads mlx_lm + invokes Qwen lives
separately (``run_with_mlx_lm`` — v0.5 Phase C). Splitting the seam
keeps the decision logic 100% testable without GPU, model weights, or
even mlx_lm installed.

Project rules:

* Frozen+slots dataclass (consistent with Iteration / BatchContext /
  EnhanceResult — v0.2.5 review pattern).
* Explicit ``__hash__ = None`` on the result (same rationale).
* Fail-safe: invariant violation / empty LLM output / oversized input
  all fall back to the ORIGINAL prompt with a diagnostic reason
  string. We never crash a generation because enhancement failed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_ENHANCE_MODEL",
    "DEFAULT_ENHANCE_TIMEOUT_S",
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "EnhanceResult",
    "RunnerError",
    "apply_length_cap",
    "build_messages",
    "build_runner_payload",
    "check_invariants",
    "decide_final_prompt",
    "enhance_iteration_prompts",
    "extract_enhanced_text",
    "parse_runner_response",
    "replace_prompt_in_cmd",
    "run_with_mlx_lm",
    "is_enhanceable",
    "should_enhance",
]


# Default model — mlx-community ships a 4-bit pre-quantized MLX-native
# Qwen2.5-7B-Instruct that matches our memory budget on M2 Pro 32 GB
# (~4.3 GB live alongside FLUX-Kontext's ~18 GB live = comfortable).
# User-overridable via config.toml [enhance] model = "..." or
# --enhance-model REF on the CLI.
DEFAULT_ENHANCE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# Generous timeout: 120 s gives the runner enough room to load weights
# (~10 s cold cache) + generate up to ~30 prompts in a batch (~3 s each)
# even on a thermally-throttled M2 Pro. A timeout still fires if mlx_lm
# hangs (memory thrash, kernel panic recovery).
DEFAULT_ENHANCE_TIMEOUT_S = 120


# Input cap: ~2 KB. The LLM-expanded version sits below 60 KB (mflux's
# argv-len budget); inputs larger than ~2 KB are almost certainly
# already a literary text from --prompt-file and don't benefit from
# further LLM expansion. Skip-with-warn instead of letting Qwen
# truncate-and-confuse.
DEFAULT_MAX_INPUT_BYTES = 2048

# Output cap: 60 KB. mflux argv (POSIX ARG_MAX-bound) lives around
# 256 KB-1 MB depending on platform; 60 KB keeps comfortable headroom
# alongside the rest of the cmd. Truncation policy below trims at a
# UTF-8 char boundary so we never emit invalid bytes.
DEFAULT_MAX_OUTPUT_BYTES = 60_000


@dataclass(frozen=True, slots=True)
class EnhanceResult:
    """Outcome of an enhancement attempt — what goes to mflux + history.

    ``final_prompt`` is what callers must use (always non-empty if
    ``original_prompt`` was non-empty). ``original_prompt`` is the
    pre-LLM constructed prompt — kept verbatim so the v=2 history
    writer has both pre and post versions of every iteration without
    needing a parallel list aligned with the iteration index (the
    parallel-list approach v0.5 Phase C-1 shipped was fragile: any
    reordering of "capture originals" vs "splice enhanced back into
    iterations" silently corrupted history). ``was_enhanced`` is True
    when the LLM's expansion survived all checks and is in
    ``final_prompt``; False means we fell back to original.
    ``fallback_reason`` explains why fallback happened (or is None if
    no fallback). ``raw_llm_output`` is kept for history / debug —
    None when the LLM was never invoked (disabled / too-long input).
    """
    final_prompt: str
    original_prompt: str
    was_enhanced: bool
    fallback_reason: str | None
    was_truncated: bool
    raw_llm_output: str | None

    # Match v0.2.5 review precedent (Iteration, BatchContext): explicit
    # ``__hash__ = None`` to opt frozen-slots dataclasses out of set/
    # dict-key use. EnhanceResult is a one-shot record, not a key.
    __hash__ = None  # type: ignore[assignment]


# ── Decision predicates ─────────────────────────────────────────────────


def is_enhanceable(
    prompt: str,
    *,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> bool:
    """Pure content check: does this prompt's text allow enhancement?

    Returns False if the prompt is empty / whitespace-only or exceeds
    ``max_input_bytes`` after UTF-8 encoding. Says nothing about the
    feature-enabled flag — call sites that already know enhancement is
    on can use this directly; everywhere else should use
    :func:`should_enhance` which also gates on ``enabled``.

    The byte cap (not char cap) matches what tokenisers see — Cyrillic
    inputs are 2x size, and we want consistent behaviour for users
    typing in both RU and EN.

    Split out of :func:`should_enhance` in v0.6.4 per the v0.5 architect
    NIT #1 — the orchestrator's ``should_enhance(p, enabled=True)``
    hardcoding the ``True`` read oddly. The orchestrator now calls
    :func:`is_enhanceable` directly; the public ``should_enhance``
    stays as the high-level "should we enhance this at all?" gate.
    """
    if not prompt.strip():
        return False
    if len(prompt.encode("utf-8")) > max_input_bytes:
        return False
    return True


def should_enhance(
    prompt: str,
    *,
    enabled: bool,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> bool:
    """Decide whether to invoke the LLM for this prompt.

    Returns False (skip) if:
      * the feature is disabled (``enabled=False``)
      * the prompt is empty or whitespace-only (nothing to expand)
      * the prompt exceeds ``max_input_bytes`` after UTF-8 encoding
        (too big for the LLM context window to add value)

    The empty / too-long checks delegate to :func:`is_enhanceable`
    (v0.6.4 split). External callers use this function as the high-
    level "should we enhance this at all?" gate; internal call sites
    that have already committed to the enhance path use
    :func:`is_enhanceable` directly.
    """
    if not enabled:
        return False
    return is_enhanceable(prompt, max_input_bytes=max_input_bytes)


# ── Message formatting ──────────────────────────────────────────────────


def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Construct the chat-format messages payload for mlx_lm.

    Standard 2-message shape: system instruction + user content. Returns
    a fresh list each call so callers can mutate it for their own needs
    (e.g. injecting few-shot examples) without poisoning a shared list.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ── LLM-output sanitisation ─────────────────────────────────────────────


def extract_enhanced_text(raw: str) -> str:
    """Strip the wrapping the LLM sometimes adds despite "output only" hints.

    Handles two common patterns:
      * Outer matching quotes: ``"foo bar"`` or ``'foo bar'`` → ``foo bar``
      * Leading/trailing whitespace and newlines → trimmed

    Does NOT strip mid-sentence quotes (``say "hi" to me``) — only
    when BOTH the first and last non-whitespace chars are the same
    quote character. Empty or whitespace-only input returns "".
    """
    text = raw.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


# ── Invariant check ─────────────────────────────────────────────────────


def check_invariants(
    enhanced: str,
    original: str,
    invariants: tuple[str, ...],
) -> tuple[bool, str | None]:
    """Verify each invariant substring that appears in ``original`` also
    appears in ``enhanced`` (case-insensitive).

    Returns ``(True, None)`` if all good; ``(False, reason)`` with a
    human-readable description if any invariant present in original was
    dropped from enhanced. Invariants NOT present in original aren't
    enforced — the user typed something without the anchor clause, fine.

    Case-insensitive on purpose: the LLM might capitalise "Preserving"
    where original had "preserving" or vice-versa; that's not drift
    we want to penalise.
    """
    if not invariants:
        return True, None
    e_lower = enhanced.lower()
    o_lower = original.lower()
    missing: list[str] = []
    for inv in invariants:
        inv_lower = inv.lower()
        if inv_lower in o_lower and inv_lower not in e_lower:
            missing.append(inv)
    if missing:
        return False, (
            f"enhanced output dropped invariant clause(s): {', '.join(missing)!r} "
            "— these were present in the original prompt and must survive "
            "enhancement to preserve subject anchoring"
        )
    return True, None


# ── Length cap ──────────────────────────────────────────────────────────


def apply_length_cap(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate ``text`` so its UTF-8 encoding fits in ``max_bytes``.

    Returns ``(possibly-truncated-text, was_truncated)``. Truncation
    respects UTF-8 char boundaries — we always emit valid UTF-8, never
    cut mid-codepoint (which could produce invalid bytes that downstream
    JSON / argv handling would choke on).

    Strategy: encode → slice bytes → decode with errors='ignore'. The
    errors='ignore' drops the trailing partial codepoint cleanly.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    truncated_bytes = encoded[:max_bytes]
    truncated_text = truncated_bytes.decode("utf-8", errors="ignore")
    return truncated_text, True


# ── Orchestrator ────────────────────────────────────────────────────────


def decide_final_prompt(
    *,
    original: str,
    enhanced_or_none: str | None,
    invariants: tuple[str, ...],
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    disabled_reason: str | None = None,
) -> EnhanceResult:
    """Choose between enhanced and original, with all fail-safe checks.

    Args:
        original: The pre-enhancement prompt (always our fallback target).
        enhanced_or_none: The LLM's raw output, or None if we never
            invoked the LLM (disabled / input too long). Pass None +
            ``disabled_reason="…"`` to record the skip cleanly in
            ``EnhanceResult.fallback_reason``.
        invariants: Substrings that must survive enhancement (per-backend
            via ``Backend.enhance_invariants``). Usually
            ``("preserving",)`` for FLUX-Kontext / Qwen-Edit.
        max_output_bytes: Cap on final prompt UTF-8 size. Defaults to
            60 KB — comfortable headroom under typical ARG_MAX.
        disabled_reason: When ``enhanced_or_none is None``, the reason
            stamped into ``fallback_reason``. Common values:
            ``"user_opt_out"``, ``"input_too_long"``, ``"empty_input"``.

    Returns an EnhanceResult with the chosen ``final_prompt`` plus
    diagnostic flags. Never raises — even the most pathological LLM
    output ends with a clean fallback to ``original``.
    """
    # Path 1: LLM was never invoked. Return original verbatim.
    if enhanced_or_none is None:
        return EnhanceResult(
            final_prompt=original,
            original_prompt=original,
            was_enhanced=False,
            fallback_reason=disabled_reason,
            was_truncated=False,
            raw_llm_output=None,
        )

    raw = enhanced_or_none
    cleaned = extract_enhanced_text(raw)

    # Path 2: LLM returned empty / whitespace-only output. Fallback.
    if not cleaned:
        return EnhanceResult(
            final_prompt=original,
            original_prompt=original,
            was_enhanced=False,
            fallback_reason="empty_llm_output",
            was_truncated=False,
            raw_llm_output=raw,
        )

    # Path 3: Length cap. Truncate first; THEN check invariants on the
    # truncated text — if truncation chopped the preserving-clause off
    # the end, that's an invariant violation and we fall back.
    capped, was_truncated = apply_length_cap(cleaned, max_output_bytes)

    # Path 4: Invariant check.
    ok, reason = check_invariants(capped, original, invariants)
    if not ok:
        return EnhanceResult(
            final_prompt=original,
            original_prompt=original,
            was_enhanced=False,
            fallback_reason="invariant_violated",
            was_truncated=False,
            raw_llm_output=raw,
        )

    # Path 5: Valid enhancement — return it (possibly truncated).
    return EnhanceResult(
        final_prompt=capped,
        original_prompt=original,
        was_enhanced=True,
        fallback_reason=None,
        was_truncated=was_truncated,
        raw_llm_output=raw,
    )


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
    from .subprocess_helpers import build_enhance_env
    runner_env = build_enhance_env()

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
      in ``raw_llm_output``. We don't want a partial enhancement
      across a multi-style run.

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
        msg = str(e)
        return [
            EnhanceResult(
                final_prompt=p,
                original_prompt=p,
                was_enhanced=False,
                fallback_reason="runner_error",
                was_truncated=False,
                raw_llm_output=msg,
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

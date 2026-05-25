"""LLM prompt enhancer — pure-function decision layer.

v0.7.16 split: this module owns the **pure decision logic** of the
enhancer (`EnhanceResult` dataclass, content predicates, message
formatting, LLM-output sanitisation + invariant checking, length cap,
fallback orchestration). The impure subprocess runner + iteration-level
orchestrator live in `enhance_runtime.py`; the public `imgen.enhance`
namespace re-exports both for backward-compat.

Pre-v0.7.16 this content lived in a 784-LoC `enhance.py` that was above
the CLAUDE.md soft cap. Split along the pure-vs-impure seam (matching
the v0.5 architect's "pure decision layer" framing in the original
module docstring): 100% testable without GPU / model weights / even
mlx_lm installed.

Project rules:

* Frozen+slots dataclass (consistent with Iteration / BatchContext /
  EnhanceResult — v0.2.5 review pattern).
* Explicit ``__hash__ = None`` on the result (same rationale).
* Fail-safe: invariant violation / empty LLM output / oversized input
  all fall back to the ORIGINAL prompt with a diagnostic reason
  string. We never crash a generation because enhancement failed.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_ENHANCE_MODEL",
    "DEFAULT_ENHANCE_TIMEOUT_S",
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "EnhanceResult",
    "apply_length_cap",
    "build_messages",
    "check_invariants",
    "decide_final_prompt",
    "extract_enhanced_text",
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
    no fallback) — a coarse token (``"empty_input"`` / ``"input_too
    _long"`` / ``"empty_llm_output"`` / ``"invariant_violated"`` /
    ``"runner_error"`` / ``"user_opt_out"`` / ``"not_supported_by_
    backend"``). ``fallback_detail`` (v0.6.4) carries the verbose
    diagnostic string when the coarse token doesn't capture the full
    why — populated for ``invariant_violated`` (the ``check_invariants``
    reason naming the dropped clause(s)) and, since v0.6.5, also for
    ``runner_error`` (the ``str(RunnerError)`` message — model-load
    trace, timeout, etc.). None elsewhere. v0.5 python I-4 / v0.6.5
    architect IMP-1: the coarse token alone was useful for aggregation
    but lost the actual violation / crash detail, making the history
    hard to debug after the fact.

    Pre-v0.6.5 the runner-error message lived in ``raw_llm_output``
    instead — a contract mismatch (``raw_llm_output`` means "what the
    LLM emitted"; a runner crash means no LLM output at all). Old v=3
    history entries written by v0.6.0–v0.6.4 carry the message in
    ``raw_llm_output`` with ``fallback_detail`` absent or null. No
    code path reads these fields back today (replay restores nothing
    from the enhance block); the docstring note is here so future
    ``imgen history --verbose`` work prefers ``fallback_detail`` first
    and falls back to ``raw_llm_output`` only when the row's
    ``fallback_reason == "runner_error"`` and ``fallback_detail`` is
    null.

    Security boundary on ``fallback_detail`` (v0.6.4 security IMP-1):
    the field may transitively carry user-supplied strings — built-
    in backend invariants are project-controlled, but user backend
    TOMLs declared in ``~/.imgen/backends.d/`` can set arbitrary
    ``enhance_invariants`` substrings that end up in the
    ``check_invariants`` reason string. Today this field is written
    only to ``~/.imgen/history.jsonl`` and never rendered to a
    terminal / log, so control-byte / escape-injection display
    hazards don't apply. If a future surface DOES render
    ``fallback_detail`` directly (e.g. an ``imgen history --verbose``
    flag), pass it through ``repr()`` or an equivalent control-byte
    escape — same discipline as the v0.4 ``warn(repr(...))`` pattern
    on user-supplied alias paths in doctor.

    ``raw_llm_output`` is kept for history / debug — None when the LLM
    was never invoked (disabled / too-long input).
    """
    final_prompt: str
    original_prompt: str
    was_enhanced: bool
    fallback_reason: str | None
    was_truncated: bool
    raw_llm_output: str | None
    fallback_detail: str | None = None

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
            # v0.5 python I-4: pass the verbose check_invariants reason
            # (which named clause(s) the LLM dropped) so history /
            # debug surfaces have more than just the "invariant_
            # violated" coarse token to work with.
            fallback_detail=reason,
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

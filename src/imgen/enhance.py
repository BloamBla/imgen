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

from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "EnhanceResult",
    "apply_length_cap",
    "build_messages",
    "check_invariants",
    "decide_final_prompt",
    "extract_enhanced_text",
    "should_enhance",
]


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
    ``original`` was non-empty). ``was_enhanced`` is True when the
    LLM's expansion survived all checks and is in ``final_prompt``;
    False means we fell back to ``original``. ``fallback_reason``
    explains why fallback happened (or is None if no fallback).
    ``raw_llm_output`` is kept for history / debug — None when the
    LLM was never invoked (disabled / too-long input).
    """
    final_prompt: str
    was_enhanced: bool
    fallback_reason: str | None
    was_truncated: bool
    raw_llm_output: str | None

    # Match v0.2.5 review precedent (Iteration, BatchContext): explicit
    # ``__hash__ = None`` to opt frozen-slots dataclasses out of set/
    # dict-key use. EnhanceResult is a one-shot record, not a key.
    __hash__ = None  # type: ignore[assignment]


# ── Decision predicates ─────────────────────────────────────────────────


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

    The byte cap (not char cap) matches what tokenisers see — Cyrillic
    inputs are 2x size, and we want consistent behaviour for users
    typing in both RU and EN.
    """
    if not enabled:
        return False
    if not prompt.strip():
        return False
    if len(prompt.encode("utf-8")) > max_input_bytes:
        return False
    return True


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
            was_enhanced=False,
            fallback_reason="invariant_violated",
            was_truncated=False,
            raw_llm_output=raw,
        )

    # Path 5: Valid enhancement — return it (possibly truncated).
    return EnhanceResult(
        final_prompt=capped,
        was_enhanced=True,
        fallback_reason=None,
        was_truncated=was_truncated,
        raw_llm_output=raw,
    )

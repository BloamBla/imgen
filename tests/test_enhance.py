"""Pure-function tests for src/imgen/enhance.py.

v0.5 Phase B: strict TDD on the LLM-free decision logic that wraps the
prompt enhancer. No mlx_lm import, no model load, no subprocess. The
impure ``run_with_mlx_lm`` wrapper that actually calls Qwen2.5-7B is
covered separately in v0.5 Phase C with subprocess mocks.

Surface under test:

* ``should_enhance(prompt, *, enabled, max_input_bytes)``
* ``build_messages(system_prompt, user_prompt)``
* ``extract_enhanced_text(llm_raw_output)``
* ``check_invariants(enhanced, original, invariants)``
* ``apply_length_cap(text, max_bytes)``
* ``decide_final_prompt(...)`` — top-level orchestrator
* ``EnhanceResult`` dataclass
"""
from __future__ import annotations

import pytest

from imgen.enhance import (
    EnhanceResult,
    apply_length_cap,
    build_messages,
    check_invariants,
    decide_final_prompt,
    extract_enhanced_text,
    should_enhance,
)


# ── should_enhance ──────────────────────────────────────────────────────


class TestShouldEnhance:
    def test_disabled_returns_false_even_for_valid_prompt(self):
        assert should_enhance("normal prompt", enabled=False) is False

    def test_empty_string_returns_false(self):
        assert should_enhance("", enabled=True) is False

    def test_whitespace_only_returns_false(self):
        assert should_enhance("   \n\t  ", enabled=True) is False

    def test_normal_prompt_returns_true(self):
        assert should_enhance("Restyle this person as anime", enabled=True) is True

    def test_at_or_below_max_input_passes(self):
        # 2048 byte default cap. Exactly 2048 passes.
        prompt = "a" * 2048
        assert should_enhance(prompt, enabled=True) is True

    def test_above_max_input_returns_false(self):
        prompt = "a" * 2049
        assert should_enhance(prompt, enabled=True) is False

    def test_custom_max_input_bytes(self):
        assert should_enhance("a" * 50, enabled=True, max_input_bytes=49) is False
        assert should_enhance("a" * 50, enabled=True, max_input_bytes=50) is True
        assert should_enhance("a" * 50, enabled=True, max_input_bytes=51) is True

    def test_max_input_is_BYTES_not_chars(self):
        # Cyrillic 'я' is 2 bytes in UTF-8. 10 'я' chars = 20 bytes.
        # The cap is on encoded length so Qwen tokeniser doesn't blow up.
        prompt = "я" * 10
        assert should_enhance(prompt, enabled=True, max_input_bytes=20) is True
        assert should_enhance(prompt, enabled=True, max_input_bytes=19) is False


# ── build_messages ──────────────────────────────────────────────────────


class TestBuildMessages:
    def test_standard_shape(self):
        msgs = build_messages("sys instruction", "user content")
        assert msgs == [
            {"role": "system", "content": "sys instruction"},
            {"role": "user", "content": "user content"},
        ]

    def test_returns_new_list_each_call(self):
        a = build_messages("s", "u")
        b = build_messages("s", "u")
        assert a == b
        assert a is not b

    def test_does_not_mutate_inputs(self):
        sys = "s"
        usr = "u"
        msgs = build_messages(sys, usr)
        msgs.append({"role": "assistant", "content": "x"})
        # Original strings can't be mutated, but the calling-site list
        # we returned was a fresh object — verifying append above didn't
        # somehow taint a hidden cache.
        assert build_messages(sys, usr) == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]


# ── extract_enhanced_text ───────────────────────────────────────────────


class TestExtractEnhancedText:
    def test_plain_text_passes_through(self):
        assert extract_enhanced_text("Restyle this person as anime") == \
            "Restyle this person as anime"

    def test_strips_leading_trailing_whitespace(self):
        assert extract_enhanced_text("  text  \n") == "text"

    def test_strips_outer_double_quotes(self):
        assert extract_enhanced_text('"hello world"') == "hello world"

    def test_strips_outer_single_quotes(self):
        assert extract_enhanced_text("'hello world'") == "hello world"

    def test_does_not_strip_unpaired_quotes(self):
        assert extract_enhanced_text('hello "world') == 'hello "world'

    def test_does_not_strip_inner_quotes(self):
        # The text contains a quoted phrase mid-sentence — leave alone.
        assert extract_enhanced_text('say "hi" to me') == 'say "hi" to me'

    def test_empty_input_returns_empty(self):
        assert extract_enhanced_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert extract_enhanced_text("   \n  ") == ""


# ── check_invariants ────────────────────────────────────────────────────


class TestCheckInvariants:
    def test_empty_invariants_always_valid(self):
        assert check_invariants("any text", "any other", ()) == (True, None)

    def test_invariant_present_in_both(self):
        ok, reason = check_invariants(
            "Restyle preserving identity",  # enhanced
            "preserving identity",          # original
            ("preserving",),
        )
        assert ok is True
        assert reason is None

    def test_invariant_in_original_but_not_enhanced_fails(self):
        ok, reason = check_invariants(
            "Make this anime style",        # enhanced (dropped "preserving")
            "anime while preserving face",  # original
            ("preserving",),
        )
        assert ok is False
        assert "preserving" in reason

    def test_invariant_not_in_original_is_skipped(self):
        # Original didn't have "preserving" → don't enforce on enhanced.
        ok, reason = check_invariants(
            "Make this anime",
            "make it anime",
            ("preserving",),
        )
        assert ok is True
        assert reason is None

    def test_multiple_invariants_all_must_pass(self):
        ok, reason = check_invariants(
            "anime style preserving identity",   # has both
            "anime while preserving face",       # has both
            ("preserving", "anime"),
        )
        assert ok is True
        ok, reason = check_invariants(
            "anime style identity",              # missing preserving
            "anime while preserving face",       # has both
            ("preserving", "anime"),
        )
        assert ok is False
        assert "preserving" in reason

    def test_invariant_match_is_case_insensitive(self):
        # LLM might capitalise differently — we don't want false alarms.
        ok, reason = check_invariants(
            "anime Preserving identity",
            "anime while preserving face",
            ("preserving",),
        )
        assert ok is True


# ── apply_length_cap ────────────────────────────────────────────────────


class TestApplyLengthCap:
    def test_shorter_than_cap_unchanged(self):
        text, truncated = apply_length_cap("abc", 10)
        assert text == "abc"
        assert truncated is False

    def test_exactly_cap_unchanged(self):
        text, truncated = apply_length_cap("a" * 10, 10)
        assert text == "a" * 10
        assert truncated is False

    def test_longer_than_cap_truncated(self):
        text, truncated = apply_length_cap("a" * 20, 10)
        assert text == "a" * 10
        assert truncated is True

    def test_zero_cap_yields_empty_and_truncated(self):
        text, truncated = apply_length_cap("anything", 0)
        assert text == ""
        assert truncated is True

    def test_cap_is_BYTES_not_chars(self):
        # 'я' = 2 bytes UTF-8. 10 chars = 20 bytes. Cap at 20 = full pass.
        # Cap at 19 = truncate to fit; we trim at a CHAR boundary to avoid
        # producing invalid UTF-8 mid-byte.
        text, truncated = apply_length_cap("я" * 10, 20)
        assert text == "я" * 10
        assert truncated is False
        text, truncated = apply_length_cap("я" * 10, 19)
        # 19 / 2 = 9 full chars; the truncation policy is "longest prefix
        # whose UTF-8 encoding fits in the cap". So 9 chars = 18 bytes.
        assert text == "я" * 9
        assert truncated is True


# ── decide_final_prompt ─────────────────────────────────────────────────


class TestDecideFinalPrompt:
    def test_disabled_returns_original_no_call(self):
        # enabled=False → should never have called llm, even if it would
        # have returned something. Caller is responsible for not invoking
        # the LLM in this case; decide_final_prompt is the one-stop
        # orchestrator that handles it cleanly.
        result = decide_final_prompt(
            original="Restyle this person preserving identity",
            enhanced_or_none=None,
            invariants=("preserving",),
            max_output_bytes=60_000,
            disabled_reason="user_opt_out",
        )
        assert isinstance(result, EnhanceResult)
        assert result.final_prompt == "Restyle this person preserving identity"
        assert result.was_enhanced is False
        assert result.fallback_reason == "user_opt_out"
        assert result.was_truncated is False

    def test_skip_too_long_input(self):
        result = decide_final_prompt(
            original="x" * 5000,
            enhanced_or_none=None,
            invariants=(),
            max_output_bytes=60_000,
            disabled_reason="input_too_long",
        )
        assert result.final_prompt == "x" * 5000
        assert result.was_enhanced is False
        assert result.fallback_reason == "input_too_long"

    def test_empty_llm_output_falls_back(self):
        result = decide_final_prompt(
            original="Restyle preserving identity",
            enhanced_or_none="",
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.final_prompt == "Restyle preserving identity"
        assert result.was_enhanced is False
        assert result.fallback_reason == "empty_llm_output"

    def test_whitespace_llm_output_falls_back(self):
        result = decide_final_prompt(
            original="Restyle preserving identity",
            enhanced_or_none="   \n  ",
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.final_prompt == "Restyle preserving identity"
        assert result.was_enhanced is False
        assert result.fallback_reason == "empty_llm_output"

    def test_invariant_violation_falls_back(self):
        # Original had "preserving", enhanced dropped it → fallback.
        result = decide_final_prompt(
            original="Restyle while preserving identity",
            enhanced_or_none="Make it anime with vibrant colors",
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.final_prompt == "Restyle while preserving identity"
        assert result.was_enhanced is False
        assert result.fallback_reason == "invariant_violated"

    def test_valid_enhancement_returned(self):
        result = decide_final_prompt(
            original="Restyle preserving identity, anime",
            enhanced_or_none=(
                "Restyle this person as cel-shaded anime while preserving "
                "facial identity, vibrant studio colors"
            ),
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.was_enhanced is True
        assert result.fallback_reason is None
        assert "cel-shaded anime" in result.final_prompt
        assert result.was_truncated is False

    def test_long_enhancement_gets_truncated_but_kept(self):
        # Enhancer accidentally returned 70KB. We truncate to 60KB and
        # keep — better partial enhancement than fallback to terse.
        result = decide_final_prompt(
            original="Restyle preserving identity",
            enhanced_or_none="preserving " + ("x" * 70_000),
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.was_enhanced is True
        assert result.was_truncated is True
        assert len(result.final_prompt.encode("utf-8")) <= 60_000
        # Invariant must still be in the truncated text — truncate from
        # END preserves the leading content where the invariant clause
        # typically sits.
        assert "preserving" in result.final_prompt.lower()

    def test_raw_llm_output_stored_in_result(self):
        # For history.jsonl / debugging — we keep the unedited LLM string
        # too, even when we used it.
        raw = "Restyle preserving identity, anime, vibrant"
        result = decide_final_prompt(
            original="Restyle preserving identity",
            enhanced_or_none=raw,
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.raw_llm_output == raw

    def test_raw_stored_even_when_fallback(self):
        raw = "Make it anime"  # invariant-violating
        result = decide_final_prompt(
            original="Restyle preserving identity",
            enhanced_or_none=raw,
            invariants=("preserving",),
            max_output_bytes=60_000,
        )
        assert result.was_enhanced is False
        assert result.raw_llm_output == raw  # preserved for debug

    def test_raw_none_when_disabled(self):
        result = decide_final_prompt(
            original="x",
            enhanced_or_none=None,
            invariants=(),
            max_output_bytes=60_000,
            disabled_reason="user_opt_out",
        )
        assert result.raw_llm_output is None


# ── EnhanceResult dataclass ─────────────────────────────────────────────


class TestEnhanceResultDataclass:
    def test_is_frozen(self):
        result = EnhanceResult(
            final_prompt="x", was_enhanced=False,
            fallback_reason="user_opt_out", was_truncated=False,
            raw_llm_output=None,
        )
        with pytest.raises((AttributeError, Exception)):
            result.final_prompt = "y"

    def test_slots(self):
        # frozen+slots = no per-instance __dict__. Project convention
        # (test_iteration_has_slots_no_dict) accepts either exception
        # type — CPython raises TypeError for frozen-dataclass setattr
        # of a slotted name in some builds, AttributeError in others.
        result = EnhanceResult(
            final_prompt="x", was_enhanced=False,
            fallback_reason=None, was_truncated=False,
            raw_llm_output=None,
        )
        assert not hasattr(result, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            result.unknown_attr = 1  # type: ignore[attr-defined]

    def test_hash_explicitly_none(self):
        # Per project convention (v0.2.5 review): explicit ``__hash__ = None``
        # on dataclasses that aren't meant to be set keys / dict keys.
        # Matches Iteration, BatchContext.
        assert EnhanceResult.__hash__ is None


# ── Backend.enhance_* fields lock-in ───────────────────────────────────


class TestBackendEnhanceFieldsLockIn:
    """Built-in FLUX + Qwen backends must carry the tuned system prompts
    and preserving-invariant. Lock-in tests so a refactor / typo silently
    drops the enhance plumbing for the default backend."""

    def test_flux_has_kontext_system_prompt(self):
        from imgen.backends import BUILTIN_BACKENDS
        sys_prompt = BUILTIN_BACKENDS["flux"].enhance_system_prompt
        assert sys_prompt is not None
        assert "Kontext" in sys_prompt
        assert "Restyle this person as X while preserving Y" in sys_prompt
        # Defense-in-depth against LLM "describing the photo".
        assert "Kontext sees it directly" in sys_prompt

    def test_qwen_has_imperative_system_prompt(self):
        from imgen.backends import BUILTIN_BACKENDS
        sys_prompt = BUILTIN_BACKENDS["qwen"].enhance_system_prompt
        assert sys_prompt is not None
        assert "Qwen-Image-Edit" in sys_prompt
        # Qwen prefers shorter directives.
        assert "shorter" in sys_prompt or "40 tokens" in sys_prompt

    def test_both_carry_preserving_invariant(self):
        from imgen.backends import BUILTIN_BACKENDS
        for name in ("flux", "qwen"):
            assert "preserving" in BUILTIN_BACKENDS[name].enhance_invariants, name

    def test_user_backends_have_no_enhance_by_default(self):
        # A bare-minimum custom backend declared via backends.d/*.toml
        # shouldn't accidentally inherit FLUX's or Qwen's system prompt
        # — that would produce wrong-shape enhancements for whatever
        # the user is wiring up. Default is None = enhancer skipped.
        from imgen.backends import Backend
        b = Backend(
            binary="custom-binary",
            needs_token=False,
            image_flag="--image-path",
            supports_strength=False,
            supports_negative=False,
            extra_args=(),
        )
        assert b.enhance_system_prompt is None
        assert b.enhance_invariants == ()

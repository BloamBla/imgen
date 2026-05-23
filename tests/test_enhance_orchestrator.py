"""Tests for the iteration-level orchestrator and argv helper that
bridge :mod:`imgen.enhance` to ``cmd_generate`` / ``cmd_batch``.

Both surfaces are pure-functional with an injected LLM callable —
no real mlx_lm load, no subprocess. Real-mlx_lm smoke testing is
manual (see CLAUDE.md release checklist).

Exercised:

* :func:`enhance_iteration_prompts` — N-prompts-in, N-results-out
  orchestrator that handles per-prompt skips + all-or-nothing
  runner-failure fallback.
* :func:`replace_prompt_in_cmd` — argv patching when an enhancement
  modifies the prompt post-build.
"""
from __future__ import annotations

import pytest

from imgen.enhance import (
    EnhanceResult,
    RunnerError,
    enhance_iteration_prompts,
    replace_prompt_in_cmd,
)


# ── replace_prompt_in_cmd ───────────────────────────────────────────────


class TestReplacePromptInCmd:
    def test_basic_replacement(self):
        cmd = [
            "/bin/mflux-generate-kontext", "--quantize", "8",
            "--image-path", "/p.jpg",
            "--prompt", "OLD PROMPT",
            "--steps", "20",
        ]
        out = replace_prompt_in_cmd(cmd, "NEW PROMPT")
        assert out[5] == "--prompt"      # flag still where it was
        assert out[6] == "NEW PROMPT"    # value replaced
        assert out[7] == "--steps"       # next flag intact
        # And original cmd is untouched.
        assert cmd[6] == "OLD PROMPT"

    def test_does_not_mutate_input(self):
        cmd = ["--prompt", "old"]
        original = list(cmd)
        replace_prompt_in_cmd(cmd, "new")
        assert cmd == original

    def test_returns_new_list_object(self):
        cmd = ["--prompt", "old"]
        out = replace_prompt_in_cmd(cmd, "new")
        assert out is not cmd

    def test_missing_prompt_flag_is_no_op(self):
        cmd = ["/bin/mflux", "--quantize", "8"]
        out = replace_prompt_in_cmd(cmd, "anything")
        assert out == cmd

    def test_malformed_argv_dangling_prompt_is_no_op(self):
        # ``--prompt`` at end with no value — shouldn't happen via our
        # build_mflux_cmd but defensive: don't IndexError.
        cmd = ["/bin/mflux", "--prompt"]
        out = replace_prompt_in_cmd(cmd, "x")
        assert out == cmd


# ── enhance_iteration_prompts: fake LLM callable ──────────────────────


def _fake_run_llm_ok(*, items, **kwargs):
    """Stand-in LLM that echoes each user prompt prefixed with "ENH:". The
    "preserving" substring is kept so the FLUX/Qwen invariant passes."""
    out = []
    for it in items:
        # If the prompt already has "preserving", keep it in output.
        suffix = " (preserving preserved)" if "preserving" in it["user"].lower() else ""
        out.append(f"ENH: {it['user']}{suffix}")
    return out


def _fake_run_llm_drops_invariant(*, items, **kwargs):
    """LLM that drops the 'preserving' anchor clause — triggers the
    invariant_violated fallback."""
    return [f"ENH but no anchor: {it['user'].replace('preserving', '')}"
            for it in items]


def _fake_run_llm_empty(*, items, **kwargs):
    """LLM returns empty strings — triggers empty_llm_output fallback."""
    return ["" for _ in items]


def _fake_run_llm_raises(*, items, **kwargs):
    """LLM runner crashed (timeout / OOM / model load failed)."""
    raise RunnerError("simulated runner failure")


def _fake_run_llm_assert_uncalled(*, items, **kwargs):
    """Used in tests that expect the LLM to NOT be invoked."""
    raise AssertionError(
        "LLM should not have been called for this scenario"
    )


# ── enhance_iteration_prompts ──────────────────────────────────────────


class TestEnhanceIterationPrompts:
    def test_all_prompts_enhanced_when_runner_ok(self):
        prompts = [
            "Restyle preserving identity, anime",
            "Restyle preserving identity, pixar",
        ]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="FLUX-Kontext system",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_ok,
        )
        assert len(results) == 2
        for r in results:
            assert r.was_enhanced is True
            assert r.final_prompt.startswith("ENH: ")
            assert r.fallback_reason is None

    def test_returns_aligned_results_for_mixed_skip_pass(self):
        # First prompt is too long (>2 KB UTF-8) → skipped at gate;
        # second is normal → goes through LLM.
        prompts = [
            "x" * 3000,  # too long → input_too_long
            "Restyle preserving identity, anime",
        ]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_ok,
        )
        assert len(results) == 2
        assert results[0].was_enhanced is False
        assert results[0].fallback_reason == "input_too_long"
        # Skipped prompts get their original back.
        assert results[0].final_prompt == prompts[0]
        assert results[1].was_enhanced is True

    def test_empty_prompt_skipped_with_correct_reason(self):
        prompts = ["", "Restyle preserving identity"]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_ok,
        )
        assert results[0].was_enhanced is False
        assert results[0].fallback_reason == "empty_input"
        assert results[0].final_prompt == ""

    def test_none_system_prompt_disables_for_all(self):
        # User-supplied backend without enhance_system_prompt → entire
        # batch skipped with not_supported_by_backend.
        prompts = ["a", "b", "c"]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt=None,                    # ← key bit
            invariants=(),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_assert_uncalled,  # never called
        )
        assert len(results) == 3
        for r, p in zip(results, prompts):
            assert r.was_enhanced is False
            assert r.fallback_reason == "not_supported_by_backend"
            assert r.final_prompt == p

    def test_all_prompts_skipped_no_subprocess(self):
        # Every prompt fails the gate (all empty) → LLM never called.
        prompts = ["", "   ", "\n"]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=(),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_assert_uncalled,  # never called
        )
        assert len(results) == 3
        for r in results:
            assert r.was_enhanced is False
            assert r.fallback_reason == "empty_input"

    def test_runner_failure_falls_back_all_results(self):
        prompts = [
            "Restyle preserving identity, anime",
            "Restyle preserving identity, ghibli",
        ]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_raises,
        )
        assert len(results) == 2
        for r, p in zip(results, prompts):
            assert r.was_enhanced is False
            assert r.fallback_reason == "runner_error"
            assert r.final_prompt == p  # original preserved
            # v0.6.5: error message lives in fallback_detail (symmetric
            # with invariant_violated). raw_llm_output is None because
            # the runner crashed before producing LLM output.
            assert r.raw_llm_output is None
            assert "simulated runner failure" in (r.fallback_detail or "")

    def test_invariant_violation_falls_back_per_prompt(self):
        prompts = ["Restyle preserving identity, anime"]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_drops_invariant,
        )
        assert results[0].was_enhanced is False
        assert results[0].fallback_reason == "invariant_violated"
        assert results[0].final_prompt == prompts[0]

    def test_empty_llm_output_falls_back_per_prompt(self):
        prompts = ["Restyle preserving identity"]
        results = enhance_iteration_prompts(
            iteration_prompts=prompts,
            system_prompt="sys",
            invariants=("preserving",),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_empty,
        )
        assert results[0].was_enhanced is False
        assert results[0].fallback_reason == "empty_llm_output"
        assert results[0].final_prompt == prompts[0]

    def test_empty_iteration_list_returns_empty(self):
        results = enhance_iteration_prompts(
            iteration_prompts=[],
            system_prompt="sys",
            invariants=(),
            model="m", temperature=0.0, max_tokens=200,
            run_llm=_fake_run_llm_assert_uncalled,
        )
        assert results == []

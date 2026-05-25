"""v0.8.2 M-1A — Iteration carries Model + GenParams for Engine.run wire-up.

Per [[project-v080-design]] §A-§E + the v0.8.2 architect pre-vet
CRITICAL-3 fix. Sub-commit M-1A is the data-shape change: each
Iteration carries its resolved Model + GenParams alongside the legacy
pre-built ``cmd`` argv. The dispatch flip (Engine.run actually called)
lands in sub-commit M-1C. These tests lock in the data invariants the
flip depends on.

Three lock-ins:

* Every Iteration produced by the 4 production build_* helpers
  (build_iterations, build_draw_iterations, build_refine_iteration,
  build_bare_i2i_iteration) has non-None ``model`` + ``params``.
  Architect MEDIUM-2 — catches construction-site drift before
  sub-commit M-1C's dispatch fence ships.
* ``apply_enhance_results_to_iterations`` dual-updates ``cmd`` AND
  ``params.prompt`` when an enhance succeeds. Architect CRITICAL-3 —
  pre-fix would silently drop enhanced prompts on Engine-dispatched
  iterations because MfluxEngine.run reads ``params.prompt``, not
  the spliced ``cmd``.
* Legacy callers constructing Iteration without ``params`` (e.g.
  test fixtures) still work — the dual-update path preserves None
  rather than crashing.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.cmd_helpers import apply_enhance_results_to_iterations
from imgen.enhance import EnhanceResult
from imgen.runs import Iteration


# ── apply_enhance dual-update (CRITICAL-3) ───────────────────────────


def _make_iter_with_params(prompt: str = "samurai") -> Iteration:
    """Construct an Iteration with both ``cmd`` AND ``params``
    populated — matches the post-M-1A production shape from
    build_iterations / build_draw_iterations / etc."""
    from imgen.engines.base import GenParams
    return Iteration(
        style_name="draw",
        prompt=prompt,
        negative="",
        final_steps=20,
        final_quantize=4,
        final_guidance=3.5,
        final_strength=0.0,
        output_path=Path("/tmp/out.png"),
        cmd=["fake-mflux", "--prompt", prompt, "--steps", "20"],
        params=GenParams(
            prompt=prompt,
            negative="",
            width=1024, height=1024,
            steps=20, guidance=3.5, seed=42, quantize=4, strength=0.0,
            input_path=None,
            output_path=Path("/tmp/out.png"),
            loras=(),
        ),
    )


def _make_enhance_result(original: str, enhanced: str) -> EnhanceResult:
    return EnhanceResult(
        final_prompt=enhanced,
        original_prompt=original,
        was_enhanced=True,
        fallback_reason=None,
        was_truncated=False,
        raw_llm_output=enhanced,
    )


def test_apply_enhance_dual_updates_cmd_and_params():
    """v0.8.2 architect CRITICAL-3 closure: when the enhancer succeeds,
    BOTH ``it.cmd`` (legacy argv) and ``it.params.prompt`` (Engine.run
    payload) must be updated. Pre-fix only ``cmd`` was patched; after
    the sub-commit M-1C dispatch flip, MfluxEngine.run would read the
    un-patched ``params.prompt`` and the enhanced prompt would be
    silently lost.
    """
    it = _make_iter_with_params(prompt="samurai")
    enhanced = "a fierce samurai standing on a misty mountain at dawn"
    result = _make_enhance_result("samurai", enhanced)

    out = apply_enhance_results_to_iterations([it], [result])
    assert len(out) == 1
    new_it = out[0]

    # Iteration.prompt updated (existing v0.5 contract)
    assert new_it.prompt == enhanced
    # cmd argv has the enhanced prompt spliced
    assert enhanced in new_it.cmd
    # NEW v0.8.2: params.prompt also carries the enhanced text
    assert new_it.params is not None
    assert new_it.params.prompt == enhanced


def test_apply_enhance_preserves_other_params_fields_on_dual_update():
    """The dual-update mutates ONLY ``params.prompt``. Every other
    GenParams field must round-trip unchanged — defence against a
    future careless rebuild that clobbers seed / steps / etc."""
    it = _make_iter_with_params(prompt="samurai")
    enhanced = "fierce samurai etc"
    result = _make_enhance_result("samurai", enhanced)

    out = apply_enhance_results_to_iterations([it], [result])
    new_params = out[0].params
    assert new_params is not None
    # Mutated:
    assert new_params.prompt == enhanced
    # All other GenParams fields unchanged:
    assert new_params.negative == ""
    assert new_params.width == 1024
    assert new_params.height == 1024
    assert new_params.steps == 20
    assert new_params.guidance == 3.5
    assert new_params.seed == 42
    assert new_params.quantize == 4
    assert new_params.strength == 0.0
    assert new_params.input_path is None
    assert new_params.output_path == Path("/tmp/out.png")


def test_apply_enhance_legacy_iteration_without_params_still_works():
    """Iterations constructed without ``params`` (legacy callers; some
    test fixtures) keep working — the dual-update preserves
    ``params=None`` rather than crashing. ``Iteration.params`` has a
    None default for exactly this transition window per architect
    MEDIUM-2."""
    legacy_it = Iteration(
        style_name="legacy",
        prompt="original",
        negative="",
        final_steps=20,
        final_quantize=4,
        final_guidance=3.5,
        final_strength=0.0,
        output_path=Path("/tmp/out.png"),
        cmd=["fake-mflux", "--prompt", "original"],
        # NO model, NO params — legacy shape
    )
    result = _make_enhance_result("original", "enhanced version")

    out = apply_enhance_results_to_iterations([legacy_it], [result])
    new_it = out[0]
    assert new_it.prompt == "enhanced version"
    assert "enhanced version" in new_it.cmd
    # No params to update — stays None, no crash
    assert new_it.params is None


def test_apply_enhance_skipped_path_doesnt_touch_params():
    """When an enhance result is a fallback (was_enhanced=False),
    Iteration is returned unchanged — including ``params``. Locks in
    that the no-op branch doesn't accidentally reach into params."""
    it = _make_iter_with_params(prompt="samurai")
    original_params = it.params  # capture for identity check
    fallback = EnhanceResult(
        final_prompt="samurai",
        original_prompt="samurai",
        was_enhanced=False,
        fallback_reason="invariant_violated",
        was_truncated=False,
        raw_llm_output=None,
    )

    out = apply_enhance_results_to_iterations([it], [fallback])
    # Unchanged iteration — same Iteration object (no replace fired)
    assert out[0] is it
    assert out[0].params is original_params


# ── build_* helpers populate Model + GenParams (MEDIUM-2) ────────────


def test_build_iterations_populates_model_and_params(monkeypatch):
    """v0.8.2 architect MEDIUM-2 lock-in: every Iteration produced by
    ``build_iterations`` (the i2i preset path) carries non-None
    ``model`` + ``params``. After sub-commit M-1C's dispatch flip,
    ``run_one_iteration`` will dispatch via Engine.run for any
    iteration where ``it.model`` is non-None — if a build_* helper
    silently leaves them None, that iteration falls into the legacy
    fallback branch instead of the new Engine path. This test
    catches construction-site drift before production hits the
    fallback.
    """
    from pathlib import Path

    import imgen.cmd_helpers as ch
    from imgen.backends import BACKENDS

    # _build_args mirrors the test_generate_helpers.py shape; minimal
    # Namespace satisfying build_iterations's getattr chain.
    args = SimpleNamespace(
        style=None,
        custom_prompt=None,
        prompt_file=None,
        scope=None,
        preview=False,
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        seed=42,
        model="flux",  # v0.7 alias; resolver maps to flux-kontext
        lora=None,
        no_lora=False,
        negative_prompt=None,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    merged_defaults = dict(
        style="pixar", model="flux-kontext", backend_draw="flux-dev",
        quantize=4, steps=20, guidance=3.5, strength=0.55,
        mlx_cache_gb=12, battery_stop=20,
    )
    iters = ch.build_iterations(
        styles_list=["anime"],
        args=args,
        effective_custom_prompt=None,
        merged_defaults=merged_defaults,
        be=BACKENDS["flux"],
        binary=Path("/fake/bin/mflux-generate-kontext"),
        input_path=Path("/tmp/photo.jpg"),
        width=1024, height=1024,
        explicit_output=None,
        run_dir=Path("/tmp/run"),
        seed=42,
    )
    assert len(iters) == 1
    it = iters[0]
    # M-1A guarantee:
    assert it.model is not None, (
        "build_iterations must populate Iteration.model "
        "for Engine.run dispatch (architect MEDIUM-2)"
    )
    assert it.params is not None, (
        "build_iterations must populate Iteration.params "
        "for Engine.run dispatch"
    )
    # Sanity: the model is the resolved v0.8 canonical Model
    assert it.model.engine == "mflux"
    # GenParams carries the same numeric quartet as Iteration
    assert it.params.steps == it.final_steps
    assert it.params.quantize == it.final_quantize
    assert it.params.guidance == it.final_guidance
    assert it.params.strength == it.final_strength
    assert it.params.seed == it.seed


def test_build_draw_iterations_populates_model_and_params():
    """Same MEDIUM-2 lock-in for the draw (t2i) path. cmd_draw's
    iterations must dispatch through Engine.run too — and
    build_draw_iterations is the sole construction site for that
    path. Architect M-1A scope: all 4 build_* helpers carry the
    payload."""
    from pathlib import Path

    import imgen.cmd_helpers as ch
    from imgen.backends import BACKENDS

    args = SimpleNamespace(
        prompt="a samurai",
        steps=None, quantize=None, guidance=None,
        seed=42,
        model="flux-dev",
        width=1024, height=1024,
        lora=None, no_lora=False,
        negative_prompt=None,
        preview=False,
        num_iterations=1,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    merged_defaults = dict(
        style="pixar", model="flux-kontext", backend_draw="flux-dev",
        quantize=4, steps=20, guidance=3.5, strength=0.55,
        mlx_cache_gb=12, battery_stop=20,
    )
    iters = ch.build_draw_iterations(
        args=args,
        prompt="a samurai",
        merged_defaults=merged_defaults,
        be=BACKENDS["flux-dev"],
        binary=Path("/fake/bin/mflux-generate"),
        width=1024, height=1024,
        explicit_output=None,
        run_dir=Path("/tmp/run"),
        base_seed=42,
        num_iterations=1,
    )
    assert len(iters) == 1
    it = iters[0]
    assert it.model is not None
    assert it.params is not None
    assert it.model.engine == "mflux"
    # Draw is t2i → input_path is None
    assert it.params.input_path is None

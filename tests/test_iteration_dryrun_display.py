"""Lock-ins for ``engine_dispatch.iteration_dryrun_display`` (v0.8.4 M-NEW-D).

Pre-v0.8.4 ``--dry-run`` printed ``format_cmd(it.cmd)``, reading the
build-time argv snapshot stored on each Iteration. v0.8.4 dropped the
``cmd`` field and replaced the print site with
``iteration_dryrun_display(it)`` which derives the dispatch shape from
``(it.model, it.params)`` at print time.

This module locks the three branches:

* mflux engine → byte-identical to ``format_cmd(MfluxEngine.build_cmd
  (model, params))``. The argv-shape itself is locked separately by
  ``test_mflux_engine_build_cmd_matches_legacy_build_mflux_cmd`` so
  this test only checks that ``iteration_dryrun_display`` routes
  through the same path.
* diffusers_mps engine → multi-line structured display of the
  stdin-JSON payload that ``DiffusersMpsEngine.run`` actually sends.
  The display MUST include every field the runner consumes — pre-tag
  review v0.8.4 caught the original draft omitting
  ``cpu_offload_threshold_mp`` (MEDIUM-1 closure).
* None model / params → static legacy-fallback sentinel string.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.engine_dispatch import iteration_dryrun_display
from imgen.engines.base import GenParams
from imgen.engines.mflux_engine import MfluxEngine
from imgen.models import BUILTIN_MODELS, Model
from imgen.runs import Iteration
from imgen.subprocess_helpers import format_cmd


def _mflux_iter(tmp_path) -> Iteration:
    model = BUILTIN_MODELS["flux-kontext"]
    params = GenParams(
        prompt="a samurai on a misty mountain",
        negative="low quality",
        width=1024, height=1024,
        steps=20, guidance=3.5, seed=42, quantize=8, strength=0.6,
        input_path=tmp_path / "in.jpg",
        output_path=tmp_path / "out.png",
        loras=(),
    )
    return Iteration(
        style_name="anime", prompt=params.prompt, negative=params.negative,
        final_steps=20, final_quantize=8, final_guidance=3.5, final_strength=0.6,
        output_path=params.output_path,
        seed=42, model=model, params=params,
    )


def test_mflux_branch_matches_engine_build_cmd(tmp_path):
    """``iteration_dryrun_display`` on an mflux Iteration must equal
    ``format_cmd(MfluxEngine.build_cmd(model, params))`` — same path
    Engine.run uses internally. Drift here would mean dry-run shows
    one argv shape while production runs a different one."""
    it = _mflux_iter(tmp_path)
    expected = format_cmd(MfluxEngine().build_cmd(it.model, it.params))
    assert iteration_dryrun_display(it) == expected


def test_mflux_branch_reflects_enhanced_params_prompt(tmp_path):
    """The point of the v0.8.4 M-NEW-D refactor: dry-run-with-enhance
    must surface the enhanced text. Pre-v0.8.4 the dual-update in
    apply_enhance kept ``it.cmd`` synced; v0.8.4 derives argv from
    ``params.prompt`` which apply_enhance updates."""
    from dataclasses import replace as _replace
    it = _mflux_iter(tmp_path)
    enhanced_params = _replace(it.params, prompt="ENHANCED — fierce samurai etc")
    enhanced_it = _replace(it, params=enhanced_params)
    display = iteration_dryrun_display(enhanced_it)
    assert "ENHANCED" in display


def _diffusers_model() -> Model:
    return Model(
        engine="diffusers_mps",
        repo="mlx-community/Qwen-Image-2512-4bit",
        cpu_offload_threshold_mp=2.0,
        ram_baseline_gb=10.0, ram_slope_gb_per_mp=5.0, encoder_ram_gb=7.0,
        param_overrides=(("true_cfg_scale", 4.0),),
    )


def _diffusers_iter(tmp_path) -> Iteration:
    model = _diffusers_model()
    params = GenParams(
        prompt="a samurai on a misty mountain",
        negative="",
        width=1024, height=1024,
        steps=50, guidance=4.0, seed=42, quantize=4, strength=0.0,
        input_path=None,
        output_path=tmp_path / "out.png",
        loras=(),
    )
    return Iteration(
        style_name="draw", prompt=params.prompt, negative="",
        final_steps=50, final_quantize=4, final_guidance=4.0, final_strength=0.0,
        output_path=params.output_path,
        seed=42, model=model, params=params,
    )


class TestDiffusersBranchShowsAllPayloadFields:
    """The diffusers_mps display must include every field the runner
    consumes (``DiffusersMpsEngine.run`` JSON payload shape). v0.8.4
    pre-tag review caught an early draft omitting
    ``cpu_offload_threshold_mp`` — lock the full set."""

    def test_shows_runner_invocation_line(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "_diffusers_runner" in display
        assert ".venv-diffusers" in display

    def test_shows_repo(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "mlx-community/Qwen-Image-2512-4bit" in display

    def test_shows_prompt(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "a samurai on a misty mountain" in display

    def test_shows_steps_guidance_seed_width_height(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        for needle in ["steps: 50", "guidance: 4.0", "seed: 42",
                       "width: 1024", "height: 1024"]:
            assert needle in display, f"missing {needle!r} in display"

    def test_shows_cpu_offload_threshold_mp(self, tmp_path):
        """v0.8.4 pre-tag review MEDIUM-1 closure — without this the
        displayed dispatch shape silently differs from what
        DiffusersMpsEngine.run actually serializes."""
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "cpu_offload_threshold_mp: 2.0" in display

    def test_shows_output_path(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "out.png" in display

    def test_shows_param_overrides_when_set(self, tmp_path):
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "param_overrides" in display
        assert "true_cfg_scale" in display

    def test_omits_input_path_when_none(self, tmp_path):
        """t2i path: ``params.input_path is None`` — the display
        shouldn't fabricate an input_path line."""
        display = iteration_dryrun_display(_diffusers_iter(tmp_path))
        assert "input_path:" not in display

    def test_shows_input_path_when_set(self, tmp_path):
        """i2i path: ``params.input_path`` carries the source photo
        — the display must surface it."""
        from dataclasses import replace as _replace
        it = _diffusers_iter(tmp_path)
        i2i_params = _replace(it.params, input_path=tmp_path / "source.png")
        i2i_it = _replace(it, params=i2i_params)
        display = iteration_dryrun_display(i2i_it)
        assert "input_path:" in display
        assert "source.png" in display


def test_legacy_iteration_without_model_returns_fallback_sentinel(tmp_path):
    """Defensive: an Iteration with no model/params (legacy test fixture
    shape; production hard-asserts non-None at dispatch) must produce
    a static sentinel string rather than crashing."""
    legacy_it = Iteration(
        style_name="legacy", prompt="x", negative="",
        final_steps=20, final_quantize=4, final_guidance=3.5, final_strength=0.0,
        output_path=tmp_path / "out.png",
        # NO model, NO params — legacy shape
    )
    assert iteration_dryrun_display(legacy_it) == \
        "(legacy Iteration — no model/params)"


def test_unknown_engine_raises_value_error(tmp_path):
    """``Model.__post_init__`` enforces engine ∈ {mflux, diffusers_mps}
    so this branch is structurally unreachable from any normal
    constructor. But the display function still has a defensive
    ``raise ValueError`` for forward-compat shapes that bypass the
    post-init guard (e.g. ``object.__setattr__`` on a frozen
    instance). Lock the error message contract so future engines
    failing to register a display branch produce a clear error."""
    it = _mflux_iter(tmp_path)
    # Bypass the frozen-dataclass guard via object.__setattr__ — this
    # is exactly the forward-compat-shape vector the defensive
    # ValueError catches. dataclasses.replace re-runs __post_init__
    # which would block the engine= override.
    object.__setattr__(it.model, "engine", "future_engine")
    try:
        with pytest.raises(ValueError, match="future_engine"):
            iteration_dryrun_display(it)
    finally:
        # Restore for any downstream side effects (frozen, but the
        # Model instance is module-cached in BUILTIN_MODELS — leaking
        # state would poison later tests).
        object.__setattr__(it.model, "engine", "mflux")

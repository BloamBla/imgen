"""v0.8.0 commit 7 — per-Model param defaults + Engine.validate lock-ins.

Per [[project-v080-design]] §M + §Q commit 7:

* ``cmd_generate / cmd_draw / cmd_refine / cmd_batch`` read
  ``model.default_steps`` / ``model.default_guidance`` when the user
  didn't pass ``--steps`` / ``--guidance`` AND ``--preview`` isn't
  active. Precedence: CLI > preview > preset > model > merged_defaults.
* ``MfluxEngine.validate`` rejects (quantize, guidance) combinations
  the underlying mflux binary would reject at argv-parse time.
* ``build_mflux_cmd`` skips ``--quantize`` emission when
  ``model.omit_quantize=True`` (prequantized model repos).
* Hardcoded special-cases (``refine.py:238`` FLUX.2 guidance pin) are
  REMOVED — replaced by per-Model min_guidance/max_guidance pins +
  the centralised ``validate_engine_params_or_die`` helper.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.models import Model


def _make_genparams(**overrides):
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="x", negative="",
        width=1024, height=1024, steps=20, guidance=3.5,
        seed=42, quantize=4, strength=0.0,
        input_path=None, output_path=Path("/tmp/out.png"),
        loras=(),
    )
    defaults.update(overrides)
    return GenParams(**defaults)


# ── §Q lock-in 1: default_steps applied when user omits ───────────────


def test_model_default_steps_applied_when_user_omits():
    """Sentinel-Model lock-in: when ``args.steps is None`` AND
    ``args.preview is False`` AND no preset override, the resolver
    picks ``model.default_steps`` over ``merged_defaults["steps"]``.

    Pre-commit-7 the chain was CLI > preview > merged_defaults;
    commit 7 inserts model.default_steps between preview/preset and
    merged_defaults so e.g. Qwen-Image-Edit's recommended 30 steps
    win for the qwen-image-edit-v1 model without the user having to
    type ``--steps 30`` every invocation.
    """
    import argparse
    from imgen.cmd_helpers import _resolve_iteration_params
    from imgen.styles import Style

    sentinel_model = Model(
        engine="mflux", binary="mflux-generate-fake",
        default_steps=42,
        default_guidance=3.5,
        min_guidance=0.0,
        max_guidance=10.0,
        ram_baseline_gb=9.0,
        ram_slope_gb_per_mp=5.0,
    )
    args = argparse.Namespace(
        steps=None, quantize=None, guidance=None, strength=None,
        preview=False,
    )
    preset = Style()  # empty: no steps/guidance/strength overrides
    merged = {"steps": 20, "quantize": 8, "guidance": 3.5, "strength": 0.55}

    params = _resolve_iteration_params(
        args=args, preset=preset, merged_defaults=merged,
        model=sentinel_model,
    )
    assert params.final_steps == 42, (
        "model.default_steps must win over merged_defaults['steps'] "
        "when user didn't pass --steps and preview is off"
    )


def test_model_default_steps_overridden_by_preview():
    """Preview mode still wins over model.default_steps — preview is
    the user's explicit speed-vs-quality choice, model defaults are
    quality-tuning."""
    import argparse
    from imgen.cmd_helpers import _resolve_iteration_params
    from imgen.defaults import PREVIEW_OVERRIDES
    from imgen.styles import Style

    sentinel_model = Model(
        engine="mflux", binary="mflux-generate-fake",
        default_steps=50,
        default_guidance=3.5,
        min_guidance=0.0,
        max_guidance=10.0,
        ram_baseline_gb=9.0,
        ram_slope_gb_per_mp=5.0,
    )
    args = argparse.Namespace(
        steps=None, quantize=None, guidance=None, strength=None,
        preview=True,
    )
    params = _resolve_iteration_params(
        args=args, preset=Style(),
        merged_defaults={
            "steps": 20, "quantize": 8, "guidance": 3.5, "strength": 0.55,
        },
        model=sentinel_model,
    )
    assert params.final_steps == PREVIEW_OVERRIDES["steps"]


# ── §Q lock-in 2: turbo + guidance error ──────────────────────────────


def test_engine_validate_rejects_turbo_with_guidance_3_5():
    """A distilled model with ``min_guidance=max_guidance=0.0`` (the
    Z-Image-Turbo / FLUX-schnell shape) MUST reject any non-zero
    guidance. mflux's distilled-binary argparse already does this;
    Engine.validate surfaces the rejection at the parent CLI layer
    so the user gets a clean error before subprocess launch.
    """
    from imgen.engines.mflux_engine import MfluxEngine
    turbo = Model(
        engine="mflux", binary="mflux-generate-z-image-turbo",
        default_steps=9,
        default_guidance=0.0,
        min_guidance=0.0,
        max_guidance=0.0,  # distilled — CFG MUST be off
        ram_baseline_gb=8.0,
        ram_slope_gb_per_mp=4.5,
    )
    params = _make_genparams(guidance=3.5, quantize=4)
    errors = MfluxEngine().validate(turbo, params)
    assert errors, "guidance=3.5 must reject for distilled turbo model"
    assert any("guidance" in e for e in errors)
    assert any("3.5" in e for e in errors)


# ── §Q lock-in 3: FLUX.2 max_guidance enforcement ─────────────────────


def test_flux2_klein_max_guidance_enforced_via_engine_validate():
    """v0.8.0 commit 7 (§M): FLUX.2-klein-edit-9b ships
    min_guidance=max_guidance=1.0 — mflux 0.17.5's
    `mflux-generate-flux2-edit` rejects anything else at argv. The
    per-Model pin replaces the pre-commit-7 ``refine.py:238`` silent
    override.
    """
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux2-klein-edit-9b"]
    # Out-of-range high
    errors_high = MfluxEngine().validate(
        model, _make_genparams(guidance=3.5, quantize=4),
    )
    assert errors_high
    assert any("3.5" in e for e in errors_high)
    # Out-of-range low (FLUX.2 also rejects guidance < 1.0)
    errors_low = MfluxEngine().validate(
        model, _make_genparams(guidance=0.5, quantize=4),
    )
    assert errors_low
    # In-range (the only accepted value): no errors
    errors_ok = MfluxEngine().validate(
        model, _make_genparams(guidance=1.0, quantize=4),
    )
    assert errors_ok == []


def test_flux_kontext_min_guidance_enforced():
    """FLUX.1-Kontext-dev's ``min_guidance=1.0`` rejects ``guidance=0.0``
    — non-distilled FLUX needs real CFG (0.0 produces blurry,
    uninstructable output)."""
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux-kontext"]
    errors = MfluxEngine().validate(
        model, _make_genparams(guidance=0.0, quantize=8),
    )
    assert errors
    assert any("0.0" in e for e in errors)


# ── §Q lock-in 4: omit_quantize skips argv ────────────────────────────


def test_omit_quantize_skips_quantize_argv_for_prequantized_models():
    """v0.8.0 commit 7 (§M + §F): ``model.omit_quantize=True`` ships
    on Model rows pointing at prequantized repos (e.g.
    ``mlx-community/Qwen-Image-2512-4bit`` — weights are already
    int4-packed; mflux's ``--quantize 4`` against them no-ops, but
    the contract is undocumented). Skipping the flag emission makes
    the contract explicit at the Model level rather than at the
    per-binary cmd_* level.
    """
    from imgen.engines.base import GenParams
    from imgen.engines.mflux_engine import MfluxEngine

    prequant_model = Model(
        engine="mflux", binary="mflux-generate-fake",
        omit_quantize=True,
        default_steps=20,
        default_guidance=4.0,
        min_guidance=0.0,
        max_guidance=10.0,
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=5.0,
    )
    params = GenParams(
        prompt="x", negative="", width=1024, height=1024,
        steps=20, guidance=4.0, seed=42, quantize=4, strength=0.0,
        input_path=None, output_path=Path("/tmp/out.png"), loras=(),
    )
    argv = MfluxEngine().build_cmd(
        prequant_model, params, binary=Path("/fake/mflux-bin"),
    )
    # NO --quantize flag in argv when omit_quantize=True
    assert "--quantize" not in argv
    # All other args still emit
    assert "--prompt" in argv
    assert "--steps" in argv
    assert "--guidance" in argv


def test_omit_quantize_default_false_keeps_quantize_argv():
    """Symmetric sanity: a Model with the default
    ``omit_quantize=False`` continues to emit ``--quantize N``. This
    locks against a regression where the default flips to True and
    every built-in suddenly drops the flag.
    """
    from imgen.engines.base import GenParams
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux-kontext"]
    assert model.omit_quantize is False  # locked default
    params = GenParams(
        prompt="x", negative="", width=1024, height=1024,
        steps=20, guidance=3.5, seed=42, quantize=8, strength=0.55,
        input_path=Path("/fake/in.png"),
        output_path=Path("/tmp/out.png"), loras=(),
    )
    argv = MfluxEngine().build_cmd(
        model, params, binary=Path("/fake/mflux-bin"),
    )
    assert "--quantize" in argv
    # The value immediately follows
    idx = argv.index("--quantize")
    assert argv[idx + 1] == "8"


# ── Wire-up lock-in: validate_engine_params_or_die centralises the gate ─


def _validate_params(model, **gen_overrides):
    """Build a GenParams for the validate gate from a Model row.
    Defaults pull from the Model so validate passes out-of-the-box;
    tests override only the axis they want to exercise.

    v0.9.3 C3: validate_engine_params_or_die signature changed to
    ``(model, *, params: GenParams)``. This helper centralises the
    GenParams construction the tests below need.
    """
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="x", negative="", width=64, height=64,
        steps=model.default_steps,
        guidance=model.default_guidance,
        seed=0,
        quantize=model.supported_quants[0] if model.supported_quants else 0,
        strength=0.0,
        input_path=None,
        output_path=Path("/tmp/_validate_placeholder.png"),
        loras=(),
        mlx_cache_gb=12, battery_stop=20,
        num_frames=1, fps=24,
    )
    defaults.update(gen_overrides)
    return GenParams(**defaults)


def test_validate_engine_params_or_die_passes_when_in_range():
    """No errors → returns None (no die)."""
    from imgen.cmd_helpers import validate_engine_params_or_die
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux-kontext"]
    params = _validate_params(model, quantize=8, guidance=3.5)
    validate_engine_params_or_die(model, params=params)
    # Reached here → no SystemExit


def test_validate_engine_params_or_die_dies_on_guidance_violation(capsys):
    """Out-of-range guidance → SystemExit(2) + stderr names the
    offending value + model.binary."""
    from imgen.cmd_helpers import validate_engine_params_or_die
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux2-klein-edit-9b"]
    params = _validate_params(model, quantize=4, guidance=3.5)
    with pytest.raises(SystemExit) as exc_info:
        validate_engine_params_or_die(model, params=params)
    assert exc_info.value.code == 2
    combined = capsys.readouterr().err
    assert "guidance" in combined
    assert "3.5" in combined


def test_validate_engine_params_or_die_noop_for_user_toml():
    """User TOMLs go through Backend (not Model at commit 7); the
    helper accepts ``model=None`` and no-ops — locked so the user-
    TOML path doesn't accidentally hit per-Model validation that
    doesn't apply to v0.7-shape Backend objects.
    """
    from imgen.cmd_helpers import validate_engine_params_or_die
    from imgen.engines.base import GenParams
    # Any GenParams shape; should not raise because model is None.
    placeholder = GenParams(
        prompt="", negative="", width=64, height=64,
        steps=1, guidance=999.0, seed=0, quantize=99, strength=0.0,
        input_path=None, output_path=Path("/tmp/_x.png"), loras=(),
    )
    validate_engine_params_or_die(None, params=placeholder)

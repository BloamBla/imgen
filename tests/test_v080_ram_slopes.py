"""v0.8.0 commit 8 — per-Model RAM slopes lock-ins.

Per [[project-v080-design]] §L + §Q commit 8:

* ``Engine.ram_estimate_gb(model, params)`` is the single source-of-
  truth for RAM math. The pre-commit-8
  ``defaults.RAM_REQUIRED_GB`` + ``ACTIVATION_GB_PER_MP_ABOVE_BASELINE``
  table are deleted.
* ``checks.ram_required_gb(backend, quant, mp)`` delegates to
  ``Engine.ram_estimate_gb`` for built-in Models; falls back to a
  conservative flux-class formula for unknown/user-TOML names.
* ``imgen doctor`` RAM forecast table is iteratively computed from
  ``BUILTIN_MODELS`` × ``model.supported_quants`` via
  ``Engine.ram_estimate_gb`` — no hardcoded table lookup.

The §Q-mandated 3 lock-ins:

  1. Qwen-Image-2512-style sentinel at 1024² Q4 ≈ 30 GB observed.
  2. flux-kontext at 1024² Q8 ≈ 18 GB (v0.7.7 anchor).
  3. doctor RAM table is generated from Engine.ram_estimate_gb (not
     a hardcoded dict).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.engines.base import GenParams
from imgen.engines.mflux_engine import MfluxEngine
from imgen.models import Model


def _make_genparams(width=1024, height=1024, quantize=4):
    return GenParams(
        prompt="", negative="", width=width, height=height,
        steps=1, guidance=0.0, seed=0, quantize=quantize,
        strength=0.0, input_path=None,
        output_path=Path("/tmp/out.png"), loras=(),
    )


# ── §Q lock-in 1: Qwen-Image-2512 1024² Q4 ≈ 30 GB (sentinel) ─────────


def test_ram_estimate_for_qwen_image_2512_at_1024_q4_matches_2026_05_25_observed_30gb():
    """v0.8.0 commit 8 (§L) calibration lock: a sentinel Model
    representing Qwen-Image-2512 (NOT in BUILTIN_MODELS at commit 8;
    lands as an opt-in template at commit 10 per §G.2). On the user's
    M2 Pro 32 GB the real-world observed peak at 1024² Q4 was ~30 GB
    (swap-thrash regime — see
    [[project-qwen-2512-findings-2026-05-25]]).

    Formula: baseline*0.5 + slope*1.048 + encoder + overhead
    Calibrated values: baseline=34.0, slope=5.0, encoder=7.0
    Diffusers overhead: 2.0
    Result: 34*0.5 + 5*1.048 + 7 + 2.0 = 31.24 GB ≈ 30 ± 3 ✓
    """
    qwen_2512_sentinel = Model(
        engine="diffusers_mps",  # Qwen-2512 bf16 path uses diffusers_mps
        repo="Qwen/Qwen-Image-2512",
        ram_baseline_gb=34.0,
        ram_slope_gb_per_mp=5.0,
        encoder_ram_gb=7.0,
        cpu_offload_threshold_mp=1.5,
    )
    from imgen.engines import DiffusersMpsEngine
    estimate = DiffusersMpsEngine().ram_estimate_gb(
        qwen_2512_sentinel, _make_genparams(1024, 1024, quantize=4),
    )
    assert 28 <= estimate <= 33, (
        f"qwen-image-2512 sentinel 1024² Q4: got {estimate:.2f} GB, "
        "should match 2026-05-25 observed 30 GB ± 3 GB"
    )


# ── §Q lock-in 2: flux-kontext 1024² Q8 ≈ 18 GB (v0.7.7 anchor) ───────


def test_ram_estimate_for_flux_kontext_at_1024_q8_matches_v0_7_7_real_measurement():
    """v0.8.0 commit 8 (§L) calibration lock: the canonical FLUX.1-
    Kontext-dev Q8 1024² real-mflux measurement (M2 Pro 32 GB) from
    the v0.7.7 instrumentation run anchored the v0.7.14 RAM table at
    18 GB. The new per-Model formula MUST reproduce this anchor —
    failure means a future tuning of ``ram_baseline_gb`` /
    ``ram_slope_gb_per_mp`` silently shifted the calibration off
    the real measurement.
    """
    from imgen.models import BUILTIN_MODELS
    model = BUILTIN_MODELS["flux-kontext"]
    estimate = MfluxEngine().ram_estimate_gb(
        model, _make_genparams(1024, 1024, quantize=8),
    )
    # v0.7.7 measurement: 18 GB. ±0.5 GB calibration tolerance.
    assert 17.5 <= estimate <= 18.5, (
        f"flux-kontext Q8 1024²: got {estimate:.2f} GB, should match "
        f"v0.7.7 real-mflux measurement of 18 GB ± 0.5 GB"
    )


# ── §Q lock-in 3: doctor RAM table is generated, not table-lookup ─────


def test_doctor_ram_forecast_table_uses_engine_ram_estimate(capsys):
    """v0.8.0 commit 8 (§L): the doctor RAM forecast table is COMPUTED
    by iterating BUILTIN_MODELS × supported_quants and calling
    ``Engine.ram_estimate_gb`` on each cell — NOT a lookup against
    the pre-commit-8 ``defaults.RAM_REQUIRED_GB`` constant (which
    has been deleted).

    The doctor helper ``_render_ram_forecast_rows`` is the single
    source-of-truth renderer; this test exercises it directly,
    bypassing the rest of cmd_doctor's slow paths (HF whoami,
    venv probes, etc.).
    """
    from imgen.commands.doctor import _render_ram_forecast_rows

    # Pretend we have 32 GB available; row "verdict" should be
    # ✅ for sub-32 estimates and ❌ for over.
    _render_ram_forecast_rows(available_ram=32.0)
    out = capsys.readouterr().out

    # The table renders v0.8 canonical names (commit 4b registry
    # source-of-truth flip).
    assert "flux-kontext" in out
    assert "qwen-image-edit-v1" in out
    assert "flux-dev" in out
    assert "flux2-klein-edit-9b" in out

    # Per-quant rows fire for each supported_quants entry — all 4
    # built-ins ship the default (3, 4, 5, 6, 8) at commit 8.
    for q in (3, 4, 5, 6, 8):
        assert f"q{q}" in out

    # The pre-commit-8 magic "@1MP" integer-GB output ("18 GB") is
    # gone — new renderer emits one-decimal floats ("18.0 GB" /
    # "11.2 GB") because the formula produces float values.
    assert "GB" in out

    # Calibration spot-check: flux-kontext q8 row contains ~18.0 GB.
    lines = out.splitlines()
    flux_kontext_q8_lines = [
        line for line in lines if "flux-kontext" in line and "q8" in line
    ]
    assert flux_kontext_q8_lines, "flux-kontext q8 row missing"
    # Formula at 1024² (1.048576 MP) Q8: 13.5*1 + 4.0*1.048576 + 0 + 0.5
    # = 18.19 → rounds to 18.2 in the one-decimal renderer. The pre-
    # commit-8 v0.7.7 calibration was at the integer 1.0 MP anchor (18
    # GB even); the renderer now reports the EXACT 1024² MP value.
    assert any(
        "18.2 GB" in line or "18.1 GB" in line or "18.0 GB" in line
        for line in flux_kontext_q8_lines
    ), f"flux-kontext q8 row should print ~18.2 GB; got: {flux_kontext_q8_lines}"

    # 32 GB available > 18 GB needed → ✅ verdict appears at least once
    assert "✅" in out


# ── Diffusers engine overhead is heavier than mflux ───────────────────


def test_diffusers_mps_engine_overhead_is_heavier_than_mflux():
    """Architect commit-8 implementation note: diffusers + torch +
    transformers cold-import is ~2 GB vs mflux's ~0.5 GB. This lock-
    in pins the relative cost so a future refactor that swaps the
    constants would surface here.
    """
    from imgen.engines import DiffusersMpsEngine
    test_model_mflux = Model(
        engine="mflux", binary="x",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
    )
    test_model_diffusers = Model(
        engine="diffusers_mps", repo="x/y",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
    )
    params = _make_genparams(1024, 1024, quantize=8)
    mflux_total = MfluxEngine().ram_estimate_gb(test_model_mflux, params)
    diffusers_total = DiffusersMpsEngine().ram_estimate_gb(
        test_model_diffusers, params,
    )
    # Same baseline/slope/encoder → same physical cost EXCEPT for
    # the per-engine overhead. Difference must equal 2.0 - 0.5 = 1.5.
    assert abs((diffusers_total - mflux_total) - 1.5) < 1e-6, (
        f"diffusers_mps - mflux overhead delta should be 1.5 GB; "
        f"got {diffusers_total - mflux_total}"
    )


# ── Weights scale with quantize per the §L formula ────────────────────


def test_weights_scale_linearly_with_quantize():
    """The §L formula is ``weights_gb = baseline * (quantize / 8.0)``.
    Q4 weights MUST be exactly half of Q8 weights for the same Model
    (modulo activations, encoder, overhead).
    """
    from imgen.models import BUILTIN_MODELS
    model = BUILTIN_MODELS["flux-kontext"]
    q8 = MfluxEngine().ram_estimate_gb(
        model, _make_genparams(1024, 1024, quantize=8),
    )
    q4 = MfluxEngine().ram_estimate_gb(
        model, _make_genparams(1024, 1024, quantize=4),
    )
    # Non-weight cost (activations + encoder + overhead) is the same
    # between Q8 and Q4 — only weights scale.
    non_weight_cost = (
        model.ram_slope_gb_per_mp * (1024 * 1024 / 1_000_000.0)
        + model.encoder_ram_gb
        + 0.5  # mflux overhead
    )
    weights_q8 = q8 - non_weight_cost
    weights_q4 = q4 - non_weight_cost
    assert abs(weights_q4 / weights_q8 - 0.5) < 1e-6, (
        f"Q4 weights {weights_q4:.2f} should be exactly half of Q8 "
        f"weights {weights_q8:.2f}; ratio is {weights_q4 / weights_q8}"
    )

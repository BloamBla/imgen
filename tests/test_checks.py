"""checks.py — resource preflight calculation surface.

v0.7.14 → v0.8.0 commit 8 (§L) closure: ``ram_required_gb`` shifted
from a per-``(backend, quant)`` table lookup (RAM_REQUIRED_GB +
ACTIVATION_GB_PER_MP_ABOVE_BASELINE constants in defaults.py) to a
per-Model formula sourced from ``Engine.ram_estimate_gb``:

    weights = baseline * (quantize / 8)
    activations = slope * megapixels
    encoder = one-time peak from VLM encoder load
    total = weights + activations + encoder + engine_overhead

Calibration anchors (locked here against real measurements):

* flux-kontext Q8 1MP ≈ 18 GB — v0.7.7 real-mflux smoke on M2 Pro 32 GB.
* flux2-klein-edit-9b Q4 at the v0.7.7 measured (1536², 2048²) points
  must still land within ±3 GB.
* qwen-image-edit-v1 Q8 1MP includes the ~7 GB Qwen2.5-VL encoder
  peak — total ≈ 25 GB matching v0.7.14's calibration row.

These tests lock the formula against the real measurements the user
performed on their M2 Pro 32 GB rig, so a future tweak of
``Model.ram_baseline_gb`` / ``ram_slope_gb_per_mp`` / ``encoder_ram_gb``
can't silently regress.
"""
from __future__ import annotations

import pytest

from imgen.checks import ram_required_gb


# ── v0.8.0 commit 8: Per-Model Q8 1MP calibration anchors ─────────────


@pytest.mark.parametrize("model_name,quant,expected,tolerance", [
    # flux-kontext Q8 1MP — locked to v0.7.7 real-mflux smoke.
    # Formula: 13.5 * 1.0 + 4.0 * 1.0 + 0.0 + 0.5 = 18.0
    ("flux-kontext", 8, 18.0, 0.5),
    # flux-kontext Q4 1MP — half-weights + same activations.
    # Formula: 13.5 * 0.5 + 4.0 * 1.0 + 0.0 + 0.5 = 11.25
    ("flux-kontext", 4, 11.25, 0.5),
    # flux-dev shares the FLUX.1 transformer envelope.
    ("flux-dev", 8, 18.0, 0.5),
    # qwen-image-edit-v1 Q8 1MP — includes 7 GB Qwen2.5-VL encoder.
    # Formula: 13.0 * 1.0 + 4.5 * 1.0 + 7.0 + 0.5 = 25.0
    ("qwen-image-edit-v1", 8, 25.0, 0.5),
    # flux2-klein-edit-9b Q8 1MP — heavier baseline (9B params).
    # Formula: 27.0 * 1.0 + 4.0 * 1.0 + 0.0 + 0.5 = 31.5
    ("flux2-klein-edit-9b", 8, 31.5, 0.5),
])
def test_ram_required_gb_q8_1mp_matches_v08_calibration(
    model_name, quant, expected, tolerance,
):
    """v0.8.0 commit 8 (§L): per-Model RAM formula calibrated against
    v0.7.7 real-mflux measurements + Qwen2.5-VL encoder accounting.
    These per-model anchors lock the formula's per-quant + per-encoder
    behaviour. Floating-point tolerance ±0.5 GB to absorb int/float
    arithmetic noise — physical RAM measurement has wider variance
    anyway (~1-2 GB swing run-to-run).
    """
    actual = ram_required_gb(
        model_name, quant, megapixels=1024 * 1024 / 1_000_000,
    )
    assert abs(actual - expected) <= tolerance, (
        f"{model_name} Q{quant} 1MP: got {actual:.2f} GB, "
        f"expected {expected} ± {tolerance} GB"
    )


# ── v0.7.7 real-measurement anchors for flux2-klein-edit-9b ────────────


def test_flux2_klein_q4_at_1024sq_unblocks_under_24gb():
    """v0.7.14 (gap 6 LIVE-CONFIRMED 2026-05-24): user got blocked on
    ``imgen photo.jpg --backend flux2-klein-edit-9b`` at 1024² because
    the pre-v0.7.14 row was the 2K² worst-case (24 GB). At 1024² the
    activations are ~3× smaller; estimate must drop below 24 GB so
    the 23.3 GB-available scenario passes preflight.

    v0.8.0 commit 8: per-Model formula keeps this property — at
    Q4 1MP for flux2-klein the new formula gives 27*0.5 + 4*1 + 0 +
    0.5 = 18 GB ≪ 24.
    """
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=1024 * 1024 / 1_000_000,
    )
    assert estimate < 24, (
        f"flux2-klein-edit-9b Q4 @ 1024² must drop below the pre-v0.7.14 "
        f"24 GB ceiling to unblock the live-confirmed failure mode; "
        f"got {estimate:.2f}"
    )


def test_flux2_klein_q4_at_1536sq_matches_v077_measurement():
    """v0.7.7 calibration: real measurement on M2 Pro 32 GB at
    1536² (~49 min run) showed ~23 GB resident peak. The formula must
    land within ±3 GB of that ground truth.

    v0.8.0 formula: 27*0.5 + 4*2.36 + 0 + 0.5 = 13.5 + 9.44 + 0.5 = 23.44 GB ✓
    """
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=1536 * 1536 / 1_000_000,
    )
    assert 20 <= estimate <= 26, (
        f"flux2-klein-edit-9b Q4 @ 1536² should track the v0.7.7 "
        f"23 GB measurement within ±3 GB; got {estimate:.2f}"
    )


def test_flux2_klein_q4_at_2048sq_matches_v077_measurement():
    """v0.7.7 calibration: real measurement on M2 Pro 32 GB at
    2048² (~110 min run) showed ~30 GB total memory pressure (resident
    + compressed + swap). The formula must land within ±3 GB of that.

    v0.8.0 formula: 27*0.5 + 4*4.19 + 0 + 0.5 = 13.5 + 16.76 + 0.5 = 30.76 GB ✓
    """
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=2048 * 2048 / 1_000_000,
    )
    assert 27 <= estimate <= 33, (
        f"flux2-klein-edit-9b Q4 @ 2048² should track the v0.7.7 "
        f"30 GB measurement within ±3 GB; got {estimate:.2f}"
    )


# ── Monotonic scaling above 1 MP ──────────────────────────────────────


def test_ram_grows_with_megapixels_above_1mp():
    """Above the 1 MP canonical, activations grow ~linearly with
    megapixels. The function MUST be monotonically non-decreasing in
    megapixels — a regression here would silently let a larger
    resolution slip through a preflight that blocked the smaller one.

    v0.8.0 commit 8 (§L): no change to monotonicity — the formula's
    ``slope * mp`` term is monotonically increasing in mp.
    """
    mp_seq = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    estimates = [
        ram_required_gb("flux-kontext", 8, mp) for mp in mp_seq
    ]
    for i in range(1, len(estimates)):
        assert estimates[i] >= estimates[i - 1], (
            f"RAM estimate dropped from {estimates[i-1]:.2f} GB at "
            f"{mp_seq[i-1]} MP to {estimates[i]:.2f} GB at {mp_seq[i]} MP — "
            "monotonic non-decreasing required"
        )


def test_ram_below_1mp_scales_smaller_activations():
    """v0.8.0 commit 8 (§L): the new formula uses ``slope * mp``
    DIRECTLY (no 1MP clamping). Sub-1MP outputs (preview 768²,
    256² thumbs) honestly report smaller activations — weights +
    encoder + overhead stay constant, but activations shrink.

    Pre-commit-8 the function clamped sub-MP to 1MP-baseline; that
    clamp was a v0.7.14 artefact of the table-baseline approach
    (where 1MP totals were bundled). New formula doesn't need the
    clamp — physical RAM at sub-1MP IS lower.
    """
    baseline = ram_required_gb("flux-kontext", 4, megapixels=1.0)
    sub_mp = ram_required_gb("flux-kontext", 4, megapixels=0.5)
    # Sub-MP is smaller (honest physics) but still positive
    assert sub_mp > 0
    assert sub_mp < baseline, (
        f"sub-1MP estimate {sub_mp} should be smaller than 1MP "
        f"baseline {baseline} — only activations scale, but they do "
        "scale down at sub-1MP. v0.7.14 clamping is gone at commit 8."
    )


# ── Unknown backend / quant fallback (v0.8.0 conservative flux-class) ─


def test_unknown_backend_uses_flux_class_fallback():
    """v0.8.0 commit 8 (§L): unknown model names (typos, future user-
    TOML registered names without v0.8 schema fields) fall back to a
    conservative flux-class Model (baseline 13.5, slope 4.0, no
    encoder, mflux overhead). The pre-commit-8 ``.get(..., 16)``
    magic-number floor is gone — the fallback is now physics-driven.

    Q4 1MP flux-class: 13.5 * 0.5 + 4.0 * 1 + 0 + 0.5 = 11.25 GB.
    """
    estimate = ram_required_gb(
        "totally-unknown-model", 4, megapixels=1.0,
    )
    assert 8 <= estimate <= 14, (
        f"unknown-model Q4 1MP should land near flux-class baseline "
        f"~11 GB; got {estimate:.2f}"
    )


def test_unknown_backend_still_scales_with_megapixels():
    """Fallback uses the same monotonic per-mp slope so a 2K² unknown
    backend gets a larger estimate than a 1MP one."""
    base = ram_required_gb("unknown", 4, megapixels=1.0)
    big = ram_required_gb("unknown", 4, megapixels=4.0)
    assert big > base, (
        f"unknown backend at 4MP {big:.2f} should exceed 1MP "
        f"baseline {base:.2f}"
    )


# ── v0.7.15 (architect Q6 advisory): megapixels_of helper ──────────────


class TestMegapixelsOf:
    """Extracted from 4 copy-pasted ``(w * h) / 1_000_000`` call sites
    (cmd_generate, cmd_batch, cmd_refine, cmd_draw) in v0.7.15. Pure
    helper — single tested seam eliminates copy-paste typo risk."""

    def test_1024sq_gives_canonical_1_048mp(self):
        """1024² = 2¹⁰ × 2¹⁰ = 1,048,576 pixels = 1.048576 MP exactly.
        This is the canonical 1 MP baseline anchor for `ram_required_gb`."""
        from imgen.cmd_helpers import megapixels_of
        assert megapixels_of(1024, 1024) == 1.048576

    def test_1536sq_matches_v077_calibration_point(self):
        """1536² used in the v0.7.7 flux2-klein-edit Q4 measurement."""
        from imgen.cmd_helpers import megapixels_of
        # 1536² = 2,359,296 → 2.359296 MP
        assert megapixels_of(1536, 1536) == 2.359296

    def test_2048sq_matches_v077_calibration_point(self):
        """2048² used in the v0.7.7 flux2-klein-edit Q4 measurement."""
        from imgen.cmd_helpers import megapixels_of
        # 2048² = 4,194,304 → 4.194304 MP
        assert megapixels_of(2048, 2048) == 4.194304

    def test_returns_float_not_int(self):
        """Even dimensions that produce an integer MP must return
        float — `ram_required_gb` formulas expect float arithmetic."""
        from imgen.cmd_helpers import megapixels_of
        # 1000 × 1000 = 1,000,000 → 1 MP exactly. Integer-valued
        # result but still float-typed.
        result = megapixels_of(1000, 1000)
        assert result == 1.0
        assert isinstance(result, float)

    def test_non_square_dimensions(self):
        """1920 × 1080 (HD) = 2,073,600 → 2.0736 MP. Locks the
        non-square path that refine + draw users with `--width`/
        `--height` overrides may hit."""
        from imgen.cmd_helpers import megapixels_of
        assert megapixels_of(1920, 1080) == 2.0736


# ── v0.10.0 commit 7 — get_battery pmset absolute path (security H-3) ──


class TestGetBatteryAbsolutePmsetPath:
    """§R.1 security H-3 closure: ``checks.get_battery`` hardcodes
    ``/usr/bin/pmset`` instead of relying on $PATH lookup.

    A $PATH-hijack attack on a compromised parent could otherwise
    redirect the pmset call to a malicious binary that returns
    fake battery readings (e.g. always 100% — letting an overnight
    training run continue past the safe battery threshold)."""

    def test_argv_uses_absolute_path(self, monkeypatch):
        import subprocess
        recorded = {"argv": None}

        def fake_check_output(cmd, *args, **kwargs):
            recorded["argv"] = list(cmd)
            return b"Battery Power\n -InternalBattery-0 (id=12345)\t85%; discharging; 4:00 remaining present: true\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        from imgen.checks import get_battery
        get_battery()
        actual = recorded["argv"][0]
        assert actual == "/usr/bin/pmset", (
            "§R.1 security H-3: get_battery must use absolute "
            f"/usr/bin/pmset path; got argv[0]={actual!r}"
        )

    def test_returns_desktop_sentinel_on_missing_pmset(self, monkeypatch):
        """Defence-in-depth: if /usr/bin/pmset is missing (broken
        macOS install, sandbox), fall back to (None, True) — same
        as the pre-§R.1 FileNotFoundError handling."""
        import subprocess

        def boom(cmd, *args, **kwargs):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(subprocess, "check_output", boom)
        from imgen.checks import get_battery
        pct, on_ac = get_battery()
        assert pct is None
        assert on_ac is True

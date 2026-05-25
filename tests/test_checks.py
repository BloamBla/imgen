"""checks.py — resource preflight calculation surface.

v0.7.14 (gap 6 closure): dimension-aware RAM estimation via
`ram_required_gb(backend, quant, megapixels)`. Pre-v0.7.14 the
preflight used a fixed `RAM_REQUIRED_GB[(backend, quant)]` table
calibrated against worst-case (2K²) for flux2-klein-edit-9b, which
over-blocked 1024² runs on 32 GB Macs that fit comfortably.

These tests lock the new function's behaviour against the v0.7.7
calibration data points the user actually measured on their M2 Pro
32 GB rig, so a future tweak of the formula can't silently regress.
"""
from __future__ import annotations

import pytest

from imgen.checks import ram_required_gb


# ── Backend/quant baseline (1 MP) preserved ─────────────────────────────


@pytest.mark.parametrize("backend,quant,expected", [
    # FLUX.1-Kontext-dev (i2i): pre-v0.7.14 1MP-canonical values kept.
    ("flux", 3, 8),
    ("flux", 4, 9),
    ("flux", 5, 12),
    ("flux", 6, 14),
    ("flux", 8, 18),
    # Qwen-Image-Edit: same.
    ("qwen", 3, 10),
    ("qwen", 4, 12),
    ("qwen", 5, 16),
    ("qwen", 6, 18),
    ("qwen", 8, 25),
    # FLUX.1-dev (t2i): shares FLUX.1 transformer envelope.
    ("flux-dev", 3, 8),
    ("flux-dev", 4, 9),
    ("flux-dev", 5, 12),
    ("flux-dev", 6, 14),
    ("flux-dev", 8, 18),
    # FLUX.2-klein-edit-9B (i2i refine backend). v0.7.14: rows
    # reverse-extrapolated to 1 MP baselines per v0.7.7 calibration
    # slope (was 2K² worst-case pre-v0.7.14). Q4 anchored to real
    # measurement at 1.05 MP; Q3 / Q5 / Q6 / Q8 extrapolated linearly
    # from quantization weight delta around the Q4 anchor. Lock-in
    # so a future table edit can't silently regress these.
    ("flux2-klein-edit-9b", 3, 12),
    ("flux2-klein-edit-9b", 4, 14),
    ("flux2-klein-edit-9b", 5, 16),
    ("flux2-klein-edit-9b", 6, 18),
    ("flux2-klein-edit-9b", 8, 20),
])
def test_ram_required_gb_at_1mp_matches_baseline(backend, quant, expected):
    """Existing flux / qwen / flux-dev backends keep their pre-v0.7.14
    1MP RAM estimates. The dimension-aware function only adds an
    activation slope for resolutions above 1 MP; the 1 MP baseline
    stays identical to what `RAM_REQUIRED_GB[(backend, quant)]`
    previously returned for these rows."""
    assert ram_required_gb(backend, quant, megapixels=1.0) == expected


# ── flux2-klein-edit-9b: v0.7.7 real measurements anchor the slope ──────


def test_flux2_klein_q4_at_1024sq_unblocks_under_24gb():
    """v0.7.14 (gap 6 LIVE-CONFIRMED 2026-05-24): user got blocked on
    `imgen photo.jpg --backend flux2-klein-edit-9b` at 1024² because
    the pre-v0.7.14 row was the 2K² worst-case (24 GB). At 1024² the
    activations are ~3× smaller; estimate must drop below 24 GB so
    the 23.3 GB-available scenario passes preflight."""
    # 1024² = 1.048 MP — the actual canonical baseline for this row.
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=1024 * 1024 / 1_000_000,
    )
    assert estimate < 24, (
        f"flux2-klein-edit-9b Q4 @ 1024² must drop below the old "
        f"worst-case 24 GB to unblock the live-confirmed failure mode; "
        f"got {estimate}"
    )


def test_flux2_klein_q4_at_1536sq_matches_v077_measurement():
    """v0.7.7 calibration: real measurement on M2 Pro 32 GB at
    1536² (~49 min run) showed ~23 GB resident peak. The formula must
    land within ±3 GB of that ground truth."""
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=1536 * 1536 / 1_000_000,
    )
    assert 20 <= estimate <= 26, (
        f"flux2-klein-edit-9b Q4 @ 1536² should track the v0.7.7 "
        f"23 GB measurement within ±3 GB; got {estimate}"
    )


def test_flux2_klein_q4_at_2048sq_matches_v077_measurement():
    """v0.7.7 calibration: real measurement on M2 Pro 32 GB at
    2048² (~110 min run) showed ~30 GB total memory pressure (resident
    + compressed + swap). The formula must land within ±3 GB of that."""
    estimate = ram_required_gb(
        "flux2-klein-edit-9b", 4, megapixels=2048 * 2048 / 1_000_000,
    )
    assert 27 <= estimate <= 33, (
        f"flux2-klein-edit-9b Q4 @ 2048² should track the v0.7.7 "
        f"30 GB measurement within ±3 GB; got {estimate}"
    )


# ── Linear scaling above 1 MP ───────────────────────────────────────────


def test_ram_grows_with_megapixels_above_1mp():
    """Above the 1 MP canonical, activations grow ~linearly with
    megapixels. The function MUST be monotonically non-decreasing in
    megapixels — a regression here would silently let a larger
    resolution slip through a preflight that blocked the smaller one."""
    mp_seq = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    estimates = [ram_required_gb("flux", 8, mp) for mp in mp_seq]
    for i in range(1, len(estimates)):
        assert estimates[i] >= estimates[i - 1], (
            f"RAM estimate dropped from {estimates[i-1]} GB at "
            f"{mp_seq[i-1]} MP to {estimates[i]} GB at {mp_seq[i]} MP — "
            "monotonic non-decreasing required"
        )


def test_ram_below_1mp_does_not_underestimate_baseline():
    """Sub-1 MP outputs (e.g. preview 768², 256² thumbs) shouldn't drop
    below the 1 MP baseline — the weight footprint + text encoders +
    MLX cache stay constant; only activations shrink. The function
    must clamp to the baseline as the lower bound."""
    baseline = ram_required_gb("flux", 4, megapixels=1.0)
    sub_mp = ram_required_gb("flux", 4, megapixels=0.5)
    assert sub_mp >= baseline, (
        f"sub-1MP estimate {sub_mp} dropped below 1MP baseline {baseline}"
    )


# ── Unknown backend / quant fallback ────────────────────────────────────


def test_unknown_backend_returns_conservative_fallback():
    """Unknown (backend, quant) combos fall back to the same 16 GB
    conservative estimate the pre-v0.7.14 dict-lookup provided. The
    fallback is the 'don't accidentally let an unbenchmarked backend
    crash a 16GB Mac' floor — keep it."""
    assert ram_required_gb("unknown-backend", 4, megapixels=1.0) == 16
    assert ram_required_gb("flux", 99, megapixels=1.0) == 16


def test_unknown_backend_still_scales_with_megapixels():
    """Fallback must also apply the activation slope so a 2K² unknown
    backend doesn't get the same 16 GB estimate as a 1 MP one."""
    base = ram_required_gb("unknown-backend", 4, megapixels=1.0)
    big = ram_required_gb("unknown-backend", 4, megapixels=4.0)
    assert big > base, (
        f"unknown-backend at 4MP {big} should exceed 1MP baseline {base}"
    )

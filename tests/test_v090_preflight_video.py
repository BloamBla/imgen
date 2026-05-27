"""v0.9 commit 7.1 (§R.2 HIGH-1) — preflight RAM math threads num_frames
+ enforces §L "+3 GB video safety buffer".

Three reviewers (python, security, architect) all flagged this:
the v0.8 preflight pipeline builds placeholder GenParams with
num_frames=1 (the GenParams default), so DiffusersMpsEngine.
_ram_estimate_video's ``0.1 * num_frames`` term silently returns 0.1
GB regardless of the actual video length. For 25-frame LTX the
under-count is 2.5 GB; for 121-frame it's 12 GB. The §L "+3 GB safety
buffer for video" gate is also missing — image-shape preflight used.

Combined gap: 25-frame LTX run silently under-reports headroom by
~5.5 GB. §A.5 "OOM on M2 Pro 32 GB" risk is functionally unmitigated
until this lands.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _ltx_video_config():
    from imgen.models import VideoConfig
    return VideoConfig(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
        force_cpu_offload=True,
    )


def _ltx_model_for_ram(**overrides):
    """LTX-shaped Model for ram_estimate. Same defaults as the
    BUILTIN_MODELS row but constructible without registry monkeypatch."""
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        supported_quants=(),
        video=_ltx_video_config(),
    )
    defaults.update(overrides)
    return Model(**defaults)


# ── ram_required_gb num_frames threading ──────────────────────────────


class TestRamRequiredGbThreadsNumFrames:
    """ram_required_gb must thread num_frames into the GenParams it
    builds, so DiffusersMpsEngine._ram_estimate_video's frame_term
    reflects the actual video length."""

    def test_video_25_frames_matches_design_envelope(self):
        """§L canonical: LTX 768×512 × 25 frames ≈ 17.07 GB.
        Pre-fix: returns ~14.7 GB (off by 2.5 GB)."""
        from imgen.checks import ram_required_gb
        # 768×512 = 0.393 MP
        est = ram_required_gb(
            "ltx-video", quantize=0, megapixels=0.393, num_frames=25,
        )
        assert 16.5 <= est <= 17.5, (
            f"25-frame LTX should match §L envelope ~17.07 GB; got {est:.2f}"
        )

    def test_video_more_frames_scales_estimate(self):
        """Frame-term is 0.1 GB per frame — 121 frames adds ~10 GB
        vs 25 frames."""
        from imgen.checks import ram_required_gb
        est_25 = ram_required_gb(
            "ltx-video", quantize=0, megapixels=0.393, num_frames=25,
        )
        est_121 = ram_required_gb(
            "ltx-video", quantize=0, megapixels=0.393, num_frames=121,
        )
        delta = est_121 - est_25
        # 96 extra frames × 0.1 GB = 9.6 GB
        assert 9.0 <= delta <= 10.0, (
            f"frame-term should scale linearly; got delta={delta:.2f}"
        )

    def test_image_num_frames_default_1_unchanged(self):
        """Image preflight (no num_frames arg) reads num_frames=1 by
        default — same shape as pre-fix. Lock-in for regression."""
        from imgen.checks import ram_required_gb
        # flux-kontext at 1024² Q8 should match v0.7.7 anchor (~18 GB)
        est = ram_required_gb(
            "flux-kontext", quantize=8, megapixels=1.048576,
        )
        assert 17.5 <= est <= 18.5, (
            f"image preflight regression — got {est:.2f}, want ~18.0"
        )


# ── §L +3 GB video buffer in preflight_resources ──────────────────────


class TestPreflightVideoBuffer:
    """§L: ``available_gb < ram_estimate + 3.0`` for video Models —
    dies with code 4 unless --force. Image Models keep the existing
    no-buffer comparison."""

    def _make_args(self, force=False):
        return SimpleNamespace(force=force)

    def _stub_check_resources(self, monkeypatch, *, required, available,
                              total=32.0):
        """Replace check_resources with a fixture returning the given
        RAM numbers + always-OK disk/battery/no-parallel-mflux."""
        import imgen.cmd_helpers as cmd_mod

        def fake(*args, **kwargs):
            return {
                "ram_required_gb": required,
                "ram_total_gb": total,
                "ram_available_gb": available,
                "ram_ok": total == 0 or available >= required,
                "disk_free_gb": 200.0,
                "disk_ok": True,
                "battery_pct": 100,
                "on_ac": True,
                "battery_ok": True,
                "other_mflux_pid": None,
            }
        monkeypatch.setattr(cmd_mod, "check_resources", fake)

    def test_video_with_3gb_buffer_passes_when_headroom_above(
        self, monkeypatch,
    ):
        """Required 17 GB, available 22 GB — 22 ≥ 17 + 3 = 20 ✓"""
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=17.0, available=22.0)
        # Must not raise
        preflight_resources(
            model="ltx-video", heaviest_quant=0, force=False,
            max_megapixels=0.393, max_num_frames=25,
        )

    def test_video_with_3gb_buffer_dies_when_headroom_below(
        self, monkeypatch, capsys,
    ):
        """Required 17 GB, available 19 GB — 19 < 17 + 3 = 20 ✗"""
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=17.0, available=19.0)
        with pytest.raises(SystemExit):
            preflight_resources(
                model="ltx-video", heaviest_quant=0, force=False,
                max_megapixels=0.393, max_num_frames=25,
            )
        stderr = capsys.readouterr().err
        assert "RAM" in stderr or "ram" in stderr

    def test_video_die_hint_includes_video_specific_knobs(
        self, monkeypatch, capsys,
    ):
        """v0.9.2 B-8 closure of §R.2 UX consistency: the video
        preflight die hint must mirror the image 'How to fix' bullet
        shape with video-appropriate knobs (--width/--height,
        --num-frames/--duration, --force). LTX has no --preview or
        --quantize so those don't appear; lower-resolution and
        shorter-clip are the equivalent dials.
        """
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=17.0, available=19.0)
        with pytest.raises(SystemExit):
            preflight_resources(
                model="ltx-video", heaviest_quant=0, force=False,
                max_megapixels=0.393, max_num_frames=25,
            )
        stderr = capsys.readouterr().err
        assert "How to fix" in stderr, (
            f"video hint must use the image-shape 'How to fix' header; "
            f"got: {stderr!r}"
        )
        assert "--width" in stderr and "--height" in stderr, (
            f"video hint must surface resolution knobs; got: {stderr!r}"
        )
        assert "--num-frames" in stderr and "--duration" in stderr, (
            f"video hint must surface clip-length knobs; got: {stderr!r}"
        )
        assert "--force" in stderr, (
            f"video hint must mention --force escape hatch; got: {stderr!r}"
        )

    def test_video_force_bypasses_buffer_check(self, monkeypatch):
        """--force skips all preflight checks INCLUDING the new video
        buffer. v0.8.2 RAM safety net (< 4 GB hard floor) is checked
        separately in subprocess_helpers, not here."""
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=30.0, available=2.0)
        # Force bypasses entirely
        preflight_resources(
            model="ltx-video", heaviest_quant=0, force=True,
            max_megapixels=0.393, max_num_frames=25,
        )

    def test_image_no_3gb_buffer_applied(self, monkeypatch):
        """Image preflight (max_num_frames=1) keeps the existing
        ``available >= required`` semantics — no +3 video buffer."""
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=18.0, available=19.0)
        # 19 ≥ 18 → passes for image (no buffer)
        preflight_resources(
            model="flux-kontext", heaviest_quant=8, force=False,
            max_megapixels=1.048576, max_num_frames=1,
        )

    def test_image_no_max_num_frames_arg_defaults_to_1(self, monkeypatch):
        """Backwards-compat: callers without max_num_frames keep
        v0.8.x semantics."""
        from imgen.cmd_helpers import preflight_resources
        self._stub_check_resources(monkeypatch, required=18.0, available=19.0)
        # No max_num_frames kwarg — image default
        preflight_resources(
            model="flux-kontext", heaviest_quant=8, force=False,
            max_megapixels=1.048576,
        )

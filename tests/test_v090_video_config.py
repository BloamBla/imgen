"""v0.9 commit 1 — VideoConfig dataclass shape + __post_init__ invariants.

Per [[project-v090-design]] §C. ``VideoConfig`` is a nested
dataclass on ``Model.video`` that flags a Model as producing video
output. Absent (``Model.video is None``) ⇒ image Model. Present ⇒
video Model.

Architect §R.1 HIGH-1 verdict: flat field expansion (7 new top-level
Model fields) is wrong because (a) image-only user TOMLs would carry
7 noise fields meaningless for image; (b) future audio/3d Models
would compound the flat bloat. Nested keeps Model at fixed top-level
cardinality.

These tests lock the VideoConfig field surface + validation matrix
so v0.9.x can't silently drop a field or relax a guard.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest


def _valid_ltx_video_config(**overrides):
    """Canonical LTX-Video VideoConfig — smallest valid construction.
    Tests start from this and mutate one field at a time."""
    from imgen.models import VideoConfig
    defaults = dict(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
    )
    defaults.update(overrides)
    return VideoConfig(**defaults)


class TestVideoConfigShape:
    """Lock the v0.9.0 VideoConfig field surface."""

    def test_minimal_video_config_instantiates(self):
        vc = _valid_ltx_video_config()
        assert vc.default_num_frames == 25
        assert vc.default_fps == 24
        assert vc.max_num_frames == 257

    def test_video_config_default_alignment_is_8(self):
        """LTX default — 8k+1 frame structure."""
        vc = _valid_ltx_video_config()
        assert vc.num_frames_alignment == 8
        assert vc.num_frames_offset == 1

    def test_video_config_default_codec_libx264(self):
        vc = _valid_ltx_video_config()
        assert vc.supports_video_codecs == ("libx264",)

    def test_video_config_default_force_cpu_offload_true(self):
        """Video defaults to forced offload (T5-XXL RAM pressure)."""
        vc = _valid_ltx_video_config()
        assert vc.force_cpu_offload is True

    def test_video_config_default_encoder_ram_gb_3(self):
        """T5-XXL transient peak when CPU-offloaded — not optional."""
        vc = _valid_ltx_video_config()
        assert vc.encoder_ram_gb == 3.0

    def test_video_config_is_frozen(self):
        """frozen=True per §C — attribute reassignment must raise."""
        vc = _valid_ltx_video_config()
        with pytest.raises(FrozenInstanceError):
            vc.default_num_frames = 33  # type: ignore[misc]

    def test_video_config_is_hashable(self):
        """frozen+slots+hashable types only — usable as dict key /
        set member (Model is frozen so all nested fields must be too)."""
        vc = _valid_ltx_video_config()
        # smoke: must not raise
        hash(vc)
        assert {vc, vc} == {vc}

    def test_video_config_field_surface_locked(self):
        """Schema lock — v0.9.x can't silently drop a field."""
        from imgen.models import VideoConfig
        names = {f.name for f in fields(VideoConfig)}
        expected = {
            "default_num_frames",
            "default_fps",
            "max_num_frames",
            "num_frames_alignment",
            "num_frames_offset",
            "supports_video_codecs",
            "force_cpu_offload",
            "encoder_ram_gb",
        }
        assert expected == names, (
            f"Field surface drift: missing={expected - names}, "
            f"extra={names - expected}"
        )


class TestVideoConfigPostInit:
    """§C __post_init__ matrix — every guard tested by mutating one
    field of an otherwise-valid VideoConfig."""

    def test_default_num_frames_below_9_raises(self):
        """Minimum 9 for usable temporal sampling per §C."""
        with pytest.raises(ValueError, match="default_num_frames"):
            _valid_ltx_video_config(default_num_frames=8)

    def test_default_num_frames_at_8_raises(self):
        with pytest.raises(ValueError, match="default_num_frames"):
            _valid_ltx_video_config(default_num_frames=8)

    def test_default_num_frames_at_9_accepted(self):
        """Boundary: 9 IS the floor, must be accepted."""
        vc = _valid_ltx_video_config(default_num_frames=9, max_num_frames=17)
        assert vc.default_num_frames == 9

    def test_fps_not_in_allowlist_raises(self):
        """v0.9.0 supports {24, 25, 30} per §C."""
        with pytest.raises(ValueError, match="default_fps"):
            _valid_ltx_video_config(default_fps=60)

    def test_fps_15_raises(self):
        with pytest.raises(ValueError, match="default_fps"):
            _valid_ltx_video_config(default_fps=15)

    def test_fps_24_accepted(self):
        vc = _valid_ltx_video_config(default_fps=24)
        assert vc.default_fps == 24

    def test_fps_25_accepted(self):
        vc = _valid_ltx_video_config(default_fps=25)
        assert vc.default_fps == 25

    def test_fps_30_accepted(self):
        vc = _valid_ltx_video_config(default_fps=30)
        assert vc.default_fps == 30

    def test_max_num_frames_below_default_raises(self):
        """max < default would let the validator accept then reject the
        same default — internal inconsistency."""
        with pytest.raises(ValueError, match="max_num_frames"):
            _valid_ltx_video_config(default_num_frames=25, max_num_frames=17)

    def test_max_num_frames_equal_default_accepted(self):
        """Boundary: max == default IS valid (model with single supported length)."""
        vc = _valid_ltx_video_config(default_num_frames=25, max_num_frames=25)
        assert vc.max_num_frames == 25

    def test_num_frames_alignment_below_1_raises(self):
        with pytest.raises(ValueError, match="num_frames_alignment"):
            _valid_ltx_video_config(num_frames_alignment=0)

    def test_num_frames_alignment_negative_raises(self):
        with pytest.raises(ValueError, match="num_frames_alignment"):
            _valid_ltx_video_config(num_frames_alignment=-1)

    def test_num_frames_alignment_1_accepted(self):
        """alignment=1 = no alignment requirement (any frame count OK)."""
        vc = _valid_ltx_video_config(num_frames_alignment=1)
        assert vc.num_frames_alignment == 1

    def test_empty_supports_video_codecs_raises(self):
        with pytest.raises(ValueError, match="supports_video_codecs"):
            _valid_ltx_video_config(supports_video_codecs=())

    def test_encoder_ram_gb_zero_raises(self):
        """encoder_ram_gb=0 means text encoder is free — never true for
        any real video model (T5-XXL is the dominant cost for LTX)."""
        with pytest.raises(ValueError, match="encoder_ram_gb"):
            _valid_ltx_video_config(encoder_ram_gb=0.0)

    def test_encoder_ram_gb_negative_raises(self):
        with pytest.raises(ValueError, match="encoder_ram_gb"):
            _valid_ltx_video_config(encoder_ram_gb=-1.0)

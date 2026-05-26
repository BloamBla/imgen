"""v0.9 commit 1 — Model.video field + cross-rules + output_type property.

Per [[project-v090-design]] §C. ``Model.video`` is the new optional
nested VideoConfig field. Image Models leave it None. Video Models
populate it. Cross-rules in ``__post_init__`` enforce:

1. video set ⇒ engine must be ``"diffusers_mps"`` (no mflux video in v0.9.0).
2. video set ⇒ cpu_offload_threshold_mp must stay at the harmless
   image default 2.0 (force_cpu_offload on VideoConfig is the
   single source of truth for video offloading).

The ``output_type`` derived property surfaces a literal ``"image"``
or ``"video"`` string for callers (``iteration_dryrun_display`` §H,
runner dispatch §F, history ``command`` field §J).

Backwards-compat lock-in per §C closing: every v0.8 BUILTIN_MODELS
row instantiates without setting ``video=`` → defaults to None →
v0.9 __post_init__ rules are no-ops for them.
"""
from __future__ import annotations

import pytest


def _valid_ltx_video_config(**overrides):
    from imgen.models import VideoConfig
    defaults = dict(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
    )
    defaults.update(overrides)
    return VideoConfig(**defaults)


def _minimal_diffusers_mps_video_model(**overrides):
    """Smallest valid video Model — diffusers_mps engine + valid VideoConfig."""
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        video=_valid_ltx_video_config(),
    )
    defaults.update(overrides)
    return Model(**defaults)


class TestModelVideoField:
    """v0.9 widens Model with optional video: VideoConfig | None."""

    def test_image_model_video_defaults_to_none(self):
        """v0.8 instantiation pattern (no video=) keeps working."""
        from imgen.models import Model
        m = Model(
            engine="mflux",
            binary="mflux-generate",
            ram_baseline_gb=9.0,
            ram_slope_gb_per_mp=5.0,
        )
        assert m.video is None

    def test_video_model_carries_video_config(self):
        m = _minimal_diffusers_mps_video_model()
        assert m.video is not None
        assert m.video.default_num_frames == 25
        assert m.video.default_fps == 24

    def test_video_field_is_in_field_surface(self):
        """v0.9 widens the v0.8 field-surface lock with `video`."""
        from dataclasses import fields
        from imgen.models import Model
        names = {f.name for f in fields(Model)}
        assert "video" in names, (
            "Model.video field missing — v0.9 widening lost"
        )


class TestModelOutputType:
    """Derived property: "image" if video is None else "video"."""

    def test_image_model_output_type_image(self):
        from imgen.models import Model
        m = Model(
            engine="mflux",
            binary="mflux-generate",
            ram_baseline_gb=9.0,
            ram_slope_gb_per_mp=5.0,
        )
        assert m.output_type == "image"

    def test_video_model_output_type_video(self):
        m = _minimal_diffusers_mps_video_model()
        assert m.output_type == "video"

    def test_output_type_is_literal_string_not_enum(self):
        """Per §C: output_type returns Literal["image", "video"] —
        a plain string, NOT an enum. Callers compare with ==."""
        from imgen.models import Model
        m_image = Model(
            engine="mflux",
            binary="mflux-generate",
            ram_baseline_gb=9.0,
            ram_slope_gb_per_mp=5.0,
        )
        m_video = _minimal_diffusers_mps_video_model()
        assert isinstance(m_image.output_type, str)
        assert isinstance(m_video.output_type, str)


class TestModelVideoPostInitCrossRules:
    """§C cross-rules — every guard tested by mutating one field of
    an otherwise-valid Model."""

    def test_video_with_mflux_engine_raises(self):
        """v0.9.0: video Models require diffusers_mps. mflux subprocess
        path has no video support."""
        from imgen.models import Model
        with pytest.raises(ValueError, match="diffusers_mps"):
            Model(
                engine="mflux",
                binary="mflux-generate",
                ram_baseline_gb=10.0,
                ram_slope_gb_per_mp=4.0,
                video=_valid_ltx_video_config(),
            )

    def test_video_with_explicit_cpu_offload_threshold_raises(self):
        """force_cpu_offload on VideoConfig is single source of truth
        for video offloading; cpu_offload_threshold_mp is image-only.
        Mixing both is a footgun — refuse it."""
        from imgen.models import Model
        with pytest.raises(ValueError, match="cpu_offload_threshold_mp"):
            Model(
                engine="diffusers_mps",
                repo="Lightricks/LTX-Video",
                ram_baseline_gb=10.0,
                ram_slope_gb_per_mp=4.0,
                cpu_offload_threshold_mp=4.0,  # non-default → reject
                video=_valid_ltx_video_config(),
            )

    def test_video_with_default_cpu_offload_threshold_accepted(self):
        """The default 2.0 is harmless — image-only field carrying its
        image-only default. Video Models keep it at default."""
        from imgen.models import Model
        m = Model(
            engine="diffusers_mps",
            repo="Lightricks/LTX-Video",
            ram_baseline_gb=10.0,
            ram_slope_gb_per_mp=4.0,
            cpu_offload_threshold_mp=2.0,  # explicit default
            video=_valid_ltx_video_config(),
        )
        assert m.video is not None

    def test_video_with_diffusers_mps_engine_accepted(self):
        """Happy path — diffusers_mps + valid VideoConfig is the v0.9
        canonical video Model shape."""
        m = _minimal_diffusers_mps_video_model()
        assert m.engine == "diffusers_mps"
        assert m.video is not None

    def test_image_diffusers_mps_model_still_works(self):
        """diffusers_mps Models without video= are still image Models."""
        from imgen.models import Model
        m = Model(
            engine="diffusers_mps",
            repo="Qwen/Qwen-Image-2512",
            ram_baseline_gb=24.0,
            ram_slope_gb_per_mp=8.0,
        )
        assert m.video is None
        assert m.output_type == "image"


class TestV08BuiltinsUnaffectedByV09FieldAddition:
    """§C closing lock-in: every v0.8 built-in instantiates without
    `video=`, defaults to None, v0.9 __post_init__ rules are no-ops.

    Required so v0.9 commit 1 cannot accidentally break the existing
    v0.8 registry. If this fires, the v0.9 field addition introduced
    an incompatible default or mistakenly required video= as a
    positional arg.
    """

    def test_v08_builtins_load_after_v09_field_addition(self):
        """BUILTIN_MODELS imports at module load — if v0.9 broke v0.8
        rows, the import itself would raise. Re-prove explicitly."""
        from imgen.models import BUILTIN_MODELS
        assert len(BUILTIN_MODELS) >= 4

    def test_every_v08_builtin_has_video_none(self):
        """v0.8 rows are image Models — video defaults to None
        unconditionally."""
        from imgen.models import BUILTIN_MODELS
        for name, m in BUILTIN_MODELS.items():
            assert m.video is None, (
                f"v0.8 built-in {name!r} unexpectedly has video set "
                "(v0.9 commit 1 should not touch v0.8 rows)"
            )

    def test_every_v08_builtin_output_type_image(self):
        from imgen.models import BUILTIN_MODELS
        for name, m in BUILTIN_MODELS.items():
            assert m.output_type == "image", (
                f"v0.8 built-in {name!r} reports output_type={m.output_type!r}, "
                "expected 'image' (video field defaulted to None)"
            )

"""v0.9 commit 3 — DiffusersMpsEngine.validate video branch + ram_estimate.

Per [[project-v090-design]] §E.0 + §E.1 + §L.

§E.0 — v0.8.1 §R.4 M-4 closure REVERSED:
  Pre-v0.9, ``DiffusersMpsEngine.validate`` was an intentional no-op
  with all payload validation delegated to the runner trust boundary.
  v0.9 REOPENS this: video-specific checks (range, alignment, fps
  allowlist) fire parent-side BEFORE subprocess spawn so misuse is
  rejected at the ~50ms parser gate instead of after ~3-5s of cold
  imports. The runner-side checks land in commit 4 as defence-in-
  depth; this commit owns the parent-side surface plus the matrix
  lock-in. Image-path validation stays no-op (existing image checks
  live in MfluxEngine.validate for the mflux paths).

§E.1 — _validate_video rules:
  * num_frames range: [default_num_frames // 2, max_num_frames]
  * alignment: (num_frames - num_frames_offset) % num_frames_alignment == 0
    Error message includes nearest valid value (architect §R.1 MED-2).
  * fps allowlist: {24, 25, 30}

§L — ram_estimate video branch:
  baseline + slope*mp + video.encoder_ram_gb + 0.1*num_frames
  (No quantize term — diffusers doesn't quantize LTX at v0.9.0.)
  (No +2.0 overhead term — baseline absorbs cold-import footprint
  per §L "T5 offloaded baseline" definition.)
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _ltx_video_config(**overrides):
    from imgen.models import VideoConfig
    defaults = dict(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
    )
    defaults.update(overrides)
    return VideoConfig(**defaults)


def _ltx_model(**overrides):
    """LTX-shaped Model with the §K production VideoConfig.
    Engine='diffusers_mps', ram_* per §L LTX envelope.

    supported_quants=() matches the production BUILTIN_MODELS row —
    LTX-Video is bf16-only at v0.9.0, no MLX-style quantization. The
    §R.2 architect HIGH-2 quantize gate keys off this empty tuple.
    """
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,        # §L "T5 offloaded baseline"
        ram_slope_gb_per_mp=4.0,
        supported_quants=(),
        video=_ltx_video_config(),
    )
    defaults.update(overrides)
    return Model(**defaults)


def _image_diffusers_model(**overrides):
    """Image-only diffusers Model (no VideoConfig) — backwards-compat
    regression fixture."""
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="Qwen/Qwen-Image-2512",
        ram_baseline_gb=24.0,
        ram_slope_gb_per_mp=8.0,
    )
    defaults.update(overrides)
    return Model(**defaults)


def _video_params(**overrides):
    """LTX-shaped GenParams: 768×512 @ 25 frames / 24 fps default.

    ``quantize=0`` default reflects LTX bf16-only semantics — paired
    with the supported_quants=() gate in _ltx_model so canonical
    "valid LTX" payloads pass the quantize check.
    """
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="a samurai walking", negative="",
        width=768, height=512,
        steps=50, guidance=3.0, seed=42, quantize=0, strength=0.0,
        input_path=None, output_path=Path("/tmp/out.mp4"), loras=(),
        num_frames=25, fps=24,
    )
    defaults.update(overrides)
    return GenParams(**defaults)


# ── §E.0 — parent-side validation reopened for video ───────────────────


class TestEngineValidateImagePathUnchanged:
    """v0.8.1 §R.4 M-4 closure stays in effect for IMAGE Models —
    only video branches reopen the validate surface."""

    def test_image_model_validate_returns_empty(self):
        """Image diffusers Model (model.video is None) — no parent-side
        validation, matches pre-v0.9 behaviour."""
        from imgen.engines import DiffusersMpsEngine
        engine = DiffusersMpsEngine()
        m = _image_diffusers_model()
        params = _video_params(num_frames=1, fps=24)  # image defaults
        assert engine.validate(m, params) == []

    def test_image_model_validate_returns_empty_even_with_silly_inputs(self):
        """No image-side checks added; previous no-op pattern preserved."""
        from imgen.engines import DiffusersMpsEngine
        engine = DiffusersMpsEngine()
        m = _image_diffusers_model()
        # Wild fps + num_frames values — irrelevant for image Models
        # (fps/num_frames silently ignored at runner-image path).
        params = _video_params(num_frames=1, fps=24)
        assert engine.validate(m, params) == []


class TestEngineValidateVideoNumFramesRange:
    """§E.1 — num_frames must be in [default_num_frames // 2, max_num_frames]."""

    def test_canonical_ltx_25_frames_accepted(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=24),
        )
        assert errors == []

    def test_at_max_num_frames_257_accepted(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=257, fps=24),
        )
        assert errors == []

    def test_exceeds_max_num_frames_rejected(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=258, fps=24),
        )
        assert errors, "258 frames exceeds max_num_frames=257; expected reject"
        assert any("num_frames" in e and "257" in e for e in errors)

    def test_below_default_half_rejected(self):
        """default_num_frames=25 → half=12. num_frames=1 below floor."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=1, fps=24),
        )
        assert errors, "1 frame below default_num_frames/2; expected reject"
        assert any("num_frames" in e for e in errors)


class TestEngineValidateVideoAlignment:
    """§E.1 — (num_frames - offset) % alignment == 0. LTX: 8k+1 frames."""

    def test_alignment_violation_rejected(self):
        """10 frames: (10-1)=9; 9 % 8 = 1 → violation."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=10, fps=24),
        )
        assert errors, "10 frames violates 8k+1; expected reject"
        assert any("alignment" in e.lower() for e in errors)

    def test_alignment_error_includes_nearest_valid(self):
        """Architect §R.1 MED-2: error message must surface nearest
        valid value so users can fix without consulting docs."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=10, fps=24),
        )
        # Nearest valid <= 10 in 8k+1 sequence: 9 (k=1)
        joined = " ".join(errors)
        assert "9" in joined, (
            f"alignment error must include nearest valid (9); got: {errors}"
        )

    def test_9_frames_accepted(self):
        """9 = 8*1 + 1; valid 8k+1. But below default_num_frames/2=12,
        so this alone tests alignment NOT rejecting — the range check
        catches it instead."""
        from imgen.engines import DiffusersMpsEngine
        # Custom model with lower default_num_frames so 9 passes range
        model = _ltx_model(video=_ltx_video_config(
            default_num_frames=9, max_num_frames=17,
        ))
        errors = DiffusersMpsEngine().validate(
            model, _video_params(num_frames=9, fps=24),
        )
        assert errors == [], (
            f"9 frames is 8k+1 valid + within range; got errors: {errors}"
        )

    def test_17_frames_accepted(self):
        """17 = 8*2 + 1; valid 8k+1."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=17, fps=24),
        )
        assert errors == []


class TestEngineValidateVideoQuantizeGate:
    """§R.2 architect HIGH-2 closure: video Models with
    ``supported_quants=()`` must reject non-zero ``params.quantize``.
    LTX-Video is bf16-only at v0.9.0 — no MLX-style quantization.
    A user-TOML video Model leaving quantize at the v0.7 default of 4
    would silently propagate a meaningless field; reject loudly."""

    def test_video_with_quantize_4_rejected_when_supported_quants_empty(self):
        """Default quantize=4 from merged_defaults flowing into a
        video Model with supported_quants=() — reject."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(quantize=4),
        )
        assert errors, "quantize=4 on supported_quants=() must reject"
        assert any("quantize" in e.lower() for e in errors)

    def test_video_with_quantize_8_rejected_when_supported_quants_empty(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(quantize=8),
        )
        assert errors, "quantize=8 on supported_quants=() must reject"

    def test_video_with_quantize_0_accepted(self):
        """quantize=0 means "no quantize" — the bf16-only semantic.
        Acceptable even when supported_quants=()."""
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(quantize=0),
        )
        assert errors == []


class TestEngineValidateVideoFpsAllowlist:
    """§E.1 — fps in {24, 25, 30}."""

    def test_fps_24_accepted(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=24),
        )
        assert errors == []

    def test_fps_25_accepted(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=25),
        )
        assert errors == []

    def test_fps_30_accepted(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=30),
        )
        assert errors == []

    def test_fps_60_rejected(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=60),
        )
        assert errors, "60 fps out of allowlist; expected reject"
        assert any("fps" in e for e in errors)

    def test_fps_15_rejected(self):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=25, fps=15),
        )
        assert errors


# ── §E.0 drift-matrix lock-in — pins the rejection set in one place ───


class TestEngineValidateVideoDriftMatrix:
    """§E.0 lock-in fixtures from the design memo. Parametrized so any
    future change to _validate_video must update this matrix
    deliberately — a silent drift in reject/accept rules surfaces here.

    Format: (num_frames, fps, expected_reject, reason).

    The runner-side leg (driving same fixtures through
    ``_validate_payload_shape``) lands in commit 4 once that function
    learns the optional video keys. Commit 3 owns the parent-side
    half of the drift lock.
    """

    @pytest.mark.parametrize(
        "num_frames,fps,expected_reject,reason",
        [
            (25, 24, False, "canonical-valid LTX"),
            (10, 24, True, "alignment violation (10 % 8 != 1)"),
            (257, 24, False, "at max_num_frames"),
            (258, 24, True, "exceeds max_num_frames"),
            (25, 60, True, "fps out of range"),
            (1, 24, True, "below default_num_frames/2"),
        ],
    )
    def test_parent_side_rejection_matches_matrix(
        self, num_frames, fps, expected_reject, reason,
    ):
        from imgen.engines import DiffusersMpsEngine
        errors = DiffusersMpsEngine().validate(
            _ltx_model(),
            _video_params(num_frames=num_frames, fps=fps),
        )
        actual_reject = bool(errors)
        assert actual_reject == expected_reject, (
            f"({num_frames=}, {fps=}, {reason=!r}): "
            f"expected reject={expected_reject}, got errors={errors}"
        )


# ── §L — ram_estimate_gb video branch ──────────────────────────────────


class TestRamEstimateImageBranchUnchanged:
    """Image branch behaviour preserved by v0.9 widening. The existing
    formula (weights = baseline * quantize/8, +activations +encoder
    +2.0 overhead) keeps firing when model.video is None."""

    def test_image_diffusers_model_uses_quantize_weighted_formula(self):
        from imgen.engines import DiffusersMpsEngine
        m = _image_diffusers_model()
        # 1024² @ Q4: weights = 24*0.5 = 12; slope*1.048 = 8.39; encoder=0; overhead=2
        # → 12 + 8.39 + 0 + 2 ≈ 22.39 GB
        params = _video_params(width=1024, height=1024, num_frames=1, quantize=4)
        est = DiffusersMpsEngine().ram_estimate_gb(m, params)
        assert 21.5 <= est <= 23.5, (
            f"image branch should match Q4 quantize-weighted formula; got {est:.2f}"
        )


class TestRamEstimateVideoBranch:
    """§L LTX envelope: 17 GB at 768×512 × 25 frames is the anchor."""

    def test_ram_estimate_ltx_768x512_25frames_matches_design_envelope(self):
        """§L canonical anchor:
          baseline 10 + slope 4*0.393 + encoder 3 + frame 0.1*25
          = 10 + 1.57 + 3 + 2.5 = 17.07 GB
        ±0.5 GB tolerance for floating-point + future tuning headroom.
        """
        from imgen.engines import DiffusersMpsEngine
        est = DiffusersMpsEngine().ram_estimate_gb(
            _ltx_model(),
            _video_params(width=768, height=512, num_frames=25),
        )
        assert 16.5 <= est <= 17.5, (
            f"LTX 768×512 × 25 frames: got {est:.2f} GB, "
            "expected ~17.07 GB per §L envelope"
        )

    def test_ram_estimate_scales_linearly_with_num_frames(self):
        """frame_term = 0.1 * num_frames. Doubling frames adds ~2.5 GB
        at 25→49 (within 0.1 GB of exact)."""
        from imgen.engines import DiffusersMpsEngine
        engine = DiffusersMpsEngine()
        m = _ltx_model()
        est_25 = engine.ram_estimate_gb(m, _video_params(num_frames=25))
        est_49 = engine.ram_estimate_gb(m, _video_params(num_frames=49))
        delta = est_49 - est_25
        # 24 extra frames × 0.1 GB = 2.4 GB exact
        assert 2.3 <= delta <= 2.5, (
            f"ram delta 25→49 frames: got {delta:.2f}, expected ~2.4"
        )

    def test_ram_estimate_uses_video_encoder_not_model_encoder(self):
        """§K design: ``Model.encoder_ram_gb`` is image-only; for video
        Models ``VideoConfig.encoder_ram_gb`` is the authoritative
        source. Verify by setting them to incompatible values."""
        from imgen.engines import DiffusersMpsEngine
        # Model.encoder_ram_gb=99 (image-field); VideoConfig.encoder_ram_gb=3
        m = _ltx_model(
            encoder_ram_gb=99.0,  # ignored when video is set
            video=_ltx_video_config(encoder_ram_gb=3.0),
        )
        est = DiffusersMpsEngine().ram_estimate_gb(
            m, _video_params(width=768, height=512, num_frames=25),
        )
        # If we used model.encoder_ram_gb=99 we'd get ~113 GB. If we
        # correctly use video.encoder_ram_gb=3 we get ~17 GB.
        assert est < 20.0, (
            f"ram_estimate should use VideoConfig.encoder_ram_gb, "
            f"not Model.encoder_ram_gb; got {est:.2f} GB"
        )

    def test_ram_estimate_video_no_quantize_term(self):
        """LTX bf16-only (supported_quants=()) — formula must NOT scale
        with params.quantize for video Models."""
        from imgen.engines import DiffusersMpsEngine
        engine = DiffusersMpsEngine()
        m = _ltx_model()
        est_q4 = engine.ram_estimate_gb(m, _video_params(quantize=4))
        est_q8 = engine.ram_estimate_gb(m, _video_params(quantize=8))
        # Video branch ignores quantize → identical estimates
        assert abs(est_q4 - est_q8) < 1e-6, (
            f"video branch must not scale with quantize; got Q4={est_q4}, Q8={est_q8}"
        )

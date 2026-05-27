"""v0.9 commit 11.3 — Parent ↔ runner validation parity (§E.0 closure).

Per [[project-v090-design]] §E.0: the v0.8.1 §R.4 M-4 closure
"DiffusersMpsEngine.validate is intentionally a no-op stub" was
REOPENED for video at v0.9.0. Parent-side validation (Engine.validate)
fires inside ``validate_engine_params_or_die`` BEFORE subprocess
spawn (~50ms gate); runner-side ``_validate_payload_shape`` mirrors
the same checks as defence-in-depth.

The §E.0 design explicitly mandated a lock-in test that drives BOTH
layers with the same fixtures so any future drift between the two
surfaces fails this test. §R.2 mid-arc review noted the absence of
this test (python MEDIUM-1, also covered by architect WATCH); commit
11.3 closes it.

This test is the §E.0-canonical drift-prevention lock. If a future
edit to DiffusersMpsEngine._validate_video changes the acceptance
matrix without a parallel change to _diffusers_runner._validate_
payload_shape (or vice versa), the parametrized fixture below
surfaces the mismatch.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _ltx_video_config():
    from imgen.models import VideoConfig
    return VideoConfig(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
        force_cpu_offload=True,
    )


def _ltx_model():
    from imgen.models import Model
    return Model(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        supported_quants=(),
        video=_ltx_video_config(),
    )


def _video_params(num_frames: int, fps: int):
    from imgen.engines.base import GenParams
    return GenParams(
        prompt="a samurai", negative="",
        width=768, height=512,
        steps=25, guidance=3.0, seed=42, quantize=0, strength=0.0,
        input_path=None, output_path=Path("/tmp/out.mp4"),
        loras=(),
        num_frames=num_frames, fps=fps,
    )


def _runner_payload(num_frames: int, fps: int) -> dict:
    """Mirror of what DiffusersMpsEngine.run constructs for video
    payloads. The runner-side ``_validate_payload_shape`` consumes
    this dict shape via JSON-on-stdin."""
    return {
        "repo": "Lightricks/LTX-Video",
        "prompt": "a samurai",
        "negative": "",
        "steps": 25,
        "guidance": 3.0,
        "width": 768,
        "height": 512,
        "seed": 42,
        "output_path": "/tmp/out.mp4",
        "num_frames": num_frames,
        "fps": fps,
        "output_type": "video",
        "pipeline_class": "LTXPipeline",
        "force_cpu_offload": True,
        "param_overrides": {},
    }


# ── §E.0 drift-matrix lock-in ──────────────────────────────────────────


class TestValidationParityParentVsRunner:
    """The §E.0-canonical drift lock. Drives both layers with the same
    matrix; any disagreement on reject/accept surfaces here."""

    @pytest.mark.parametrize(
        "num_frames,fps,expected_reject,reason",
        [
            # Canonical-valid LTX — both layers accept.
            (25, 24, False, "canonical-valid LTX"),
            (257, 24, False, "at max_num_frames"),
            (49, 25, False, "valid at higher fps"),
            (17, 30, False, "valid 8k+1 at 30 fps"),
            # fps out of allowlist — both reject.
            (25, 60, True, "fps out of {24,25,30}"),
            (25, 15, True, "fps below allowlist"),
            (25, 23, True, "fps not in allowlist (close miss)"),
            # num_frames lower-bound parity (v0.9.1 B-12 closure of
            # §R.3 r2 python NIT-1): both layers must reject 0. The
            # =1 case lives in TestKnownDivergence below because it
            # exposes the per-model minimum (parent rejects via
            # VideoConfig, runner accepts via basic 1..1024 sanity).
            (0, 24, True, "num_frames below sanity minimum"),
        ],
    )
    def test_parent_and_runner_agree_on_matrix(
        self, num_frames, fps, expected_reject, reason,
    ):
        """For each (num_frames, fps) row, parent and runner must
        produce the same reject/accept decision. The two layers
        own different surface — parent reads ``model.video`` for
        per-Model rules (alignment, max_num_frames); runner uses
        hardcoded basic checks (fps allowlist, num_frames sanity
        range 1..1024). Where their concerns OVERLAP (fps allowlist
        + simple range), they must agree.
        """
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape

        parent_errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames, fps),
        )
        parent_reject = bool(parent_errors)

        runner_rc = _validate_payload_shape(
            _runner_payload(num_frames, fps),
        )
        runner_reject = runner_rc != 0

        assert parent_reject == runner_reject == expected_reject, (
            f"DRIFT detected for ({num_frames=}, {fps=}, {reason=!r}): "
            f"parent_reject={parent_reject} ({parent_errors!r}), "
            f"runner_reject={runner_reject} (rc={runner_rc}), "
            f"expected_reject={expected_reject}"
        )


# ── Non-overlapping surfaces — documented divergence ──────────────────


class TestKnownDivergence:
    """Parent and runner check DIFFERENT subsets of the contract.
    Parent has model.video access (per-Model alignment + max_num_frames);
    runner has only the payload (basic range + fps allowlist). Where
    one rejects but the other accepts, the divergence is documented
    here so a future refactor doesn't accidentally "fix" what's by
    design.
    """

    def test_alignment_violation_caught_only_by_parent(self):
        """LTX alignment 8k+1: 10 frames violates (10-1)=9, 9%8=1.
        Parent rejects (knows model.video.num_frames_alignment).
        Runner accepts (no alignment field in payload schema)."""
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape

        parent_errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=10, fps=24),
        )
        runner_rc = _validate_payload_shape(_runner_payload(10, 24))

        # Parent catches alignment
        assert parent_errors, "parent must reject alignment violation"
        # Runner doesn't see per-Model alignment — accepts
        assert runner_rc == 0, (
            "runner schema has no alignment field; accepts 10 frames"
        )

    def test_per_model_max_caught_only_by_parent(self):
        """LTX max_num_frames=257. 258 violates per-Model cap.
        Parent rejects. Runner only enforces basic 1..1024 sanity
        cap so 258 is accepted at the runner side."""
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape

        parent_errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=258, fps=24),
        )
        runner_rc = _validate_payload_shape(_runner_payload(258, 24))

        assert parent_errors, "parent must reject 258 > max=257"
        assert runner_rc == 0, "runner sanity cap 1024 accepts 258"

    def test_per_model_min_caught_only_by_parent(self):
        """v0.9.1 B-12 lower-bound divergence: parent enforces a
        per-Model floor derived from VideoConfig.default_num_frames
        (~12 for LTX's default of 25). Runner only enforces the basic
        sanity floor of 1. ``num_frames=1`` therefore rejects at the
        parent layer but accepts at the runner — same structural
        divergence as ``num_frames=258`` (max side) and alignment."""
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape

        parent_errors = DiffusersMpsEngine().validate(
            _ltx_model(), _video_params(num_frames=1, fps=24),
        )
        runner_rc = _validate_payload_shape(_runner_payload(1, 24))

        assert parent_errors, "parent must reject num_frames=1 < per-Model floor"
        assert runner_rc == 0, "runner sanity floor 1 accepts 1"

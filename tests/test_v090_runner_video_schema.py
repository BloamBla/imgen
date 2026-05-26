"""v0.9 commit 4 — _diffusers_runner._validate_payload_shape video extensions.

Per [[project-v090-design]] §F + §E.0 (drift lock-in completes here
on the runner side).

The runner-side schema gains five OPTIONAL keys to support video
payloads. None are unconditionally required (image payloads keep
working without them), but ``output_type=="video"`` triggers
conditional required-ness for ``num_frames``, ``fps``, and
``pipeline_class``. Bool-vs-int discipline preserved across all
numeric fields.

Security boundary (security §R.1 HIGH-1):
``pipeline_class`` is validated via a LITERAL allowlist BEFORE the
diffusers import. ``getattr``-on-module is the anti-pattern this
avoids — even a typoed allowlist entry would fail closed because
the dict lookup raises KeyError.

Defence-in-depth: ``_SAFE_OUTPUT_EXTS`` in ``_diffusers_runner.py``
duplicates the source-of-truth ``paths.SAFE_OUTPUT_EXTS`` (no imgen
imports in ``.venv-diffusers/`` so the runner stays self-contained).
A lock-in test pins them to identical sets to catch drift.
"""
from __future__ import annotations

import pytest


def _valid_image_payload(**overrides):
    """Pre-v0.9 image payload — the existing happy path."""
    p = dict(
        repo="Lightricks/LTX-Video",
        prompt="a samurai",
        negative="",
        steps=50,
        guidance=3.0,
        width=768,
        height=512,
        seed=42,
        output_path="/tmp/out.png",
    )
    p.update(overrides)
    return p


def _valid_video_payload(**overrides):
    """v0.9 video payload — required v0.8 keys + output_type + the
    triple (num_frames, fps, pipeline_class) required when video."""
    p = dict(
        repo="Lightricks/LTX-Video",
        prompt="a samurai",
        negative="",
        steps=50,
        guidance=3.0,
        width=768,
        height=512,
        seed=42,
        output_path="/tmp/out.mp4",
        num_frames=25,
        fps=24,
        output_type="video",
        pipeline_class="LTXPipeline",
        force_cpu_offload=True,
    )
    p.update(overrides)
    return p


# ── Backwards compat ──────────────────────────────────────────────────


class TestImagePayloadStillAccepted:
    """v0.8 image payloads (no v0.9 keys) keep passing — backwards
    compat invariant."""

    def test_pre_v09_image_payload_passes(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(_valid_image_payload()) == 0

    def test_image_payload_with_explicit_output_type_image_passes(self):
        """Allowed: explicit output_type='image' for clarity."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_image_payload(output_type="image"),
        ) == 0

    def test_image_payload_with_png_extension_passes(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_image_payload(output_path="/tmp/x.png"),
        ) == 0


# ── output_type allowlist ─────────────────────────────────────────────


class TestOutputTypeAllowlist:
    """output_type must be in {"image", "video"} when present."""

    def test_output_type_image_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_image_payload(output_type="image"),
        ) == 0

    def test_output_type_video_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(_valid_video_payload()) == 0

    def test_output_type_unknown_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(output_type="audio"),
        )
        assert rc != 0

    def test_output_type_int_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_image_payload(output_type=1),  # type: ignore[arg-type]
        )
        assert rc != 0

    def test_output_type_none_rejected(self):
        """JSON null for output_type is not the same as absent — explicit
        null fails the str-only check."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_image_payload(output_type=None),
        )
        assert rc != 0


# ── pipeline_class allowlist (SECURITY-CRITICAL) ──────────────────────


class TestPipelineClassAllowlist:
    """Security §R.1 HIGH-1: pipeline_class must be validated via a
    LITERAL allowlist BEFORE any diffusers import. The runner must
    REJECT introspection / path-traversal / dunder strings without
    importing diffusers, so even a buggy import order can't widen
    the attack surface."""

    def test_pipeline_class_ltx_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(_valid_video_payload()) == 0

    def test_pipeline_class_diffusion_pipeline_accepted(self):
        """DiffusionPipeline is the generic v0.8 fallback — kept in
        the allowlist for image-via-video parity."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(pipeline_class="DiffusionPipeline"),
        ) == 0

    def test_pipeline_class_dunder_class_rejected(self):
        """``__class__`` getattr would reach object.__class__ at the
        Python attribute level. Allowlist must reject the literal."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(pipeline_class="__class__"),
        )
        assert rc != 0

    def test_pipeline_class_dunder_bases_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(
                pipeline_class="DiffusionPipeline.__bases__",
            ),
        )
        assert rc != 0

    def test_pipeline_class_path_traversal_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(pipeline_class="../../etc/passwd"),
        )
        assert rc != 0

    def test_pipeline_class_empty_string_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(pipeline_class=""),
        )
        assert rc != 0

    def test_pipeline_class_int_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(pipeline_class=42),  # type: ignore[arg-type]
        )
        assert rc != 0

    def test_pipeline_class_unknown_diffusers_class_rejected(self):
        """A real-but-unallowlisted class name fails — only entries on
        the literal allowlist may flow into _resolve_pipeline_class."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        rc = _validate_payload_shape(
            _valid_video_payload(pipeline_class="StableDiffusionPipeline"),
        )
        assert rc != 0


# ── Conditional required keys for video payloads ──────────────────────


class TestVideoConditionalRequired:
    """When output_type=="video", three additional keys become
    required: num_frames, fps, pipeline_class. Image payloads
    don't carry these and that's fine."""

    def test_video_without_num_frames_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _valid_video_payload()
        del payload["num_frames"]
        assert _validate_payload_shape(payload) != 0

    def test_video_without_fps_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _valid_video_payload()
        del payload["fps"]
        assert _validate_payload_shape(payload) != 0

    def test_video_without_pipeline_class_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _valid_video_payload()
        del payload["pipeline_class"]
        assert _validate_payload_shape(payload) != 0

    def test_video_without_force_cpu_offload_accepted(self):
        """force_cpu_offload is OPTIONAL; defaults to False when absent."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _valid_video_payload()
        del payload["force_cpu_offload"]
        assert _validate_payload_shape(payload) == 0


# ── num_frames type + range ───────────────────────────────────────────


class TestNumFramesValidation:

    def test_num_frames_int_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=25),
        ) == 0

    def test_num_frames_float_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=25.0),  # type: ignore[arg-type]
        ) != 0

    def test_num_frames_bool_rejected(self):
        """Bool subclasses int but is semantically wrong (and would
        slip through a naive ``isinstance(v, int)`` check)."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=True),  # type: ignore[arg-type]
        ) != 0

    def test_num_frames_zero_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=0),
        ) != 0

    def test_num_frames_negative_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=-25),
        ) != 0

    def test_num_frames_above_cap_rejected(self):
        """Sanity cap 1024 — covers any video pipeline. Per-Model cap
        (model.video.max_num_frames) enforced at parent-side validate
        (commit 3)."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(num_frames=2048),
        ) != 0


# ── fps allowlist ─────────────────────────────────────────────────────


class TestFpsValidation:

    @pytest.mark.parametrize("fps", [24, 25, 30])
    def test_fps_in_allowlist_accepted(self, fps):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(fps=fps),
        ) == 0

    def test_fps_60_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(fps=60),
        ) != 0

    def test_fps_bool_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(fps=True),  # type: ignore[arg-type]
        ) != 0

    def test_fps_float_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(fps=24.0),  # type: ignore[arg-type]
        ) != 0


# ── force_cpu_offload bool ────────────────────────────────────────────


class TestForceCpuOffloadValidation:

    def test_force_cpu_offload_true_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(force_cpu_offload=True),
        ) == 0

    def test_force_cpu_offload_false_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(force_cpu_offload=False),
        ) == 0

    def test_force_cpu_offload_int_rejected(self):
        """Bool-not-int discipline — int 1 is not a valid bool here."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(force_cpu_offload=1),  # type: ignore[arg-type]
        ) != 0

    def test_force_cpu_offload_string_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(force_cpu_offload="true"),  # type: ignore[arg-type]
        ) != 0


# ── Output path extension × output_type matrix ────────────────────────


class TestOutputPathExtensionByType:
    """Image payloads must use .png/.jpg/.jpeg/.webp.
    Video payloads must use .mp4. Cross-combinations rejected."""

    def test_video_with_mp4_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(output_path="/tmp/x.mp4"),
        ) == 0

    def test_video_with_png_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_video_payload(output_path="/tmp/x.png"),
        ) != 0

    def test_image_with_mp4_rejected(self):
        """Image output_type with .mp4 — schema must reject the
        mismatch before reaching PIL.save which would fail
        cryptically."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_image_payload(output_path="/tmp/x.mp4"),
        ) != 0

    def test_image_with_webp_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        assert _validate_payload_shape(
            _valid_image_payload(output_path="/tmp/x.webp"),
        ) == 0


# ── SAFE_OUTPUT_EXTS structural lock-in ───────────────────────────────


class TestSafeOutputExtsDefenceInDepth:
    """v0.8.0 §E.1: paths.SAFE_OUTPUT_EXTS is the source of truth.
    The runner-local frozenset duplicates it because the .venv-
    diffusers/ venv can't import imgen.paths. Drift between them is
    a silent security regression — lock-in pins them identical."""

    def test_runner_safe_output_exts_equals_paths_safe_output_exts(self):
        from imgen.paths import SAFE_OUTPUT_EXTS as paths_set
        from imgen.engines._diffusers_runner import _SAFE_OUTPUT_EXTS as runner_set
        assert set(paths_set) == set(runner_set), (
            f"SAFE_OUTPUT_EXTS drift between paths.py and "
            f"_diffusers_runner.py: paths={sorted(paths_set)}, "
            f"runner={sorted(runner_set)}"
        )

    def test_mp4_in_paths_safe_output_exts(self):
        """v0.9 commit 4: .mp4 added for MP4 video output."""
        from imgen.paths import SAFE_OUTPUT_EXTS
        assert ".mp4" in SAFE_OUTPUT_EXTS

    def test_mp4_in_runner_safe_output_exts(self):
        from imgen.engines._diffusers_runner import _SAFE_OUTPUT_EXTS
        assert ".mp4" in _SAFE_OUTPUT_EXTS

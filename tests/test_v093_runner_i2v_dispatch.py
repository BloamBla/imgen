"""v0.9.3 C4 — runner + engine i2v dispatch tests.

Per [[project-v093-i2v]] C4. Covers:

* Schema accepts the v0.9.3 i2v shape: ``output_type=="video"`` +
  ``input_path`` set. The video branch tightens the input-extension
  allowlist to {.png, .jpg, .jpeg} (parent-side ``_i2v_resolve``
  uses the same set; defense-in-depth re-checks at the trust
  boundary).
* Schema rejects i2v with .webp / .heic input — these extensions
  pass the image-i2i input gate (because mflux+PIL accept them) but
  must be refused on the video path where LTX VAE's behaviour on
  those formats hasn't been smoke-verified.
* ``_run_video`` threads the loaded conditioning image into
  ``pipe(image=...)`` when ``input_path`` is present in the payload.
  Done via diffusers' ``load_image`` (same path as the image i2i
  branch in ``_run_image``).
* Engine: when ``params.input_path`` is set on a video Model,
  ``image_path`` shows up in the runner payload alongside the other
  video fields. The Engine doesn't decide the pipeline_class itself —
  that's already on ``VideoConfig.pipeline_class`` per C2 (B-1
  closure); ``cmd_video`` (C5) constructs the i2v-flavoured Model
  via ``dataclasses.replace`` before reaching the engine.

These tests use mocks at the pipe boundary; the real LTX subprocess
runs only via the pre-tag real-smoke matrix on M2 Pro hardware.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _i2v_video_payload(**overrides):
    p = dict(
        repo="Lightricks/LTX-Video",
        prompt="wind blows softly",
        negative="static, still, frozen, no motion",
        steps=50,
        guidance=5.0,
        width=512,
        height=512,
        seed=42,
        output_path="/tmp/out.mp4",
        num_frames=25,
        fps=24,
        output_type="video",
        pipeline_class="LTXImageToVideoPipeline",
        force_cpu_offload=True,
        input_path="/tmp/cond.png",
    )
    p.update(overrides)
    return p


# ── Schema accepts the i2v shape ────────────────────────────────────────


class TestSchemaAcceptsI2vShape:
    """``_validate_payload_shape`` accepts the v0.9.3 i2v shape (video
    output_type + input_path set). Mirrors the t2v schema test in
    test_v090_runner_video_schema.py but exercises the new branch."""

    def test_i2v_payload_with_png_input_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.png")
        assert _validate_payload_shape(payload) == 0

    def test_i2v_payload_with_jpg_input_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.jpg")
        assert _validate_payload_shape(payload) == 0

    def test_i2v_payload_with_jpeg_input_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.jpeg")
        assert _validate_payload_shape(payload) == 0


# ── Schema rejects unsafe i2v input extensions ──────────────────────────


class TestSchemaRejectsUnsafeI2vInputs:
    """For ``output_type=="video"``, ``input_path`` must use one of
    the LTX-VAE-verified extensions {.png, .jpg, .jpeg}. The broader
    image-i2i set (.webp, .heic, .heif) is refused — those formats
    have not been smoke-verified through LTX's VAE encode path."""

    def test_i2v_webp_input_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.webp")
        assert _validate_payload_shape(payload) != 0

    def test_i2v_heic_input_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.heic")
        assert _validate_payload_shape(payload) != 0

    def test_i2v_gif_input_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.gif")
        assert _validate_payload_shape(payload) != 0

    def test_i2v_mp4_input_rejected(self):
        """LTX i2v takes a single still, not a video clip — .mp4
        input is a category error."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond.mp4")
        assert _validate_payload_shape(payload) != 0

    def test_i2v_no_extension_rejected(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(input_path="/tmp/cond")
        assert _validate_payload_shape(payload) != 0


# ── t2v payload unaffected by i2v branch ────────────────────────────────


class TestT2vPayloadUnchanged:
    """Pre-v0.9.3 t2v payloads (no input_path) still pass the schema
    cleanly. Backward compat with v0.9.0 t2v traces."""

    def test_t2v_payload_without_input_path_accepted(self):
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(pipeline_class="LTXPipeline")
        # Remove the i2v field to make it t2v.
        del payload["input_path"]
        assert _validate_payload_shape(payload) == 0

    def test_t2v_payload_with_explicit_null_input_path_accepted(self):
        """Some encoders may emit ``"input_path": null`` for t2v;
        treat as absent."""
        from imgen.engines._diffusers_runner import _validate_payload_shape
        payload = _i2v_video_payload(pipeline_class="LTXPipeline")
        payload["input_path"] = None
        assert _validate_payload_shape(payload) == 0


# ── _run_video threads image= kwarg when input_path set ─────────────────


class TestRunVideoThreadsImageKwarg:
    """When the validated payload contains ``input_path``,
    ``_run_video`` loads the image and threads it into ``pipe(...)``
    as the ``image=`` kwarg. This is the integration point that
    flips LTX from t2v generation to i2v conditioning."""

    def test_run_video_passes_loaded_image_to_pipe(self, tmp_path):
        """Mock the pipeline + load_image, verify the image= kwarg
        propagates from payload through to pipe(**kwargs)."""
        cond_image = tmp_path / "cond.png"
        cond_image.write_bytes(b"fake-png")
        output_path = tmp_path / "out.mp4"

        payload = _i2v_video_payload(
            input_path=str(cond_image),
            output_path=str(output_path),
        )

        fake_image = object()  # sentinel — load_image returns this
        fake_frames = [MagicMock()] * 25
        fake_result = MagicMock()
        fake_result.frames = [fake_frames]
        fake_pipe = MagicMock(return_value=fake_result)
        fake_pipeline_class = MagicMock()
        fake_pipeline_class.from_pretrained.return_value = fake_pipe

        with patch(
            "imgen.engines._diffusers_runner._resolve_pipeline_class",
            return_value=fake_pipeline_class,
        ), patch.dict("sys.modules", {
            "torch": MagicMock(),
            "imageio": MagicMock(),
        }):
            # Patch the diffusers.utils.load_image at the import
            # point inside _run_video (lazy import).
            import sys
            fake_diffusers_utils = MagicMock()
            fake_diffusers_utils.load_image.return_value = fake_image
            fake_diffusers = MagicMock()
            fake_diffusers.utils = fake_diffusers_utils
            sys.modules["diffusers"] = fake_diffusers
            sys.modules["diffusers.utils"] = fake_diffusers_utils
            try:
                from imgen.engines._diffusers_runner import _run_video
                rc = _run_video(payload)
            finally:
                # Clean up the module patches so subsequent tests get
                # the real modules back.
                sys.modules.pop("diffusers", None)
                sys.modules.pop("diffusers.utils", None)

        # The pipe call must have received image=fake_image.
        assert fake_pipe.called
        call_kwargs = fake_pipe.call_args.kwargs
        assert call_kwargs.get("image") is fake_image
        assert call_kwargs["num_frames"] == 25

    def test_run_video_omits_image_kwarg_for_t2v(self, tmp_path):
        """t2v path: input_path absent → no ``image=`` kwarg in the
        pipe call. Backward compat with v0.9.0."""
        output_path = tmp_path / "out.mp4"
        payload = _i2v_video_payload(
            pipeline_class="LTXPipeline",
            output_path=str(output_path),
        )
        del payload["input_path"]

        fake_frames = [MagicMock()] * 25
        fake_result = MagicMock()
        fake_result.frames = [fake_frames]
        fake_pipe = MagicMock(return_value=fake_result)
        fake_pipeline_class = MagicMock()
        fake_pipeline_class.from_pretrained.return_value = fake_pipe

        with patch(
            "imgen.engines._diffusers_runner._resolve_pipeline_class",
            return_value=fake_pipeline_class,
        ), patch.dict("sys.modules", {
            "torch": MagicMock(),
            "imageio": MagicMock(),
        }):
            from imgen.engines._diffusers_runner import _run_video
            _run_video(payload)

        assert fake_pipe.called
        assert "image" not in fake_pipe.call_args.kwargs


# ── Engine threads image_path into payload ──────────────────────────────


class TestEngineThreadsImagePathIntoPayload:
    """When the Model is a video Model AND ``params.input_path`` is
    non-None, the engine writes ``input_path`` into the payload. The
    ``pipeline_class`` is read from ``model.video.pipeline_class``
    per C2 — cmd_video (C5) constructs an i2v-flavoured Model via
    ``dataclasses.replace`` before reaching the engine."""

    def test_engine_writes_input_path_for_i2v_video(self, tmp_path, monkeypatch):
        import json
        from dataclasses import replace
        from imgen.engines.diffusers_mps_engine import DiffusersMpsEngine
        from imgen.engines.base import GenParams
        from imgen.models import BUILTIN_MODELS

        # Build an i2v-flavoured LTX Model: VideoConfig pipeline_class
        # flipped to LTXImageToVideoPipeline.
        ltx = BUILTIN_MODELS["ltx-video"]
        i2v_video_cfg = replace(ltx.video, pipeline_class="LTXImageToVideoPipeline")
        model = replace(ltx, video=i2v_video_cfg)

        captured: dict = {}

        def fake_run(argv, env, stdin_data, log_file):
            captured["payload"] = json.loads(stdin_data.decode("utf-8"))
            return 0

        monkeypatch.setattr(
            "imgen.subprocess_helpers.run_with_stderr_redaction", fake_run,
        )
        monkeypatch.setattr("imgen.paths.IMGEN_INSTALL_ROOT", tmp_path)
        (tmp_path / ".venv-diffusers" / "bin").mkdir(parents=True)
        (tmp_path / ".venv-diffusers" / "bin" / "python").write_text("")
        (tmp_path / ".venv-diffusers" / "bin" / "python").chmod(0o755)

        cond_path = tmp_path / "still.png"
        cond_path.write_bytes(b"fake-png")

        params = GenParams(
            prompt="wind blows", negative="static, still, frozen, no motion",
            width=512, height=512, steps=50, guidance=5.0,
            seed=42, quantize=0, strength=0.0,
            input_path=cond_path,
            output_path=tmp_path / "out.mp4",
            loras=(), mlx_cache_gb=12, battery_stop=20,
            num_frames=25, fps=24,
        )

        engine = DiffusersMpsEngine()
        engine.run(model=model, params=params)

        payload = captured["payload"]
        assert payload["pipeline_class"] == "LTXImageToVideoPipeline"
        assert payload["input_path"] == str(cond_path)
        assert payload["output_type"] == "video"

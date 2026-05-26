"""v0.9 commit 4 — DiffusersMpsEngine.run video payload construction.

Per [[project-v090-design]] §F. Engine.run branches by
``model.video`` when building the JSON payload that crosses the
stdin trust boundary into the runner. Image Models keep the v0.8
payload shape (backwards-compat); video Models add the v0.9 keys
(output_type, num_frames, fps, pipeline_class, force_cpu_offload).

The pipeline_class is HARDCODED to "LTXPipeline" at v0.9.0 because
LTX-Video is the only built-in video Model in this ship. v0.9.x
will add a Model-level field when the second video Model lands —
captured as a backlog note in commit body.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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
        video=_ltx_video_config(),
    )


def _image_model():
    from imgen.models import Model
    return Model(
        engine="diffusers_mps",
        repo="Qwen/Qwen-Image-2512",
        ram_baseline_gb=24.0,
        ram_slope_gb_per_mp=8.0,
    )


def _video_genparams(**overrides):
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="a samurai walking",
        negative="",
        width=768,
        height=512,
        steps=25,
        guidance=3.0,
        seed=42,
        quantize=8,
        strength=0.0,
        input_path=None,
        output_path=Path("/tmp/out.mp4"),
        loras=(),
        num_frames=25,
        fps=24,
    )
    defaults.update(overrides)
    return GenParams(**defaults)


def _image_genparams(**overrides):
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="a samurai",
        negative="",
        width=1024,
        height=1024,
        steps=50,
        guidance=4.0,
        seed=42,
        quantize=4,
        strength=0.0,
        input_path=None,
        output_path=Path("/tmp/out.png"),
        loras=(),
    )
    defaults.update(overrides)
    return GenParams(**defaults)


def _setup_fake_venv_and_capture(monkeypatch, tmp_path):
    from imgen import paths, subprocess_helpers

    install_root = tmp_path / "root"
    venv_python = install_root / ".venv-diffusers" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python3\n")
    venv_python.chmod(0o755)
    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)

    captured = {}

    def fake_run(*args, **kwargs):
        if args:
            captured["argv"] = args[0]
        else:
            captured["argv"] = kwargs.get("argv")
        captured["stdin_data"] = kwargs.get("stdin_data")
        return 0

    monkeypatch.setattr(
        subprocess_helpers, "run_with_stderr_redaction", fake_run,
    )
    return captured


class TestEngineRunVideoPayload:
    """Lock-in: video Models construct the v0.9 payload with all
    required keys; image Models keep the v0.8 shape."""

    def test_video_payload_carries_output_type_video(
        self, monkeypatch, tmp_path,
    ):
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_ltx_model(), _video_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert payload["output_type"] == "video"

    def test_video_payload_carries_num_frames(self, monkeypatch, tmp_path):
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(
            _ltx_model(), _video_genparams(num_frames=49),
        )
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert payload["num_frames"] == 49

    def test_video_payload_carries_fps(self, monkeypatch, tmp_path):
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(
            _ltx_model(), _video_genparams(fps=30),
        )
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert payload["fps"] == 30

    def test_video_payload_pipeline_class_ltx(self, monkeypatch, tmp_path):
        """v0.9.0: LTX is the only built-in video Model — pipeline_class
        is hardcoded to "LTXPipeline" until v0.9.x adds a Model field."""
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_ltx_model(), _video_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert payload["pipeline_class"] == "LTXPipeline"

    def test_video_payload_force_cpu_offload_from_video_config(
        self, monkeypatch, tmp_path,
    ):
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_ltx_model(), _video_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert payload["force_cpu_offload"] is True

    def test_video_payload_no_input_path(self, monkeypatch, tmp_path):
        """v0.9.0 LTX is t2v only — no image conditioning input."""
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_ltx_model(), _video_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        # input_path may be present-but-null OR absent — both fine
        assert payload.get("input_path") is None


class TestEngineRunImagePayloadUnchanged:
    """Backwards-compat: image Models produce the v0.8 payload — no
    output_type, no v0.9 video keys. A drift here would have broken
    the existing diffusers_mps test surface."""

    def test_image_payload_no_output_type(self, monkeypatch, tmp_path):
        """v0.8 image payloads omit output_type entirely — runner
        defaults to image when key is absent."""
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_image_model(), _image_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert "output_type" not in payload, (
            "image Model payload must NOT carry output_type "
            "(v0.8 backwards-compat)"
        )

    def test_image_payload_no_video_keys(self, monkeypatch, tmp_path):
        from imgen.engines import DiffusersMpsEngine
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_image_model(), _image_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        for k in ("num_frames", "fps", "pipeline_class",
                  "force_cpu_offload"):
            assert k not in payload, (
                f"image Model payload contains video-only key {k!r}; "
                "Engine.run drift detected"
            )


class TestPayloadPassesRunnerSchema:
    """End-to-end lock-in: the payload Engine.run produces MUST pass
    the runner's _validate_payload_shape. Catches any drift between
    Engine.run's payload construction and the schema's required-key
    set."""

    def test_video_payload_passes_validate_payload_shape(
        self, monkeypatch, tmp_path,
    ):
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_ltx_model(), _video_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert _validate_payload_shape(payload) == 0, (
            f"Engine.run video payload failed runner schema; payload: {payload}"
        )

    def test_image_payload_passes_validate_payload_shape(
        self, monkeypatch, tmp_path,
    ):
        from imgen.engines import DiffusersMpsEngine
        from imgen.engines._diffusers_runner import _validate_payload_shape
        captured = _setup_fake_venv_and_capture(monkeypatch, tmp_path)
        DiffusersMpsEngine().run(_image_model(), _image_genparams())
        payload = json.loads(captured["stdin_data"].decode("utf-8"))
        assert _validate_payload_shape(payload) == 0, (
            f"Engine.run image payload failed runner schema; payload: {payload}"
        )

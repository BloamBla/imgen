"""v0.9 commit 4 — _diffusers_runner main() dispatch + _run_video.

Per [[project-v090-design]] §F. Tests cover:

* ``main()`` dispatches to ``_run_video`` when ``output_type=="video"``
  and to ``_run_image`` (the v0.8 path, now refactored) otherwise.
* ``_resolve_pipeline_class`` fail-closed defence-in-depth: even with
  the schema allowlist bypassed, an unknown class name raises.
* Atomic-rename MP4 write: ``tempfile.NamedTemporaryFile`` →
  ``os.rename``. Race window between filename-check and write closes
  at the atomic rename op.
* ``output_dir.is_symlink()`` guard rejects parent-traversal symlink
  attacks BEFORE any pipeline run.

These tests heavily mock the diffusers pipeline because the actual
LTX integration is exercised by the alpha-smoke step after this
commit lands (per [[project-next-session-pickup-2026-05-26-v090-design-locked]]
operational notes). The mock surface stays tight enough that drift
in the production code path is still surfaced — every assertion
pins one specific runtime behaviour.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _valid_video_payload(**overrides):
    p = dict(
        repo="Lightricks/LTX-Video",
        prompt="a samurai",
        negative="",
        steps=25,
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


# ── _resolve_pipeline_class — fail-closed allowlist ───────────────────


class TestResolvePipelineClassFailClosed:
    """Defence-in-depth: even if a future refactor bypassed
    _validate_payload_shape's allowlist check, _resolve_pipeline_class
    must still raise on unknown names. Schema + resolver together
    form the two-layer security wall around getattr-style introspection."""

    def test_unknown_class_raises(self):
        from imgen.engines._diffusers_runner import _resolve_pipeline_class
        with pytest.raises((KeyError, ValueError)):
            _resolve_pipeline_class("StableDiffusionPipeline")

    def test_dunder_class_raises(self):
        from imgen.engines._diffusers_runner import _resolve_pipeline_class
        with pytest.raises((KeyError, ValueError)):
            _resolve_pipeline_class("__class__")

    def test_path_traversal_raises(self):
        from imgen.engines._diffusers_runner import _resolve_pipeline_class
        with pytest.raises((KeyError, ValueError)):
            _resolve_pipeline_class("../../etc/passwd")

    def test_empty_string_raises(self):
        from imgen.engines._diffusers_runner import _resolve_pipeline_class
        with pytest.raises((KeyError, ValueError)):
            _resolve_pipeline_class("")


# ── main() dispatch by output_type ────────────────────────────────────


class TestMainDispatchByOutputType:
    """main() reads payload.output_type and routes to _run_video or
    _run_image. Image is the v0.8-compatible default when the key
    is absent."""

    def _stdin_payload(self, payload_dict):
        import io
        import json
        return io.BytesIO(json.dumps(payload_dict).encode("utf-8"))

    def test_main_dispatches_to_run_video_when_output_type_video(self, monkeypatch):
        from imgen.engines import _diffusers_runner

        captured = []

        def fake_run_video(payload):
            captured.append(("video", payload))
            return 0

        def fake_run_image(payload):
            captured.append(("image", payload))
            return 0

        monkeypatch.setattr(_diffusers_runner, "_run_video", fake_run_video)
        monkeypatch.setattr(_diffusers_runner, "_run_image", fake_run_image)

        # Plant payload on stdin
        payload = _valid_video_payload()
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(buffer=self._stdin_payload(payload)),
        )

        rc = _diffusers_runner.main()
        assert rc == 0
        assert len(captured) == 1
        assert captured[0][0] == "video", (
            f"output_type=='video' should dispatch to _run_video, "
            f"got: {captured[0][0]}"
        )

    def test_main_dispatches_to_run_image_when_output_type_image(self, monkeypatch):
        from imgen.engines import _diffusers_runner

        captured = []

        def fake_run_image(payload):
            captured.append(("image", payload))
            return 0

        def fake_run_video(payload):
            captured.append(("video", payload))
            return 0

        monkeypatch.setattr(_diffusers_runner, "_run_image", fake_run_image)
        monkeypatch.setattr(_diffusers_runner, "_run_video", fake_run_video)

        payload = dict(
            repo="Qwen/Qwen-Image-2512",
            prompt="x", negative="", steps=20, guidance=3.5,
            width=512, height=512, seed=42,
            output_path="/tmp/x.png",
            output_type="image",
        )
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(buffer=self._stdin_payload(payload)),
        )

        rc = _diffusers_runner.main()
        assert rc == 0
        assert captured[0][0] == "image"

    def test_main_dispatches_to_run_image_when_output_type_absent_v08_compat(
        self, monkeypatch,
    ):
        """v0.8 image payloads omit output_type entirely — must still
        route to _run_image (the existing v0.8 behaviour)."""
        from imgen.engines import _diffusers_runner

        captured = []

        def fake_run_image(payload):
            captured.append("image")
            return 0

        def fake_run_video(payload):
            captured.append("video")
            return 0

        monkeypatch.setattr(_diffusers_runner, "_run_image", fake_run_image)
        monkeypatch.setattr(_diffusers_runner, "_run_video", fake_run_video)

        payload = dict(
            repo="Qwen/Qwen-Image-2512",
            prompt="x", negative="", steps=20, guidance=3.5,
            width=512, height=512, seed=42,
            output_path="/tmp/x.png",
            # NO output_type — v0.8 payload shape
        )
        monkeypatch.setattr(
            "sys.stdin",
            MagicMock(buffer=self._stdin_payload(payload)),
        )

        rc = _diffusers_runner.main()
        assert rc == 0
        assert captured == ["image"], (
            f"v0.8 payload (no output_type) must dispatch to image; got: {captured}"
        )


# ── _run_video — atomic-rename pattern ────────────────────────────────


class TestRunVideoAtomicRename:
    """§F: ``imageio.mimsave`` doesn't accept fd-mode for libx264
    muxing (moov atom requires seekable container). The atomic-rename
    pattern is the workaround: write to a NamedTemporaryFile in the
    same directory, then os.rename to the user-controlled path. The
    rename is atomic on same-fs and replaces any pre-existing
    symlink with a regular file."""

    def _setup_mock_pipeline(self, monkeypatch):
        """Build a fake pipeline class that produces a 25-frame video
        on call. Returns the patch target so the test can inject more
        granular assertions."""
        from imgen.engines import _diffusers_runner

        # Fake PIL-like frame — just enough surface for the
        # imageio.mimsave call. The actual mimsave is also mocked
        # so the frame contents don't matter.
        fake_frame = MagicMock(name="fake_pil_frame")
        fake_frames = [fake_frame] * 25
        fake_result = MagicMock(frames=[fake_frames])

        fake_pipe_instance = MagicMock(name="fake_pipe_instance")
        fake_pipe_instance.return_value = fake_result

        fake_pipe_class = MagicMock(name="fake_pipe_class")
        fake_pipe_class.from_pretrained.return_value = fake_pipe_instance

        monkeypatch.setattr(
            _diffusers_runner, "_resolve_pipeline_class",
            lambda name: fake_pipe_class,
        )

        # Stub torch — _run_video uses torch.bfloat16 + torch.Generator.
        fake_torch = MagicMock(name="torch")
        fake_torch.bfloat16 = "bfloat16-sentinel"
        fake_gen = MagicMock()
        fake_gen.manual_seed.return_value = fake_gen
        fake_torch.Generator.return_value = fake_gen

        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        return fake_pipe_instance

    def test_run_video_writes_via_tempfile_then_atomic_rename(
        self, tmp_path, monkeypatch,
    ):
        """Lock-in: _run_video MUST use tempfile.NamedTemporaryFile +
        os.rename, NOT direct write at output_path. Capture os.rename
        calls to verify."""
        from imgen.engines import _diffusers_runner

        self._setup_mock_pipeline(monkeypatch)

        # Capture os.rename invocations
        rename_calls = []
        real_rename = os.rename

        def capturing_rename(src, dst):
            rename_calls.append((str(src), str(dst)))
            real_rename(src, dst)

        monkeypatch.setattr(os, "rename", capturing_rename)

        # Stub imageio so we don't need it installed in main venv —
        # write a tiny file at the temp path it's called with.
        fake_imageio = MagicMock()

        def fake_mimsave(path, frames, **kwargs):
            # imitate libx264 muxing — just write some bytes
            Path(path).write_bytes(b"FAKE_MP4_BYTES")

        fake_imageio.mimsave.side_effect = fake_mimsave
        monkeypatch.setitem(
            __import__("sys").modules, "imageio", fake_imageio,
        )

        output_path = tmp_path / "out.mp4"
        payload = _valid_video_payload(output_path=str(output_path))

        rc = _diffusers_runner._run_video(payload)
        assert rc == 0
        assert output_path.exists(), "output file missing after _run_video"
        assert output_path.read_bytes() == b"FAKE_MP4_BYTES"

        # Atomic-rename pattern: rename was called with (tmp_path, output_path).
        assert len(rename_calls) == 1, (
            f"expected single os.rename call (atomic-rename pattern); "
            f"got {len(rename_calls)}: {rename_calls}"
        )
        src, dst = rename_calls[0]
        assert dst == str(output_path)
        assert src != str(output_path), (
            "src and dst must differ — atomic rename requires temp file"
        )
        # Temp file lived in the SAME directory (same-fs requirement
        # for atomic os.rename)
        assert Path(src).parent == output_path.parent

    def test_run_video_refuses_symlink_at_output_dir(
        self, tmp_path, monkeypatch,
    ):
        """Parent-traversal symlink attack: ``--output /tmp/foo.mp4``
        where ``/tmp`` is replaced with a symlink to ``/etc``. Runner
        must refuse to write BEFORE pipeline run."""
        from imgen.engines import _diffusers_runner

        self._setup_mock_pipeline(monkeypatch)

        # Create symlinked output_dir
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        symlinked_dir = tmp_path / "linked"
        symlinked_dir.symlink_to(real_dir)
        output_path = symlinked_dir / "out.mp4"

        # imageio stub still in case it's reached
        fake_imageio = MagicMock()
        monkeypatch.setitem(
            __import__("sys").modules, "imageio", fake_imageio,
        )

        payload = _valid_video_payload(output_path=str(output_path))
        rc = _diffusers_runner._run_video(payload)
        assert rc != 0, (
            "symlinked output_dir must be refused"
        )
        # Pipeline should NOT have been called (refuse fires BEFORE
        # the inference work).
        fake_imageio.mimsave.assert_not_called()

    def test_run_video_cleans_up_tempfile_on_mimsave_failure(
        self, tmp_path, monkeypatch,
    ):
        """If mimsave raises (codec error, ffmpeg crash, etc.) the
        tempfile in output_dir must not linger — leak-free error
        path. NOTE: re-raises the original exception; caller (main)
        catches it and emits a static stderr line."""
        from imgen.engines import _diffusers_runner

        self._setup_mock_pipeline(monkeypatch)

        fake_imageio = MagicMock()
        fake_imageio.mimsave.side_effect = RuntimeError("ffmpeg crashed")
        monkeypatch.setitem(
            __import__("sys").modules, "imageio", fake_imageio,
        )

        output_path = tmp_path / "out.mp4"
        payload = _valid_video_payload(output_path=str(output_path))

        # _run_video may re-raise OR return non-zero — either is
        # acceptable defensive behaviour; the lock-in here is that
        # no temp file lingers in output_dir afterward.
        try:
            _diffusers_runner._run_video(payload)
        except Exception:
            pass

        # Output_dir contains no .imgen-video-* tempfiles
        leftover = list(tmp_path.glob(".imgen-video-*"))
        assert leftover == [], (
            f"tempfile not cleaned up on mimsave failure: {leftover}"
        )
        assert not output_path.exists(), (
            "output_path must NOT exist if mimsave failed"
        )

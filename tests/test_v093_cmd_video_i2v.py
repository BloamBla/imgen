"""v0.9.3 C5 — parser + cmd_video + build_video_iteration wiring for i2v.

Per [[project-v093-i2v]] C5. Covers:

* ``--image PATH`` flag exists on the ``imgen video`` parser stanza.
  Default ``None`` keeps v0.9.0 t2v behaviour byte-identically.
* ``cmd_video`` validates the path via
  :func:`imgen._i2v_resolve.validate_image_path_or_die` (dies code=2
  on missing / unsupported ext / symlink-with-traversal-target).
* ``cmd_video`` resolves i2v motion defaults via
  :func:`imgen._i2v_resolve.resolve_i2v_motion_defaults` and bakes the
  effective guidance + negative_prompt onto ``args`` before reaching
  the orchestrator. User-supplied flags take precedence over the i2v
  defaults.
* ``build_video_iteration`` reads ``args.image`` (set by cmd_video)
  and flips the resolved Model's ``VideoConfig.pipeline_class`` from
  ``"LTXPipeline"`` to ``"LTXImageToVideoPipeline"`` via
  ``dataclasses.replace``. This keeps the Engine policy-free per the
  B-1 architecture (Engine just reads ``model.video.pipeline_class``;
  build_video_iteration is the video-domain layer that knows the
  flip).
* Confirm-gate display shows "conditioned on: PATH" when i2v is
  active.

Mocks at the orchestrator + engine.run boundary; no real LTX
subprocess.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imgen.defaults import DEFAULTS


def _make_video_args(tmp_path, **overrides):
    """SimpleNamespace shaped like parsed `imgen video` args."""
    defaults = dict(
        prompt="a samurai walks",
        prompt_file=None,
        output=None,
        output_dir=None,
        duration=None,
        num_frames=None,
        fps=None,
        steps=None,
        guidance=None,
        negative_prompt=None,
        seed=42,
        model="ltx-video",
        quantize=None,
        width=512,
        height=512,
        preview=False,
        no_open=True,
        yes=True,
        dry_run=True,
        force=True,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        lora=None,
        no_lora=False,
        strength=None,
        scope=None,
        style=None,
        custom_prompt=None,
        image=None,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        imgen_config_enhance={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Parser: --image flag exists ─────────────────────────────────────────


class TestParserImageFlag:
    """The v0.9.3 ``--image PATH`` flag lands on the video parser. The
    default value MUST be None so v0.9.0 t2v invocations (without the
    flag) keep their existing args namespace shape."""

    def _parse(self, argv):
        from imgen.parser import build_parser
        parser = build_parser()
        return parser.parse_args(argv)

    def test_video_parser_accepts_image_flag(self):
        args = self._parse([
            "video", "wind blows", "--image", "/tmp/still.png",
        ])
        assert getattr(args, "image", None) == Path("/tmp/still.png")

    def test_video_parser_image_default_is_none(self):
        """No --image flag → args.image is None (v0.9.0 t2v compat)."""
        args = self._parse(["video", "a samurai"])
        assert getattr(args, "image", None) is None

    def test_video_parser_image_accepts_relative_path(self):
        """Argparse accepts relative paths verbatim; resolution +
        existence checks happen at cmd_video boundary via
        validate_image_path_or_die."""
        args = self._parse(["video", "x", "--image", "still.png"])
        assert getattr(args, "image", None) == Path("still.png")


# ── build_video_iteration flips pipeline_class when args.image set ──────


class TestBuildVideoIterationFlipsPipelineClass:
    """When ``args.image`` is set on the namespace,
    ``build_video_iteration`` constructs an i2v-flavoured Model by
    flipping ``VideoConfig.pipeline_class`` from ``"LTXPipeline"`` to
    ``"LTXImageToVideoPipeline"``. The flip happens via
    ``dataclasses.replace`` — keeps BUILTIN_MODELS pristine."""

    def _ltx_backend(self):
        from imgen.backends import BUILTIN_BACKENDS
        return BUILTIN_BACKENDS["ltx-video"]

    def test_t2v_default_pipeline_class_preserved(self, tmp_path):
        """Backward compat: no args.image → Iteration carries
        ``model.video.pipeline_class == "LTXPipeline"``."""
        from imgen.build_iteration import build_video_iteration
        args = _make_video_args(tmp_path)
        iters = build_video_iteration(
            args=args, prompt="x",
            merged_defaults=dict(DEFAULTS), be=self._ltx_backend(),
            width=512, height=512,
            explicit_output=tmp_path / "out.mp4",
            run_dir=None, base_seed=42,
        )
        assert iters[0].model.video.pipeline_class == "LTXPipeline"

    def test_i2v_flips_pipeline_class_via_args_image(self, tmp_path):
        """args.image set → Iteration carries the i2v pipeline_class."""
        from imgen.build_iteration import build_video_iteration

        cond = tmp_path / "cond.png"
        cond.write_bytes(b"fake-png")
        args = _make_video_args(tmp_path, image=cond)

        iters = build_video_iteration(
            args=args, prompt="wind blows",
            merged_defaults=dict(DEFAULTS), be=self._ltx_backend(),
            width=512, height=512,
            explicit_output=tmp_path / "out.mp4",
            run_dir=None, base_seed=42,
        )
        it = iters[0]
        assert it.model.video.pipeline_class == "LTXImageToVideoPipeline"
        assert it.params.input_path == cond

    def test_i2v_args_image_takes_precedence_over_kwarg(self, tmp_path):
        """When both ``args.image`` AND ``image_path=`` kwarg are
        present, args.image wins (the production path through
        cmd_video always sets args.image)."""
        from imgen.build_iteration import build_video_iteration

        cond_args = tmp_path / "via_args.png"
        cond_args.write_bytes(b"x")
        cond_kw = tmp_path / "via_kwarg.png"
        cond_kw.write_bytes(b"x")
        args = _make_video_args(tmp_path, image=cond_args)

        iters = build_video_iteration(
            args=args, prompt="x",
            merged_defaults=dict(DEFAULTS), be=self._ltx_backend(),
            width=512, height=512,
            explicit_output=tmp_path / "out.mp4",
            run_dir=None, base_seed=42,
            image_path=cond_kw,
        )
        assert iters[0].params.input_path == cond_args


# ── cmd_video validates + resolves motion defaults ──────────────────────


class TestCmdVideoValidatesAndResolves:
    """cmd_video boundary: validate path + bake motion defaults
    onto args before reaching the orchestrator."""

    def test_cmd_video_dies_on_nonexistent_image(self, tmp_path, capsys):
        """Missing --image PATH → die(code=2) BEFORE any orchestrator
        work (no expensive imports, no subprocess spawn)."""
        from imgen.commands.video import cmd_video
        args = _make_video_args(
            tmp_path, image=tmp_path / "does-not-exist.png",
        )
        with pytest.raises(SystemExit) as exc:
            cmd_video(args)
        assert exc.value.code == 2

    def test_cmd_video_dies_on_unsupported_image_extension(
        self, tmp_path, capsys,
    ):
        """`.webp` / `.gif` / `.heic` rejected by the i2v allowlist
        (parent-side, before runner trust boundary)."""
        from imgen.commands.video import cmd_video
        bad = tmp_path / "x.webp"
        bad.write_bytes(b"webp")
        args = _make_video_args(tmp_path, image=bad)
        with pytest.raises(SystemExit) as exc:
            cmd_video(args)
        assert exc.value.code == 2

    def test_cmd_video_resolves_image_to_absolute_path_on_args(
        self, tmp_path, monkeypatch,
    ):
        """cmd_video mutates args.image to the resolved absolute Path
        so downstream consumers (build_video_iteration, history) all
        see the same canonical form."""
        from imgen.commands.video import cmd_video

        cond = tmp_path / "still.png"
        cond.write_bytes(b"fake-png")
        args = _make_video_args(tmp_path, image=cond)

        captured: dict = {}

        def fake_orchestrate(received_args, **kwargs):
            captured["image_after"] = received_args.image
            return 0

        monkeypatch.setattr(
            "imgen.commands.video._orchestrate_t2x", fake_orchestrate,
        )
        cmd_video(args)
        assert captured["image_after"] == cond.resolve()
        assert captured["image_after"].is_absolute()

    def test_cmd_video_bakes_i2v_guidance_when_user_did_not_set(
        self, tmp_path, monkeypatch,
    ):
        """User didn't pass --guidance → args.guidance becomes 5.0
        (i2v default). cmd_video resolves the i2v defaults at the
        boundary so the rest of the pipeline reads them via the
        normal args.guidance / args.negative_prompt paths."""
        from imgen.commands.video import cmd_video

        cond = tmp_path / "still.png"
        cond.write_bytes(b"x")
        args = _make_video_args(tmp_path, image=cond)
        # args.guidance starts at None (no CLI override)
        assert args.guidance is None

        captured: dict = {}

        def fake_orchestrate(received_args, **kwargs):
            captured["guidance"] = received_args.guidance
            captured["negative"] = received_args.negative_prompt
            return 0

        monkeypatch.setattr(
            "imgen.commands.video._orchestrate_t2x", fake_orchestrate,
        )
        cmd_video(args)
        assert captured["guidance"] == 5.0
        assert captured["negative"] == "static, still, frozen, no motion"

    def test_cmd_video_user_guidance_override_wins(
        self, tmp_path, monkeypatch,
    ):
        """Explicit --guidance 7 survives the i2v default-resolve."""
        from imgen.commands.video import cmd_video

        cond = tmp_path / "still.png"
        cond.write_bytes(b"x")
        args = _make_video_args(tmp_path, image=cond, guidance=7.0)

        captured: dict = {}

        def fake_orchestrate(received_args, **kwargs):
            captured["guidance"] = received_args.guidance
            return 0

        monkeypatch.setattr(
            "imgen.commands.video._orchestrate_t2x", fake_orchestrate,
        )
        cmd_video(args)
        assert captured["guidance"] == 7.0

    def test_cmd_video_t2v_path_unchanged_when_no_image(
        self, tmp_path, monkeypatch,
    ):
        """v0.9.0 t2v: no --image → args.guidance / args.negative_prompt
        stay at None (caller's t2v default-resolve handles them)."""
        from imgen.commands.video import cmd_video

        args = _make_video_args(tmp_path)
        captured: dict = {}

        def fake_orchestrate(received_args, **kwargs):
            captured["guidance"] = received_args.guidance
            captured["negative"] = received_args.negative_prompt
            captured["image"] = received_args.image
            return 0

        monkeypatch.setattr(
            "imgen.commands.video._orchestrate_t2x", fake_orchestrate,
        )
        cmd_video(args)
        assert captured["guidance"] is None
        assert captured["negative"] is None
        assert captured["image"] is None

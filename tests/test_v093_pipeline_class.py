"""v0.9.3 C2 — VideoConfig.pipeline_class typed field, closes B-1.

Pre-v0.9.3 the runner-payload key ``pipeline_class`` was hardcoded in
two places:

* ``diffusers_mps_engine.py:203`` set ``payload["pipeline_class"] =
  "LTXPipeline"`` literally for every video run.
* ``engine_dispatch.py:214`` rendered ``pipeline_class:  LTXPipeline``
  literally in the dry-run output.

This worked for v0.9.0 because LTX-Video was the only video Model and
only t2v needed. v0.9.3 adds i2v which uses
``LTXImageToVideoPipeline`` on the SAME checkpoint, so the engine must
choose between two pipeline classes at runtime. B-1 backlog (originally
worded as "2nd video Model") closes here even though the actual trigger
is "2nd pipeline class on the same Model" — the underlying anti-pattern
is the same.

Tests in this file lock the v0.9.3 contract:

* :class:`VideoConfig` gains a ``pipeline_class`` field with default
  ``"LTXPipeline"`` (v0.9.0 backward compat — every existing video
  Model construction without the new kwarg keeps working).
* The new field is validated against an :data:`_VIDEO_PIPELINE_CLASS_
  ALLOWLIST` literal frozenset at ``__post_init__``. Anything outside
  raises ``ValueError`` — same fail-closed pattern as the runner-side
  allowlist on the trust boundary, but enforced one layer earlier.
* The runner's ``_PIPELINE_CLASS_ALLOWLIST`` is expanded to include
  ``"LTXImageToVideoPipeline"`` (the v0.9.3 i2v class). The two
  allowlists must be consistent — VideoConfig's set must be a subset
  of the runner's set (parent-side rejection can be stricter, never
  looser, than runner-side).
* ``DiffusersMpsEngine.run`` reads ``model.video.pipeline_class``
  instead of hardcoding ``"LTXPipeline"``.
* The dry-run formatter in ``engine_dispatch._format_diffusers_video
  _dryrun`` reads from the VideoConfig too — dry-run truth must match
  payload truth.
"""
from __future__ import annotations

import pytest


# ── VideoConfig.pipeline_class field shape ──────────────────────────────


class TestVideoConfigPipelineClassField:
    """The new field exists, defaults to LTXPipeline (v0.9.0 compat),
    and is validated against the video-pipeline allowlist."""

    def test_videoconfig_default_pipeline_class_is_ltxpipeline(self):
        """v0.9.0 backward compat — existing LTX-Video VideoConfig
        constructions without the new kwarg keep producing the
        v0.9.0 t2v pipeline class."""
        from imgen.models import VideoConfig
        vc = VideoConfig(
            default_num_frames=25, default_fps=24, max_num_frames=257,
        )
        assert vc.pipeline_class == "LTXPipeline"

    def test_videoconfig_accepts_ltx_image_to_video(self):
        """The v0.9.3 i2v pipeline class is a valid VideoConfig value
        for the LTX-Video Model row (same checkpoint, different
        pipeline)."""
        from imgen.models import VideoConfig
        vc = VideoConfig(
            default_num_frames=25, default_fps=24, max_num_frames=257,
            pipeline_class="LTXImageToVideoPipeline",
        )
        assert vc.pipeline_class == "LTXImageToVideoPipeline"

    def test_videoconfig_rejects_unknown_pipeline_class(self):
        """Anything outside the video-pipeline allowlist raises at
        construction. Fail-closed prevents user TOMLs from declaring
        ``pipeline_class = "StableDiffusionPipeline"`` and having the
        error surface only at runner trust-boundary (which would still
        catch it, but later)."""
        from imgen.models import VideoConfig
        with pytest.raises(ValueError, match="pipeline_class"):
            VideoConfig(
                default_num_frames=25, default_fps=24, max_num_frames=257,
                pipeline_class="StableDiffusionPipeline",
            )

    def test_videoconfig_rejects_empty_pipeline_class(self):
        """Empty string is not a legitimate pipeline class — would
        bypass introspection-style anti-patterns less helpfully than
        the explicit allowlist."""
        from imgen.models import VideoConfig
        with pytest.raises(ValueError, match="pipeline_class"):
            VideoConfig(
                default_num_frames=25, default_fps=24, max_num_frames=257,
                pipeline_class="",
            )

    def test_videoconfig_rejects_image_pipeline_class(self):
        """The image-fallback class ``DiffusionPipeline`` lives in the
        RUNNER's full allowlist (it's needed for the v0.8 image path)
        but NOT in the VIDEO sub-allowlist. A VideoConfig that
        declared the image class would mis-route inside the runner;
        reject at construction."""
        from imgen.models import VideoConfig
        with pytest.raises(ValueError, match="pipeline_class"):
            VideoConfig(
                default_num_frames=25, default_fps=24, max_num_frames=257,
                pipeline_class="DiffusionPipeline",
            )


# ── Allowlist subset relationship ───────────────────────────────────────


class TestAllowlistSubsetInvariant:
    """The parent-side VideoConfig allowlist must be a subset of the
    runner-side _PIPELINE_CLASS_ALLOWLIST. Parent can be stricter
    (e.g. reject image classes on video Models), never looser (any
    string accepted at the parent must reach a runner class object)."""

    def test_video_allowlist_is_subset_of_runner_allowlist(self):
        from imgen.models import _VIDEO_PIPELINE_CLASS_ALLOWLIST
        from imgen.engines._diffusers_runner import _PIPELINE_CLASS_ALLOWLIST
        assert _VIDEO_PIPELINE_CLASS_ALLOWLIST <= _PIPELINE_CLASS_ALLOWLIST, (
            f"video allowlist not subset of runner: "
            f"{_VIDEO_PIPELINE_CLASS_ALLOWLIST - _PIPELINE_CLASS_ALLOWLIST}"
        )

    def test_runner_allowlist_contains_ltx_image_to_video(self):
        """v0.9.3 requirement: i2v pipeline must be runner-reachable."""
        from imgen.engines._diffusers_runner import _PIPELINE_CLASS_ALLOWLIST
        assert "LTXImageToVideoPipeline" in _PIPELINE_CLASS_ALLOWLIST

    def test_runner_resolve_returns_ltx_image_to_video_class(self):
        """``_resolve_pipeline_class("LTXImageToVideoPipeline")`` must
        successfully resolve to the diffusers class object (not raise).
        This test imports diffusers — skipped if the .venv-diffusers
        deps aren't installed in the test environment."""
        try:
            from imgen.engines._diffusers_runner import _resolve_pipeline_class
            cls = _resolve_pipeline_class("LTXImageToVideoPipeline")
        except ImportError:
            pytest.skip("diffusers not importable in this environment")
        assert cls is not None
        assert cls.__name__ == "LTXImageToVideoPipeline"


# ── Built-in LTX-Video row keeps backward compat ────────────────────────


class TestBuiltinLtxVideoRowUnchanged:
    """The v0.9.0 LTX-Video Model row continues to declare the v0.9.0
    t2v pipeline as its default — no behavioural change to existing
    callers. i2v is opt-in via the explicit kwarg path."""

    def test_builtin_ltx_video_pipeline_class_default(self):
        from imgen.models import BUILTIN_MODELS
        ltx = BUILTIN_MODELS["ltx-video"]
        assert ltx.video is not None
        assert ltx.video.pipeline_class == "LTXPipeline"


# ── Engine.run threads pipeline_class from VideoConfig ──────────────────


class TestEngineThreadsPipelineClassFromVideoConfig:
    """``DiffusersMpsEngine.run`` must read ``model.video.pipeline_class``
    instead of hardcoding the string. After the refactor, an LTX-Video
    Model variant with ``pipeline_class="LTXImageToVideoPipeline"``
    threads that exact string into the runner payload."""

    def _make_i2v_model(self):
        """LTX-Video Model row mutated to declare i2v pipeline class."""
        from dataclasses import replace
        from imgen.models import BUILTIN_MODELS
        ltx = BUILTIN_MODELS["ltx-video"]
        assert ltx.video is not None
        i2v_video_cfg = replace(ltx.video, pipeline_class="LTXImageToVideoPipeline")
        return replace(ltx, video=i2v_video_cfg)

    def test_engine_threads_i2v_pipeline_class_into_payload(
        self, tmp_path, monkeypatch,
    ):
        """Capture the JSON payload that the engine hands to the
        subprocess and assert ``pipeline_class`` matches the
        VideoConfig field, not the hardcoded v0.9.0 string.

        The lazy imports inside ``DiffusersMpsEngine.run`` resolve
        ``run_with_stderr_redaction`` and ``IMGEN_INSTALL_ROOT`` from
        their canonical modules, so we monkeypatch THOSE module
        attributes (not the engine module which doesn't import them
        at module load time)."""
        import json
        from imgen.engines.diffusers_mps_engine import DiffusersMpsEngine
        from imgen.engines.base import GenParams

        captured: dict = {}

        def fake_run_with_stderr_redaction(argv, env, stdin_data, log_file):
            captured["argv"] = argv
            captured["payload"] = json.loads(stdin_data.decode("utf-8"))
            return 0

        monkeypatch.setattr(
            "imgen.subprocess_helpers.run_with_stderr_redaction",
            fake_run_with_stderr_redaction,
        )
        monkeypatch.setattr("imgen.paths.IMGEN_INSTALL_ROOT", tmp_path)
        # Create the venv layout the engine guard expects.
        (tmp_path / ".venv-diffusers" / "bin").mkdir(parents=True)
        real_python = tmp_path / ".venv-diffusers" / "bin" / "python"
        real_python.write_text("")
        real_python.chmod(0o755)

        engine = DiffusersMpsEngine()
        model = self._make_i2v_model()
        params = GenParams(
            prompt="a samurai", negative="", width=512, height=512,
            steps=50, guidance=5.0, seed=42, quantize=0, strength=0.0,
            input_path=None, output_path=tmp_path / "out.mp4",
            loras=(), mlx_cache_gb=12, battery_stop=20,
            num_frames=25, fps=24,
        )

        engine.run(model=model, params=params)

        assert "pipeline_class" in captured["payload"]
        assert captured["payload"]["pipeline_class"] == "LTXImageToVideoPipeline"


# ── Dry-run dispatch reads from VideoConfig ─────────────────────────────


class TestDryrunReadsFromVideoConfig:
    """The dry-run formatter (``engine_dispatch._format_diffusers_video
    _dryrun``) must surface the actual pipeline_class the runner will
    receive — not the v0.9.0 literal. Otherwise a user inspecting
    ``imgen video --image foo.png --dry-run`` would see the t2v
    pipeline name even though the runner is about to load the i2v one."""

    def _make_iteration(self, *, model, params):
        from imgen.runs import Iteration
        return Iteration(
            style_name="ltx-video",
            prompt=params.prompt,
            negative=params.negative,
            final_steps=params.steps,
            final_quantize=params.quantize,
            final_guidance=params.guidance,
            final_strength=params.strength,
            output_path=params.output_path,
            loras=(),
            seed=params.seed,
            model=model,
            params=params,
        )

    def _make_video_params(self, tmp_path):
        from imgen.engines.base import GenParams
        return GenParams(
            prompt="a samurai", negative="", width=512, height=512,
            steps=50, guidance=5.0, seed=42, quantize=0, strength=0.0,
            input_path=None, output_path=tmp_path / "out.mp4",
            loras=(), mlx_cache_gb=12, battery_stop=20,
            num_frames=25, fps=24,
        )

    def test_dryrun_shows_i2v_pipeline_when_videoconfig_declares_it(
        self, tmp_path,
    ):
        from dataclasses import replace
        from imgen.engine_dispatch import _format_diffusers_video_dryrun
        from imgen.models import BUILTIN_MODELS

        ltx = BUILTIN_MODELS["ltx-video"]
        assert ltx.video is not None
        i2v_video_cfg = replace(ltx.video, pipeline_class="LTXImageToVideoPipeline")
        model = replace(ltx, video=i2v_video_cfg)
        params = self._make_video_params(tmp_path)
        it = self._make_iteration(model=model, params=params)

        text = _format_diffusers_video_dryrun(it)
        assert "LTXImageToVideoPipeline" in text
        # Negative assertion — the v0.9.0 literal must NOT appear as a
        # standalone token when i2v is the declared pipeline_class. The
        # full class name "LTXImageToVideoPipeline" contains the
        # substring "LTXPipeline"? Actually no — "LTXImageToVideoPipeline"
        # contains "Pipeline" but NOT "LTXPipeline". Pure substring
        # check below works.
        assert "pipeline_class:  LTXPipeline\n" not in text + "\n"

    def test_dryrun_shows_t2v_pipeline_for_default_ltx(self, tmp_path):
        """Backward compat: the v0.9.0 LTX-Video row keeps showing
        ``LTXPipeline`` in the dry-run."""
        from imgen.engine_dispatch import _format_diffusers_video_dryrun
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["ltx-video"]
        params = self._make_video_params(tmp_path)
        # default LTX uses guidance=3 not 5; tweak.
        from dataclasses import replace
        params = replace(params, guidance=3.0)
        it = self._make_iteration(model=model, params=params)

        text = _format_diffusers_video_dryrun(it)
        assert "LTXPipeline" in text
        assert "LTXImageToVideoPipeline" not in text

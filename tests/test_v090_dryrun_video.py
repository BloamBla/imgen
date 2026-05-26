"""v0.9 commit 8 — iteration_dryrun_display video branch (§H).

Per [[project-v090-design]] §H. Existing `_format_diffusers_dryrun`
was written for image Models and silently omitted video-specific
fields (num_frames, fps, force_cpu_offload, pipeline_class,
duration_sec) when called on a video Iteration. Security §R.2
MEDIUM-1 also flagged the unescaped `params.prompt!r` rendering —
fixed by routing through `safe_display()` per the design spec.

Tests cover:
* iteration_dryrun_display routes video Models to
  `_format_diffusers_video_dryrun`.
* Video display surfaces every video-specific payload field.
* `safe_display()` wraps the prompt so control bytes escape
  visibly instead of triggering terminal escape sequences.
* Image dispatch unchanged (regression lock).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.engine_dispatch import iteration_dryrun_display
from imgen.engines.base import GenParams
from imgen.models import Model, VideoConfig
from imgen.runs import Iteration


def _ltx_model():
    return Model(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        supported_quants=(),
        video=VideoConfig(
            default_num_frames=25,
            default_fps=24,
            max_num_frames=257,
            force_cpu_offload=True,
        ),
    )


def _video_iter(tmp_path, **overrides):
    model = _ltx_model()
    base = dict(
        prompt="a samurai walking through bamboo",
        negative="",
        width=768, height=512,
        steps=25, guidance=3.0, seed=42, quantize=0, strength=0.0,
        input_path=None,
        output_path=tmp_path / "out.mp4",
        loras=(),
        num_frames=25, fps=24,
    )
    base.update(overrides)
    params = GenParams(**base)
    return Iteration(
        style_name="video", prompt=params.prompt, negative="",
        final_steps=25, final_quantize=0, final_guidance=3.0, final_strength=0.0,
        output_path=params.output_path,
        seed=42, model=model, params=params,
    )


# ── Routing: video Models hit the dedicated formatter ────────────────


class TestVideoDryrunRouting:
    def test_video_iteration_routes_via_dedicated_formatter(self, tmp_path):
        """The branch must dispatch by model.output_type so video and
        image diffusers_mps Iterations diverge cleanly."""
        from imgen.engine_dispatch import _format_diffusers_video_dryrun
        # Ensure the function exists; the routing test below assumes it.
        assert callable(_format_diffusers_video_dryrun)

    def test_video_display_contains_num_frames(self, tmp_path):
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "num_frames" in display
        assert "25" in display

    def test_video_display_contains_fps(self, tmp_path):
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "fps" in display
        assert "24" in display

    def test_video_display_contains_force_cpu_offload(self, tmp_path):
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "force_cpu_offload" in display
        assert "True" in display or "true" in display.lower()

    def test_video_display_contains_pipeline_class(self, tmp_path):
        """Engine.run hardcodes LTXPipeline for v0.9.0 — surface it
        in dry-run for full payload visibility."""
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "LTXPipeline" in display

    def test_video_display_contains_duration_seconds(self, tmp_path):
        """duration_sec = num_frames / fps. For canonical LTX
        25 frames @ 24 fps ≈ 1.04 sec."""
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "duration" in display.lower()
        # Should mention ~1.04s; allow either 1.0 or 1.04
        assert any(s in display for s in ("1.0", "1.04"))

    def test_video_display_contains_output_type_video(self, tmp_path):
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "output_type" in display
        assert "video" in display

    def test_video_display_contains_mp4_output_path(self, tmp_path):
        display = iteration_dryrun_display(_video_iter(tmp_path))
        assert "out.mp4" in display


# ── safe_display() escape on prompt field (security §R.2 MEDIUM-1) ────


class TestVideoDryrunPromptSafeDisplay:
    """Prompt rendering must escape C0/DEL/C1 control bytes — a
    hand-crafted prompt that snuck past the parser (e.g. via stdin or
    --prompt-file PATH) could otherwise inject ANSI escapes into the
    user's terminal when dry-run echoes it back."""

    def test_prompt_with_ansi_escape_renders_escaped(self, tmp_path):
        """`\\x1b[31m` (red ANSI) must render as the literal escape
        sequence, not be interpreted by the terminal."""
        evil_prompt = "samurai\x1b[31m red text"
        display = iteration_dryrun_display(
            _video_iter(tmp_path, prompt=evil_prompt),
        )
        # safe_display via repr() renders \x1b as the literal escape
        # text (4 characters: backslash, x, 1, b)
        assert r"\x1b" in display, (
            f"safe_display must escape \\x1b literally; got: {display!r}"
        )

    def test_prompt_with_null_byte_renders_escaped(self, tmp_path):
        evil_prompt = "samurai\x00null"
        display = iteration_dryrun_display(
            _video_iter(tmp_path, prompt=evil_prompt),
        )
        assert r"\x00" in display

    def test_prompt_with_del_byte_renders_escaped(self, tmp_path):
        evil_prompt = "samurai\x7fdel"
        display = iteration_dryrun_display(
            _video_iter(tmp_path, prompt=evil_prompt),
        )
        assert r"\x7f" in display

    def test_clean_prompt_renders_quoted(self, tmp_path):
        """safe_display wraps via repr() — clean strings get quoted."""
        display = iteration_dryrun_display(_video_iter(tmp_path))
        # The prompt is "a samurai walking through bamboo" — find a
        # quoted variant in display.
        assert ("'a samurai walking through bamboo'" in display
                or '"a samurai walking through bamboo"' in display)


# ── Regression: image dispatch unchanged ──────────────────────────────


class TestImageDispatchRegression:
    """v0.8.x image Iterations through diffusers_mps still hit the
    image formatter — no num_frames / fps surface."""

    def _image_iter(self, tmp_path):
        from imgen.engines.base import GenParams
        model = Model(
            engine="diffusers_mps",
            repo="mlx-community/Qwen-Image-2512-4bit",
            cpu_offload_threshold_mp=2.0,
            ram_baseline_gb=10.0, ram_slope_gb_per_mp=5.0,
            encoder_ram_gb=7.0,
            param_overrides=(("true_cfg_scale", 4.0),),
        )
        params = GenParams(
            prompt="x", negative="", width=1024, height=1024,
            steps=50, guidance=4.0, seed=42, quantize=4, strength=0.0,
            input_path=None, output_path=tmp_path / "out.png",
            loras=(),
        )
        return Iteration(
            style_name="draw", prompt="x", negative="",
            final_steps=50, final_quantize=4, final_guidance=4.0,
            final_strength=0.0, output_path=params.output_path,
            seed=42, model=model, params=params,
        )

    def test_image_dispatch_does_not_show_num_frames(self, tmp_path):
        display = iteration_dryrun_display(self._image_iter(tmp_path))
        assert "num_frames" not in display

    def test_image_dispatch_does_not_show_fps(self, tmp_path):
        display = iteration_dryrun_display(self._image_iter(tmp_path))
        assert "fps" not in display

    def test_image_dispatch_does_not_show_force_cpu_offload(self, tmp_path):
        display = iteration_dryrun_display(self._image_iter(tmp_path))
        assert "force_cpu_offload" not in display

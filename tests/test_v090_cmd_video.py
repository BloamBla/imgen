"""v0.9 commit 7 — cmd_video + parser + build_video_iteration tests.

Per [[project-v090-design]] §I. Covers four surfaces:

* Parser stanza — --duration/--num-frames mutex, --fps allowlist,
  range gates fire before any subprocess spawn.
* build_video_iteration — resolves num_frames from
  --duration/--num-frames/default via VideoConfig alignment; mp4
  output path; single-shot semantics.
* cmd_video dispatch — args.enhance die-early gate, dry-run skips
  ensure_video_deps_or_die, _orchestrate_t2x routing.
* CLI surface — `imgen video ...` route through _HANDLERS;
  BatchContext.command="video" accepted by the Literal widening.

The actual LTX subprocess never spawns in these tests — mocks at
the engine.run / load_backend_and_token seams keep the suite under
10s and free of real GPU work.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imgen.defaults import DEFAULTS


# ── Test fixtures ─────────────────────────────────────────────────────


def _ltx_video_config():
    from imgen.models import VideoConfig
    return VideoConfig(
        default_num_frames=25,
        default_fps=24,
        max_num_frames=257,
        force_cpu_offload=True,
    )


def _ltx_backend():
    """Get the LTX Backend wrapper from BUILTIN_BACKENDS (derived
    from BUILTIN_MODELS automatically)."""
    from imgen.backends import BUILTIN_BACKENDS
    return BUILTIN_BACKENDS["ltx-video"]


def _ltx_model_via_user_toml(tmp_path, monkeypatch):
    """Set up a user TOML so 'ltx-video' resolves via the merged
    registry (built-in LTX row lands at commit 9; until then we use
    user-TOML scaffolding for tests)."""
    # Build a Model + register in BUILTIN_MODELS so _model_for_validate
    # returns it. Cheaper than wiring user-TOML loader.
    from imgen import models
    from imgen.models import Model
    ltx = Model(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        video=_ltx_video_config(),
    )
    patched = dict(models.BUILTIN_MODELS)
    patched["ltx-video"] = ltx
    monkeypatch.setattr(models, "BUILTIN_MODELS", patched)
    # backends.BUILTIN_BACKENDS is derived from BUILTIN_MODELS — extend it too
    from imgen import backends
    if hasattr(backends, "BUILTIN_BACKENDS"):
        be_patched = dict(backends.BUILTIN_BACKENDS)
        # Skip if backends-side derive fails (some Model shapes don't
        # convert); cmd_video tests primarily exercise the Model path.
        monkeypatch.setattr(backends, "BUILTIN_BACKENDS", be_patched)
    return ltx


def _make_video_args(**overrides):
    """Build a SimpleNamespace shaped like parsed `imgen video` args."""
    defaults = dict(
        prompt="a samurai walking",
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
        width=768,
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
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        imgen_config_enhance={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Parser stanza ─────────────────────────────────────────────────────


class TestVideoParserStanza:
    """Parser must accept `imgen video <prompt>` + enforce mutex
    + range gates BEFORE any subprocess spawn (~50ms gate per §S)."""

    def _parse(self, argv):
        from imgen.parser import build_parser
        parser = build_parser(defaults=DEFAULTS)
        return parser.parse_args(argv)

    def test_imgen_video_with_positional_prompt_parses(self):
        args = self._parse(["video", "a samurai"])
        assert args.command == "video"
        assert args.prompt == "a samurai"

    def test_default_width_768_height_512(self):
        args = self._parse(["video", "a samurai"])
        assert args.width == 768
        assert args.height == 512

    def test_default_model_ltx_video(self):
        args = self._parse(["video", "a samurai"])
        assert args.model == "ltx-video"

    def test_duration_and_num_frames_mutex_rejected(self):
        with pytest.raises(SystemExit) as exc:
            self._parse([
                "video", "a samurai",
                "--duration", "1.0", "--num-frames", "25",
            ])
        assert exc.value.code == 2

    def test_duration_in_range_accepted(self):
        args = self._parse(["video", "a samurai", "--duration", "1.5"])
        assert args.duration == 1.5

    def test_duration_above_60_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["video", "a samurai", "--duration", "65"])

    def test_num_frames_in_range_accepted(self):
        args = self._parse(["video", "a samurai", "--num-frames", "25"])
        assert args.num_frames == 25

    def test_num_frames_above_cap_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["video", "a samurai", "--num-frames", "2000"])

    def test_num_frames_zero_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["video", "a samurai", "--num-frames", "0"])

    @pytest.mark.parametrize("fps", [24, 25, 30])
    def test_fps_in_allowlist_accepted(self, fps):
        args = self._parse(["video", "a samurai", "--fps", str(fps)])
        assert args.fps == fps

    def test_fps_60_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["video", "a samurai", "--fps", "60"])

    def test_output_path_must_be_safe_extension(self):
        with pytest.raises(SystemExit):
            self._parse(["video", "a samurai", "--output", "/tmp/x.exe"])

    def test_output_path_mp4_accepted(self):
        args = self._parse(["video", "a samurai", "--output", "/tmp/x.mp4"])
        assert args.output == "/tmp/x.mp4"


# ── build_video_iteration ─────────────────────────────────────────────


class TestBuildVideoIteration:
    """Single-shot iteration builder with VideoConfig-driven
    num_frames resolution."""

    def test_returns_single_iteration(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=25)
        iterations = build_video_iteration(
            args=args, prompt="a samurai",
            merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert len(iterations) == 1

    def test_num_frames_explicit_wins(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=49)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].params.num_frames == 49

    def test_duration_resolves_to_aligned_num_frames(self, tmp_path, monkeypatch):
        """1.0s @ 24fps = 24 frames; LTX alignment 8k+1 → ceil to 25."""
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(duration=1.0)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        # ceil(24-1)/8 = 3 → 3*8+1 = 25
        assert iters[0].params.num_frames == 25

    def test_duration_rounds_up_not_down(self, tmp_path, monkeypatch):
        """Architect §R.1 MED-2: ceil UP so output >= requested duration."""
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        # 0.5s @ 24fps = 12 frames; ceil to next 8k+1 → 17
        args = _make_video_args(duration=0.5)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].params.num_frames == 17

    def test_duration_rounding_emits_warn(
        self, tmp_path, monkeypatch, capsys,
    ):
        """v0.9.2 B-5 closure of design memo §I.1 'Warn line if rounding
        occurred'. When --duration produces a num_frames different from
        the naive int(duration*fps), surface a warn line so the user
        knows the output is longer than requested."""
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        # 0.5s → naive target 12 → aligned ceil 17 (≠ 12, warn must fire).
        args = _make_video_args(duration=0.5)
        build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        # warn() writes to stdout (only err() routes to stderr in
        # colors.py); see test_parser.py convention.
        out = capsys.readouterr().out
        assert "--duration" in out and "0.5" in out, (
            f"warn must mention --duration value; got: {out!r}"
        )
        assert "17" in out and "frames" in out, (
            f"warn must mention resolved frame count; got: {out!r}"
        )

    def test_duration_exact_alignment_no_warn(
        self, tmp_path, monkeypatch, capsys,
    ):
        """If --duration happens to land precisely on the alignment
        (rare but possible with future Models), no warn line. v0.9.0
        LTX with alignment 8k+1 has no exact-integer-second match at
        24 fps so we use --num-frames explicit (which bypasses the
        rounding path entirely) as the negative case."""
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        # Explicit --num-frames bypasses _resolve's duration branch
        # entirely; no rounding consideration → no warn.
        args = _make_video_args(num_frames=25)
        build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        out = capsys.readouterr().out
        assert "rounded" not in out.lower(), (
            f"explicit --num-frames must not trigger rounding warn; "
            f"got: {out!r}"
        )

    def test_neither_uses_default_num_frames(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args()  # no duration, no num_frames
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].params.num_frames == 25  # default_num_frames

    def test_fps_default_from_video_config(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=25)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].params.fps == 24  # default_fps

    def test_fps_explicit_wins(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=25, fps=30)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].params.fps == 30

    def test_output_path_uses_mp4_suffix(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=25)
        iters = build_video_iteration(
            args=args, prompt="a samurai walking", merged_defaults=DEFAULTS,
            be=_ltx_backend(),
            width=768, height=512,
            explicit_output=None, run_dir=tmp_path,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].output_path.suffix == ".mp4"

    def test_explicit_output_overrides_slug_path(self, tmp_path, monkeypatch):
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        explicit = tmp_path / "custom.mp4"
        args = _make_video_args(num_frames=25)
        iters = build_video_iteration(
            args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
            width=768, height=512,
            explicit_output=explicit, run_dir=None,
            base_seed=42, num_iterations=1,
        )
        assert iters[0].output_path == explicit

    def test_num_iterations_above_1_rejected(self, tmp_path, monkeypatch):
        """v0.9.0 single-shot per §I — N>1 is v0.9.x roadmap."""
        from imgen.build_iteration import build_video_iteration
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        args = _make_video_args(num_frames=25)
        with pytest.raises(ValueError, match="single-shot"):
            build_video_iteration(
                args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
                width=768, height=512,
                explicit_output=None, run_dir=tmp_path,
                base_seed=42, num_iterations=2,
            )

    def test_non_video_model_rejected(self, tmp_path):
        """Defensive: build_video_iteration on a model without
        VideoConfig should raise (cmd_video gates upstream but the
        helper still defends)."""
        from imgen.build_iteration import build_video_iteration
        args = _make_video_args(num_frames=25, model="flux-dev")
        with pytest.raises(ValueError, match="VideoConfig"):
            build_video_iteration(
                args=args, prompt="x", merged_defaults=DEFAULTS, be=_ltx_backend(),
                width=768, height=512,
                explicit_output=None, run_dir=tmp_path,
                base_seed=42, num_iterations=1,
            )


# ── cmd_video dispatch ────────────────────────────────────────────────


class TestCmdVideoDispatch:
    """cmd_video routing through _orchestrate_t2x with video-specific
    gates (enhance die-early, lazy deps install)."""

    def test_enhance_dies_early_with_helpful_message(self, tmp_path, monkeypatch, capsys):
        """Per §S.4: LTX has no enhancer in v0.9.0 → die() rather
        than silently ignore."""
        from imgen.commands.video import cmd_video
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        # Stub load_backend_and_token so the dispatch reaches enhancer gate
        from imgen.backends import BACKENDS
        monkeypatch.setattr(
            "imgen.cmd_helpers.load_backend_and_token",
            lambda args: ("ltx-video", BACKENDS.get("flux-dev"), None,
                          Path("/fake"), None),
        )
        args = _make_video_args(enhance=True, dry_run=True,
                                 output_dir=str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            cmd_video(args)
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "enhance" in stderr.lower()

    def test_dry_run_skips_ensure_video_deps_or_die(self, tmp_path, monkeypatch):
        """§E.5.7: --dry-run must NOT trigger the deps install
        prompt. Orchestrator gates pre_dispatch_fn on args.dry_run."""
        from imgen.commands import video as video_mod
        _ltx_model_via_user_toml(tmp_path, monkeypatch)
        from imgen.backends import BACKENDS
        monkeypatch.setattr(
            "imgen.cmd_helpers.load_backend_and_token",
            lambda args: ("ltx-video", BACKENDS.get("flux-dev"), None,
                          Path("/fake"), None),
        )

        deps_called = {"n": 0}

        def fake_ensure():
            deps_called["n"] += 1

        monkeypatch.setattr(
            video_mod, "ensure_video_deps_or_die", fake_ensure,
        )

        args = _make_video_args(dry_run=True, output_dir=str(tmp_path))
        rc = video_mod.cmd_video(args)
        assert rc == 0
        assert deps_called["n"] == 0, (
            "ensure_video_deps_or_die must NOT fire under --dry-run"
        )


# ── BatchContext command Literal extension ────────────────────────────


class TestBatchContextCommandVideo:
    """§G architect §R.1 MED-5: BatchContext.command Literal widened
    with "video"."""

    def test_batch_context_command_video_accepted(self):
        from imgen.runs import BatchContext
        ctx = BatchContext(
            model="ltx-video", seed=42, width=768, height=512,
            input_path=None, effective_custom_prompt=None,
            args=SimpleNamespace(), batch_id=None, env={},
            command="video",
        )
        assert ctx.command == "video"


# ── CLI routing ───────────────────────────────────────────────────────


class TestCliVideoRouting:
    """`imgen video` must route through _HANDLERS to cmd_video."""

    def test_video_subcommand_in_known_subcommands(self):
        from imgen.cli import _KNOWN_SUBCOMMANDS
        assert "video" in _KNOWN_SUBCOMMANDS

    def test_video_handler_is_cmd_video(self):
        from imgen.cli import _HANDLERS
        from imgen.commands.video import cmd_video
        assert _HANDLERS["video"] is cmd_video

    def test_cmd_video_exported_from_commands_package(self):
        from imgen.commands import cmd_video as exported
        from imgen.commands.video import cmd_video as direct
        assert exported is direct

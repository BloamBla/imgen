"""v0.7.5: `imgen refine <input>` — Hires-Fix orchestrator.

Self-contained subcommand (not a delegation to cmd_generate) — owns
its own prompt resolution + dimension scaling + Iteration build.
Mocks mflux at the existing run_with_stderr_redaction seam so no
real GPU work in the suite.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.defaults import DEFAULTS


# ── Parser stanza ────────────────────────────────────────────────────


def _parse_refine(*argv):
    from imgen.parser import build_parser
    return build_parser({
        "style": "pixar", "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 3.5, "strength": 0.55,
        "mlx_cache_gb": 12, "battery_stop": 20,
    }).parse_args(["refine", *argv])


class TestRefineParser:
    def test_positional_input_required(self):
        with pytest.raises(SystemExit):
            _parse_refine()

    def test_default_backend_is_flux2_klein_edit(self):
        """v0.7.5 ships FLUX.2-klein-9B as the default refine backend
        (native ~4 MP support past FLUX.1's ~1.5K ceiling)."""
        args = _parse_refine("photo.png")
        assert args.backend == "flux2-klein-edit-9b"

    def test_default_quantize_is_4(self):
        """Q4 is the safe default — Q8 + 2K² activations on 32GB Mac
        can OOM with klein-9B."""
        args = _parse_refine("photo.png")
        assert args.quantize == 4

    def test_default_strength_is_0_3(self):
        """Low strength preserves input composition (the whole point
        of Hires-Fix)."""
        args = _parse_refine("photo.png")
        assert args.strength == 0.3

    def test_scale_arg(self):
        args = _parse_refine("photo.png", "--scale", "2.0")
        assert args.scale == 2.0

    def test_explicit_width_height(self):
        args = _parse_refine("photo.png", "--width", "1536", "--height", "1536")
        assert args.width == 1536
        assert args.height == 1536


# ── _round_to_multiple_of_16 ─────────────────────────────────────────


class TestRoundToMultipleOf16:
    def test_already_multiple(self):
        from imgen.commands.refine import _round_to_multiple_of_16
        assert _round_to_multiple_of_16(1024) == 1024
        assert _round_to_multiple_of_16(1536) == 1536

    def test_rounds_up_from_below(self):
        from imgen.commands.refine import _round_to_multiple_of_16
        # 1500 + 8 = 1508; 1508 // 16 = 94; 94 * 16 = 1504
        assert _round_to_multiple_of_16(1500) == 1504

    def test_rounds_to_nearest(self):
        from imgen.commands.refine import _round_to_multiple_of_16
        # 1512 → halfway between 1504 and 1520 → rounds up
        assert _round_to_multiple_of_16(1512) == 1520


# ── _resolve_target_dimensions ───────────────────────────────────────


class TestResolveTargetDimensions:
    """v0.7.5: pure function — dims threaded in by cmd_refine, no
    PIL I/O inside _resolve_target_dimensions itself."""

    def test_default_scale_1_5(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=None, width=None, height=None)
        w, h = _resolve_target_dimensions(args, 1024, 1024)
        assert w == 1536
        assert h == 1536

    def test_scale_2(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=2.0, width=None, height=None)
        w, h = _resolve_target_dimensions(args, 1024, 1024)
        assert w == 2048
        assert h == 2048

    def test_explicit_dims(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=None, width=1280, height=720)
        w, h = _resolve_target_dimensions(args, 1024, 1024)
        assert w == 1280
        assert h == 720

    def test_explicit_dims_rounded_to_16(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=None, width=1500, height=1500)
        w, h = _resolve_target_dimensions(args, 1024, 1024)
        assert w == 1504
        assert h == 1504

    def test_scale_and_dims_mutex(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=2.0, width=1280, height=720)
        with pytest.raises(SystemExit):
            _resolve_target_dimensions(args, 1024, 1024)

    def test_dims_require_both(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=None, width=1280, height=None)
        with pytest.raises(SystemExit):
            _resolve_target_dimensions(args, 1024, 1024)

    def test_rectangular_input_scales_both_dims(self):
        from imgen.commands.refine import _resolve_target_dimensions
        args = SimpleNamespace(scale=1.5, width=None, height=None)
        w, h = _resolve_target_dimensions(args, 1024, 768)
        assert w == 1536
        # 768 * 1.5 = 1152, multiple of 16
        assert h == 1152


class TestReadImageDimensions:
    """v0.7.5 IMPORTANT #1: narrow except (OSError | UnidentifiedImageError)
    + Pillow-missing diagnostic. Lock-in tests for the structural fix."""

    def test_reads_png_dims(self, tmp_path):
        from PIL import Image
        from imgen.commands.refine import _read_image_dimensions
        p = tmp_path / "in.png"
        Image.new("RGB", (1280, 720), "white").save(p)
        assert _read_image_dimensions(p) == (1280, 720)

    def test_non_image_file_dies(self, tmp_path):
        """Garbage bytes → UnidentifiedImageError → die(code=2),
        NOT a swallowed AttributeError/KeyError from a downstream
        consumer."""
        from imgen.commands.refine import _read_image_dimensions
        p = tmp_path / "garbage.png"
        p.write_bytes(b"this is not a PNG file")
        with pytest.raises(SystemExit) as exc_info:
            _read_image_dimensions(p)
        assert exc_info.value.code == 2

    def test_missing_file_dies(self, tmp_path):
        """Missing file → OSError (FileNotFoundError) → die(code=2)."""
        from imgen.commands.refine import _read_image_dimensions
        with pytest.raises(SystemExit) as exc_info:
            _read_image_dimensions(tmp_path / "nonexistent.png")
        assert exc_info.value.code == 2


# ── build_refine_iteration ───────────────────────────────────────────


def _make_args(**overrides):
    """SimpleNamespace mirroring the refine parser shape."""
    defaults = dict(
        input="photo.png",
        scale=None,
        width=None,
        height=None,
        prompt=None,
        output=None,
        output_dir=None,
        steps=None,
        guidance=None,
        strength=0.3,
        seed=42,
        backend="flux2-klein-edit-9b",
        quantize=4,
        preview=False,
        no_open=True,
        yes=True,
        dry_run=False,
        force=True,
        lora=None,
        no_lora=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        imgen_config_enhance={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildRefineIteration:
    def test_no_style_no_negative(self, tmp_path):
        """v0.7.5: refine bypasses styles machinery. Iteration's
        negative is empty, NOT pixar's "deformed, blurry...". Iteration's
        style_name is "refine", NOT "pixar"."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_refine_iteration
        it = build_refine_iteration(
            args=_make_args(),
            input_path=Path("/tmp/in.png"),
            prompt="refine prompt here",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux2-klein-edit-9b"],
            binary=Path("/fake/mflux-generate-flux2-edit"),
            width=1536,
            height=1536,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert it.style_name == "refine"
        assert it.negative == ""

    def test_output_path_has_refined_suffix(self, tmp_path):
        """v0.7.5: output naming is `<input.stem>-refined.png`, not
        `<input.stem>-<style>.png`."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_refine_iteration
        it = build_refine_iteration(
            args=_make_args(),
            input_path=Path("/tmp/samurai.png"),
            prompt="refine",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux2-klein-edit-9b"],
            binary=Path("/fake/mflux"),
            width=1536, height=1536,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert it.output_path.name == "samurai-refined.png"

    def test_cmd_uses_image_paths_plural(self, tmp_path):
        """FLUX.2-klein-edit uses `--image-paths` (plural, multi-image
        capable), NOT `--image-path` (FLUX-Kontext singular). Confirm
        build_mflux_cmd respects backend.image_flag."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_refine_iteration
        it = build_refine_iteration(
            args=_make_args(),
            input_path=Path("/tmp/in.png"),
            prompt="refine",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux2-klein-edit-9b"],
            binary=Path("/fake/mflux"),
            width=1536, height=1536,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert "--image-paths" in it.cmd
        idx = it.cmd.index("--image-paths")
        assert it.cmd[idx + 1] == "/tmp/in.png"

    def test_seed_recorded_on_iteration(self, tmp_path):
        """v0.7.3 fix carry-forward: Iteration.seed is the per-iter
        seed for history reproducibility."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_refine_iteration
        it = build_refine_iteration(
            args=_make_args(),
            input_path=Path("/tmp/in.png"),
            prompt="refine",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux2-klein-edit-9b"],
            binary=Path("/fake/mflux"),
            width=1536, height=1536,
            explicit_output=None,
            run_dir=tmp_path,
            seed=12345,
        )
        assert it.seed == 12345


# ── cmd_refine integration (dry-run) ─────────────────────────────────


class TestCmdRefineDryRun:
    def test_dry_run_prints_cmd(self, tmp_path, monkeypatch, capsys):
        """End-to-end: cmd_refine dry-run emits a valid argv with
        flux2-klein-edit-9b backend + flux2-klein-9b model selector."""
        from imgen.backends import BACKENDS
        from imgen.commands.refine import cmd_refine
        from PIL import Image

        # Create a real input file for resolution detection.
        input_path = tmp_path / "samurai.png"
        Image.new("RGB", (1024, 1024), "white").save(input_path)

        def fake_load(args):
            return ("flux2-klein-edit-9b", BACKENDS["flux2-klein-edit-9b"],
                    "tok", Path("/fake/mflux-generate-flux2-edit"), None)
        monkeypatch.setattr(
            "imgen.commands.refine.load_backend_and_token", fake_load,
        )

        args = _make_args(
            input=str(input_path),
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_refine(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Target dimensions from default --scale 1.5.
        assert "1536×1536" in out
        # Binary + model selector.
        assert "mflux-generate-flux2-edit" in out
        assert "flux2-klein-9b" in out
        # Default refine prompt content.
        assert "sharper detail" in out
        # NOT pixar's negative prompt or style suffix.
        assert "deformed, blurry" not in out
        assert "-pixar.png" not in out

    def test_flux2_klein_edit_pins_guidance_to_1(
        self, tmp_path, monkeypatch, capsys,
    ):
        """v0.7.6 hotfix: mflux-generate-flux2-edit refuses --guidance
        != 1.0 on non-base FLUX.2 models (klein-9b is the distilled
        edit variant). cmd_refine pins guidance=1.0 for this backend
        regardless of what the user passed — model property, not user
        knob. Lock-in: even an explicit --guidance 3.5 must collapse
        to --guidance 1.0 in argv when this backend is selected."""
        from imgen.backends import BACKENDS
        from imgen.commands.refine import cmd_refine
        from PIL import Image

        input_path = tmp_path / "samurai.png"
        Image.new("RGB", (1024, 1024), "white").save(input_path)

        def fake_load(args):
            return ("flux2-klein-edit-9b", BACKENDS["flux2-klein-edit-9b"],
                    "tok", Path("/fake/mflux-generate-flux2-edit"), None)
        monkeypatch.setattr(
            "imgen.commands.refine.load_backend_and_token", fake_load,
        )

        # User explicitly passes the FLUX.1-Kontext-shaped 3.5 default.
        # Backend constraint must override.
        args = _make_args(
            input=str(input_path),
            dry_run=True,
            output_dir=str(tmp_path),
            guidance=3.5,
        )
        rc = cmd_refine(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "--guidance 1.0" in out
        assert "--guidance 3.5" not in out

    def test_input_not_found_dies(self, tmp_path):
        from imgen.commands.refine import cmd_refine
        args = _make_args(
            input=str(tmp_path / "nonexistent.png"),
            dry_run=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_refine(args)
        assert exc_info.value.code == 2

    def test_input_is_directory_dies(self, tmp_path):
        from imgen.commands.refine import cmd_refine
        args = _make_args(input=str(tmp_path), dry_run=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_refine(args)
        assert exc_info.value.code == 2


# ── Backend registration ─────────────────────────────────────────────


class TestFlux2KleinEditBackend:
    def test_registered(self):
        from imgen.backends import BACKENDS
        assert "flux2-klein-edit-9b" in BACKENDS

    def test_uses_flux2_edit_binary(self):
        from imgen.backends import BACKENDS
        assert BACKENDS["flux2-klein-edit-9b"].binary == "mflux-generate-flux2-edit"

    def test_uses_image_paths_plural(self):
        from imgen.backends import BACKENDS
        assert BACKENDS["flux2-klein-edit-9b"].image_flag == "--image-paths"

    def test_supports_strength_false(self):
        from imgen.backends import BACKENDS
        assert BACKENDS["flux2-klein-edit-9b"].supports_strength is False

    def test_model_selector_in_extra_args(self):
        from imgen.backends import BACKENDS
        extra = BACKENDS["flux2-klein-edit-9b"].extra_args
        assert "-m" in extra
        assert "flux2-klein-9b" in extra

    def test_unique_lora_compat_group(self):
        """FLUX.2 is architecturally different from FLUX.1 — LoRAs
        don't cross-load. Compat group must be unique."""
        from imgen.backends import BACKENDS
        assert BACKENDS["flux2-klein-edit-9b"].lora_compat_group == "flux2-klein-9b"
        # Sanity: not accidentally sharing with FLUX.1 / flux-dev groups.
        assert BACKENDS["flux"].lora_compat_group == "flux-1"
        assert BACKENDS["flux-dev"].lora_compat_group == "flux-dev"

    def test_hf_gated_repo_populated(self):
        from imgen.backends import BACKENDS
        assert BACKENDS["flux2-klein-edit-9b"].hf_gated_repo == (
            "black-forest-labs/FLUX.2-klein-9B"
        )

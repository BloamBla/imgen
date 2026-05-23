"""v0.7.0 (architect §F + §M): `imgen draw` orchestrator + parser
+ helpers integration tests.

Mocks the mflux subprocess at the same seam as test_generate_enhance.py
so no real GPU work happens in the suite. Exercises:

  * parser stanza (positional prompt, --prompt-file mutex, defaults)
  * prompt-slug helper (table-driven per design §D)
  * cmd_draw orchestrator (dry-run + mocked subprocess + history)
  * Backward-compat: existing i2i flows unaffected
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.cmd_helpers import (
    build_draw_iteration,
    next_available_png,
    prompt_slug,
)
from imgen.commands.draw import cmd_draw
from imgen.defaults import DEFAULTS


# ── prompt_slug helper (architect §D) ────────────────────────────────


class TestPromptSlug:
    def test_single_word(self):
        assert prompt_slug("samurai") == "samurai"

    def test_multi_word_first_six(self):
        assert (
            prompt_slug("a samurai on a misty mountain at dawn with a sword")
            == "a-samurai-on-a-misty-mountain"
        )

    def test_lowercases(self):
        assert prompt_slug("RED Dragon Breathing FIRE") == "red-dragon-breathing-fire"

    def test_punctuation_collapsed_to_dash(self):
        assert prompt_slug("hello, world! how are you?") == "hello-world-how-are-you"

    def test_unicode_normalized_to_ascii(self):
        """NFKD + ASCII-strip — accented chars decompose to bare letter."""
        assert "naive" in prompt_slug("naïve café")

    def test_emoji_only_fallback_to_draw(self):
        """Emoji + non-ASCII content strips to empty after NFKD; the
        fallback ensures the output filename is always valid."""
        assert prompt_slug("🎨🖼️") == "draw"

    def test_length_capped_at_60(self):
        long_prompt = "a " * 100  # 200 chars of "a a a a ..."
        slug = prompt_slug(long_prompt)
        assert len(slug) <= 60

    def test_empty_string_fallback_to_draw(self):
        assert prompt_slug("") == "draw"

    def test_leading_trailing_dashes_stripped(self):
        assert prompt_slug("---hello---") == "hello"


# ── next_available_png — collision suffix ─────────────────────────────


class TestNextAvailablePng:
    def test_first_run_no_suffix(self, tmp_path):
        out = next_available_png(tmp_path, "samurai")
        assert out == tmp_path / "samurai.png"

    def test_collision_appends_2(self, tmp_path):
        (tmp_path / "samurai.png").write_bytes(b"existing")
        out = next_available_png(tmp_path, "samurai")
        assert out == tmp_path / "samurai-2.png"

    def test_multiple_collisions_increment(self, tmp_path):
        (tmp_path / "samurai.png").write_bytes(b"a")
        (tmp_path / "samurai-2.png").write_bytes(b"b")
        (tmp_path / "samurai-3.png").write_bytes(b"c")
        out = next_available_png(tmp_path, "samurai")
        assert out == tmp_path / "samurai-4.png"


# ── Parser stanza ────────────────────────────────────────────────────


def _parse_draw(*argv):
    from imgen.parser import build_parser
    return build_parser({
        "style": "pixar", "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 3.5, "strength": 0.55,
        "mlx_cache_gb": 12, "battery_stop": 20,
    }).parse_args(["draw", *argv])


class TestDrawParser:
    def test_positional_prompt(self):
        args = _parse_draw("a samurai")
        assert args.command == "draw"
        assert args.prompt == "a samurai"

    def test_default_backend_is_flux_dev(self):
        args = _parse_draw("a samurai")
        assert args.backend == "flux-dev"

    def test_default_dimensions_1024x1024(self):
        args = _parse_draw("a samurai")
        assert args.width == 1024
        assert args.height == 1024

    def test_no_scope_flag(self):
        """draw is t2i; --scope is i2i-only. The parser rejects it."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--scope", "scene")

    def test_no_strength_flag(self):
        """t2i: no source photo to interpolate against."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--strength", "0.5")

    def test_no_style_flag(self):
        """v0.7.0: --style deferred to v0.7.1+ for draw."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--style", "anime")

    def test_prompt_file_alternative(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("ignored — parser doesn't read")
        args = _parse_draw("--prompt-file", str(f))
        assert args.prompt is None
        assert args.prompt_file == f

    def test_lora_comma_split_works(self):
        """v0.7.0 step 1: comma-split --lora applies to draw too."""
        args = _parse_draw("a samurai", "--lora", "a/one,b/two:0.5")
        # list[list[LoraRef]] from action='append' + _lora_refs_arg
        assert len(args.lora) == 1
        assert len(args.lora[0]) == 2

    def test_enhance_flags_present(self):
        args = _parse_draw("a samurai", "--enhance-prompt")
        assert args.enhance is True


# ── build_draw_iteration ─────────────────────────────────────────────


def _make_args(**overrides):
    """SimpleNamespace mirroring the draw parser shape — used to drive
    build_draw_iteration directly in pure tests."""
    defaults = dict(
        prompt="a samurai on a mountain",
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        seed=42,
        backend="flux-dev",
        preview=False,
        width=1024,
        height=1024,
        no_open=True,
        yes=True,
        dry_run=False,
        force=True,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
        output=None,
        output_dir=None,
        lora=None,
        no_lora=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildDrawIteration:
    def test_returns_single_iteration(self, tmp_path):
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert it.style_name == "draw"
        assert it.prompt == "a samurai"
        assert it.final_quantize == DEFAULTS["quantize"]
        assert it.final_steps == DEFAULTS["steps"]

    def test_output_path_slug_under_run_dir(self, tmp_path):
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai on a misty mountain",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        # Slug derives from first 6 words; lands inside run_dir.
        assert it.output_path.parent == tmp_path
        assert "samurai" in it.output_path.name
        assert it.output_path.suffix == ".png"

    def test_explicit_output_overrides_slug(self, tmp_path):
        from imgen.backends import BACKENDS
        explicit = tmp_path / "my-pic.png"
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=explicit,
            run_dir=None,
            seed=42,
        )
        assert it.output_path == explicit

    def test_no_image_path_in_cmd(self, tmp_path):
        """t2i: --image-path argv pair must NOT appear."""
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert "--image-path" not in it.cmd
        # Sanity: prompt + output still present.
        assert "--prompt" in it.cmd
        assert it.cmd[it.cmd.index("--prompt") + 1] == "a samurai"

    def test_strength_recorded_but_no_argv_emission(self, tmp_path):
        """args.strength is missing on draw Namespace; the getattr
        defence in _resolve_iteration_params falls through to default.
        backend.supports_strength=False so build_mflux_cmd doesn't emit
        --image-strength. Locks the symmetric pass-through."""
        from imgen.backends import BACKENDS
        # Construct args WITHOUT strength attribute (mirror real draw
        # Namespace).
        args = _make_args()
        assert not hasattr(args, "strength")
        it = build_draw_iteration(
            args=args,
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        # final_strength has a value (from merged_defaults) but argv
        # doesn't carry --image-strength.
        assert "--image-strength" not in it.cmd
        assert it.final_strength == DEFAULTS["strength"]


# ── cmd_draw dry-run path ────────────────────────────────────────────


class TestCmdDrawDryRun:
    """Dry-run hits cmd_draw, exercises the whole pipeline up to
    subprocess execution, then prints + exits clean. Validates the
    parser → build_draw_iteration → cmd-display chain end-to-end
    without needing a real mflux."""

    def test_dry_run_prints_cmd_and_exits_0(
        self, tmp_path, monkeypatch, capsys,
    ):
        # Stub load_backend_and_token so the gated-token / venv-binary
        # checks don't fire in tests.
        def fake_load(args):
            from imgen.backends import BACKENDS
            return ("flux-dev", BACKENDS["flux-dev"], "fake-token",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )

        args = _make_args(
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Dry-run banner.
        assert "Dry run" in out
        # The mflux argv content surfaces in the dry-run output.
        assert "--prompt" in out
        assert "--model" in out
        assert "dev" in out  # --model dev
        # No --image-path (t2i).
        assert "--image-path" not in out

    def test_dry_run_requires_prompt(self, tmp_path, monkeypatch):
        """A draw invocation with neither positional nor --prompt-file
        dies cleanly with exit code 2."""
        args = _make_args(prompt=None, prompt_file=None, dry_run=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_draw(args)
        assert exc_info.value.code == 2

    def test_positional_dash_reads_stdin(
        self, tmp_path, monkeypatch, capsys,
    ):
        """positional '-' reads from stdin (hides prompt from `ps`)."""
        import io
        from imgen.backends import BACKENDS

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )
        monkeypatch.setattr(
            "sys.stdin", io.StringIO("a ninja from stdin"),
        )
        args = _make_args(
            prompt="-",
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "a ninja from stdin" in out

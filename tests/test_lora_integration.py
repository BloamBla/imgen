"""v0.6 Phase 2B — build_iterations integration of the LoRA stack.

Covers:

* :func:`resolve_effective_loras` — combines style-declared LoRAs +
  CLI ``--lora`` + ``--no-lora`` opt-out into the final tuple.
* :func:`prepend_trigger_words` — auto-prepends each compatible
  LoRA's ``trigger`` to the prompt when missing (LoRAs often require
  a specific token to activate; user shouldn't have to know which).
* End-to-end via build_iterations: style.loras → effective_loras →
  prompt with trigger prepended → mflux argv contains --lora-paths /
  --lora-scales for compatible entries, drops incompatible with warn.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.backends import Backend
from imgen.cmd_helpers import (
    prepend_trigger_words,
    resolve_effective_loras,
)
from imgen.defaults import DEFAULTS
from imgen.styles import LoraRef


# ── resolve_effective_loras ────────────────────────────────────────────


class TestResolveEffectiveLoras:
    def test_neither_style_nor_cli_yields_empty(self):
        preset = {"prompt": "x"}
        out = resolve_effective_loras(preset, cli_lora=None, no_lora=False)
        assert out == ()

    def test_no_lora_overrides_style_loras(self):
        preset = {
            "prompt": "x",
            "loras": (LoraRef(ref="style/lora", weight=0.8),),
        }
        out = resolve_effective_loras(preset, cli_lora=None, no_lora=True)
        assert out == ()

    def test_no_lora_overrides_cli_loras_too(self):
        """argparse enforces --lora + --no-lora mutex at parse time, but
        if a programmatic caller bypasses that and passes both,
        --no-lora wins (the explicit drop-all signal)."""
        preset = {"prompt": "x"}
        cli = [LoraRef(ref="cli/lora")]
        out = resolve_effective_loras(preset, cli_lora=cli, no_lora=True)
        assert out == ()

    def test_style_loras_passed_through_when_no_cli(self):
        a = LoraRef(ref="a/1", weight=0.8)
        b = LoraRef(ref="b/2", weight=0.4)
        preset = {"prompt": "x", "loras": (a, b)}
        out = resolve_effective_loras(preset, cli_lora=None, no_lora=False)
        assert out == (a, b)

    def test_cli_only_when_style_has_none(self):
        cli_lora = [LoraRef(ref="cli/lora", weight=0.7)]
        preset = {"prompt": "x"}
        out = resolve_effective_loras(preset, cli_lora=cli_lora, no_lora=False)
        assert out == tuple(cli_lora)

    def test_cli_appends_to_style_loras_in_order(self):
        """Style LoRAs come FIRST, then CLI LoRAs. mflux applies in
        argv order, so the user's CLI additions layer ON TOP of the
        style's curated base."""
        style_a = LoraRef(ref="style/a", weight=0.8)
        style_b = LoraRef(ref="style/b", weight=0.4)
        cli_x = LoraRef(ref="cli/x", weight=0.5)
        cli_y = LoraRef(ref="cli/y", weight=0.3)
        preset = {"prompt": "x", "loras": (style_a, style_b)}
        out = resolve_effective_loras(
            preset, cli_lora=[cli_x, cli_y], no_lora=False,
        )
        assert out == (style_a, style_b, cli_x, cli_y)

    def test_empty_cli_list_treated_as_none(self):
        """argparse passes None for absent --lora; if a caller passes
        [], same effect — no CLI additions."""
        preset = {"prompt": "x", "loras": (LoraRef(ref="s/1"),)}
        out = resolve_effective_loras(preset, cli_lora=[], no_lora=False)
        assert out == (LoraRef(ref="s/1"),)

    def test_returns_tuple_not_list(self):
        """Downstream Iteration / build_mflux_cmd expects tuple."""
        out = resolve_effective_loras(
            {"prompt": "x"}, cli_lora=None, no_lora=False,
        )
        assert isinstance(out, tuple)


# ── prepend_trigger_words ──────────────────────────────────────────────


class TestPrependTriggerWords:
    def test_empty_loras_returns_prompt_unchanged(self):
        assert prepend_trigger_words("anime portrait", ()) == "anime portrait"

    def test_lora_without_trigger_no_change(self):
        loras = (LoraRef(ref="x/y", trigger=None),)
        out = prepend_trigger_words("anime portrait", loras)
        assert out == "anime portrait"

    def test_lora_with_trigger_already_in_prompt_no_change(self):
        """Case-insensitive substring check — trigger already present
        means no need to prepend."""
        loras = (LoraRef(ref="x/y", trigger="Animeo"),)
        out = prepend_trigger_words("Animeo anime portrait", loras)
        assert out == "Animeo anime portrait"

    def test_lora_with_trigger_case_insensitive(self):
        """User's prompt might use a different case than the LoRA's
        registered trigger — still counts as present."""
        loras = (LoraRef(ref="x/y", trigger="Animeo"),)
        out = prepend_trigger_words("animeo style portrait", loras)
        assert out == "animeo style portrait"

    def test_trigger_missing_gets_prepended(self):
        loras = (LoraRef(ref="x/y", trigger="Animeo"),)
        out = prepend_trigger_words("anime portrait", loras)
        assert out == "Animeo, anime portrait"

    def test_multi_lora_triggers_combined_when_all_missing(self):
        """Multiple triggers join with ", " and get prepended once."""
        loras = (
            LoraRef(ref="a/1", trigger="Pixar 3D"),
            LoraRef(ref="b/2", trigger="cinematic"),
        )
        out = prepend_trigger_words("portrait of person", loras)
        assert out == "Pixar 3D, cinematic, portrait of person"

    def test_multi_lora_partial_present_only_missing_prepended(self):
        loras = (
            LoraRef(ref="a/1", trigger="Pixar 3D"),
            LoraRef(ref="b/2", trigger="cinematic"),
        )
        out = prepend_trigger_words("Pixar 3D portrait", loras)
        # "Pixar 3D" already present → skipped. Only "cinematic"
        # prepended.
        assert out == "cinematic, Pixar 3D portrait"

    def test_duplicate_triggers_across_loras_deduped(self):
        """Two LoRAs sharing the same trigger word → prepend once, not
        twice. Defensive against author of a style TOML giving every
        LoRA the same trigger."""
        loras = (
            LoraRef(ref="a/1", trigger="anime"),
            LoraRef(ref="b/2", trigger="anime"),
            LoraRef(ref="c/3", trigger="anime"),
        )
        out = prepend_trigger_words("portrait", loras)
        assert out == "anime, portrait"

    def test_empty_string_trigger_treated_as_missing(self):
        """An empty/whitespace ``trigger`` field is equivalent to
        ``None`` — no prepending, no spurious empty trigger."""
        loras = (LoraRef(ref="x/y", trigger="   "),)
        out = prepend_trigger_words("anime portrait", loras)
        assert out == "anime portrait"

    def test_trigger_with_whitespace_around_stripped(self):
        loras = (LoraRef(ref="x/y", trigger="  Animeo  "),)
        out = prepend_trigger_words("anime portrait", loras)
        # Trigger stripped before comparison + prepending.
        assert out == "Animeo, anime portrait"


# ── End-to-end via build_iterations ────────────────────────────────────


def _build_args(**overrides) -> SimpleNamespace:
    """Reusable argparse-Namespace shape for build_iterations entry."""
    defaults = dict(
        image="/p.jpg",
        style=["anime"],
        custom_prompt=None,
        prompt_file=None,
        steps=None, quantize=None, guidance=None, strength=None,
        seed=42, preview=False, backend="flux",
        scope=None, width=None, height=None,
        output=None, output_dir=None,
        force=True, yes=True, no_open=True, dry_run=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        enhance=None, enhance_model=None, enhance_temperature=None,
        imgen_config_enhance={},
        lora=None,
        no_lora=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _flux_backend() -> Backend:
    return Backend(
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
        lora_compat_group="flux-1",
    )


def _build(*, fake_styles, tmp_path, **overrides):
    """Thin wrapper around build_iterations with the per-style stub."""
    from imgen.cmd_helpers import build_iterations
    fake_binary = tmp_path / "fake-mflux"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    base = dict(
        styles_list=["anime"],
        args=_build_args(),
        effective_custom_prompt=None,
        merged_defaults=DEFAULTS,
        be=_flux_backend(),
        binary=fake_binary,
        input_path=tmp_path / "p.jpg",
        width=1024, height=1024,
        explicit_output=None,
        run_dir=tmp_path / "out",
        seed=42,
    )
    base.update(overrides)
    # Stub the styles registry so we don't need a real BUILTIN_STYLES.
    import imgen.cmd_helpers as ch

    def fake_get_style(name: str) -> dict:
        return fake_styles[name]

    import imgen.styles as styles_mod
    original_get_style = styles_mod.get_style
    styles_mod.get_style = fake_get_style  # type: ignore[assignment]
    ch.get_style = fake_get_style  # type: ignore[assignment]
    try:
        return build_iterations(**base)
    finally:
        styles_mod.get_style = original_get_style  # type: ignore[assignment]
        ch.get_style = original_get_style  # type: ignore[assignment]


class TestBuildIterationsLoRA:
    def test_style_with_compatible_lora_lands_in_cmd(self, tmp_path):
        """Style ships a flux-1 LoRA; backend is flux-1; iteration's
        cmd contains --lora-paths + --lora-scales for it."""
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(
                    ref="strangerzonehf/Flux-Animeo-v1-LoRA",
                    weight=0.8,
                    compatible_with=("flux-1",),
                ),),
            },
        }
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        assert "--lora-paths" in its[0].cmd
        i = its[0].cmd.index("--lora-paths")
        assert its[0].cmd[i + 1] == "strangerzonehf/Flux-Animeo-v1-LoRA"

    def test_style_with_no_loras_produces_no_lora_argv(self, tmp_path):
        """Backward compat: a style without `loras` field works as v0.5."""
        styles = {"anime": {"prompt": "anime portrait"}}
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        assert "--lora-paths" not in its[0].cmd

    def test_cli_lora_appended_to_style_loras_in_cmd(self, tmp_path):
        """--lora REF appends to the style's stack — argv shows BOTH
        style and CLI refs in order."""
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(ref="style/lora", weight=0.8),),
            },
        }
        args = _build_args(
            lora=[LoraRef(ref="cli/lora", weight=0.5)],
        )
        its = _build(fake_styles=styles, tmp_path=tmp_path, args=args)
        i = its[0].cmd.index("--lora-paths")
        # Style first, CLI second.
        assert its[0].cmd[i + 1:i + 3] == ["style/lora", "cli/lora"]

    def test_no_lora_drops_style_loras_from_cmd(self, tmp_path):
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(ref="style/lora", weight=0.8),),
            },
        }
        args = _build_args(no_lora=True)
        its = _build(fake_styles=styles, tmp_path=tmp_path, args=args)
        assert "--lora-paths" not in its[0].cmd

    def test_trigger_word_prepended_to_iteration_prompt(self, tmp_path):
        """LoRA with trigger → iteration.prompt has the trigger
        prepended."""
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(
                    ref="strangerzonehf/Flux-Animeo-v1-LoRA",
                    weight=0.8,
                    trigger="Animeo",
                ),),
            },
        }
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        # Iteration's stored prompt starts with the trigger.
        assert its[0].prompt.startswith("Animeo, ")
        # And mflux argv carries the same trigger-prepended prompt.
        i = its[0].cmd.index("--prompt")
        assert its[0].cmd[i + 1].startswith("Animeo, ")

    def test_trigger_already_in_prompt_not_duplicated(self, tmp_path):
        """If the style's preset prompt already contains the trigger,
        no prepending."""
        styles = {
            "anime": {
                "prompt": "Animeo anime portrait",  # trigger present
                "loras": (LoraRef(
                    ref="x/y", weight=0.8, trigger="Animeo",
                ),),
            },
        }
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        # Counted ONCE in the final prompt, not duplicated.
        assert its[0].prompt.count("Animeo") == 1

    def test_incompatible_lora_skipped_with_warn(self, tmp_path, capsys):
        """A FLUX-2 LoRA on a FLUX-1 backend → argv excludes it +
        warn names the ref."""
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(
                    ref="flux2-only/x",
                    weight=0.8,
                    compatible_with=("flux-2",),
                ),),
            },
        }
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        assert "--lora-paths" not in its[0].cmd
        captured = capsys.readouterr()
        message = captured.out + captured.err
        assert "flux2-only/x" in message

    def test_incompatible_lora_trigger_not_prepended(self, tmp_path):
        """Trigger only fires for COMPATIBLE LoRAs. An incompatible
        LoRA's trigger isn't prepended (it wouldn't fire anyway —
        prepending would pollute the prompt for no benefit)."""
        styles = {
            "anime": {
                "prompt": "anime portrait",
                "loras": (LoraRef(
                    ref="flux2-only/x", weight=0.8,
                    compatible_with=("flux-2",),
                    trigger="FLUX2 only token",
                ),),
            },
        }
        its = _build(fake_styles=styles, tmp_path=tmp_path)
        # Trigger not in the prompt — the incompatible LoRA was filtered
        # out before trigger prepending.
        assert "FLUX2 only token" not in its[0].prompt
        assert its[0].prompt == "anime portrait"

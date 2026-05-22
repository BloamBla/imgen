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

    def test_no_lora_keeps_cli_loras_for_replay_path(self):
        """v0.6 carve-out: argparse enforces --lora + --no-lora mutex
        from CLI, but the replay path bypasses argparse and passes
        BOTH (cli_lora=stored_stack + no_lora=True) so the style's
        CURRENT built-in LoRAs are suppressed while the stored
        snapshot is reproduced. Architect-CRITICAL #1 fix from the
        v0.6 pre-tag review. Without this carve-out replay would
        silently drop the stored stack."""
        preset = {
            "prompt": "x",
            "loras": (LoraRef(ref="style/current_builtin"),),
        }
        cli = [LoraRef(ref="replay/stored")]
        out = resolve_effective_loras(preset, cli_lora=cli, no_lora=True)
        # Style's current built-in suppressed; replay's stored stack survives.
        assert out == (LoraRef(ref="replay/stored"),)

    def test_no_lora_with_empty_cli_returns_empty_tuple(self):
        """When no_lora=True and cli_lora is None / [], the carve-out
        falls back to the original v0.5 semantics: empty tuple. Models
        the user's --no-lora invocation (drop everything) and the
        v=3 history entry with loras=[] (text-only original run)."""
        preset = {
            "prompt": "x",
            "loras": (LoraRef(ref="style/lora", weight=0.8),),
        }
        assert resolve_effective_loras(preset, cli_lora=None, no_lora=True) == ()
        assert resolve_effective_loras(preset, cli_lora=[], no_lora=True) == ()

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

    # ── v0.6 python-reviewer IMP-2: word-boundary anchoring ────────

    def test_short_trigger_does_not_false_positive_on_substring(self):
        """v0.5 used unanchored ``trig_lower in prompt_lower`` — a
        3-character user trigger like ``"ani"`` would falsely match
        any prompt containing ``"animation"`` / ``"fanatical"`` /
        ``"sanitary"`` and silently skip prepending. v0.6 uses regex
        word-boundary (``\\b``) anchoring so short triggers behave
        correctly. Built-in triggers (Animeo / Pixar 3D / Ghibli style)
        are long enough that the v0.5 regression was latent, but the
        surface is public-via-user-styles."""
        loras = (LoraRef(ref="x/y", trigger="ani"),)
        # "ani" does not appear as a whole word in this prompt.
        out = prepend_trigger_words("animation portrait", loras)
        assert out == "ani, animation portrait"

    def test_short_trigger_matches_when_whole_word(self):
        """The flip side: when the trigger IS a whole word in the
        prompt, it counts as present and no prepending happens.
        Symmetric with the substring-rejection case above."""
        loras = (LoraRef(ref="x/y", trigger="ani"),)
        out = prepend_trigger_words("ani style portrait", loras)
        assert out == "ani style portrait"

    def test_multi_word_trigger_matches_only_at_word_boundaries(self):
        """``"Pixar 3D"`` must match in a prompt only when bracketed by
        word boundaries (start/end of string OR non-word chars). A
        prompt with ``"superPixar 3D"`` does NOT contain the trigger
        as a whole token."""
        loras = (LoraRef(ref="x/y", trigger="Pixar 3D"),)
        # No word-boundary before "Pixar" → counts as missing.
        out = prepend_trigger_words("superPixar 3D portrait", loras)
        assert out == "Pixar 3D, superPixar 3D portrait"
        # Word-boundary present → counts as present.
        out = prepend_trigger_words("Pixar 3D portrait", loras)
        assert out == "Pixar 3D portrait"

    def test_trigger_with_regex_metacharacters_safe(self):
        """If a user-defined LoRA trigger happens to contain regex
        metacharacters (``.``/``+``/``(``/...), the search must treat
        them as literal — ``re.escape`` handles this. Defensive against
        a future LoRA whose trigger is ``v1.0`` or ``a+b``."""
        loras = (LoraRef(ref="x/y", trigger="v1.0"),)
        # Without re.escape, "v1.0" would match "v100" via the regex
        # ".". With escape, the literal dot is required.
        out = prepend_trigger_words("portrait of v100 model", loras)
        assert out == "v1.0, portrait of v100 model"
        # Whole-word literal match → skip prepending.
        out = prepend_trigger_words("portrait of v1.0 model", loras)
        assert out == "portrait of v1.0 model"


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

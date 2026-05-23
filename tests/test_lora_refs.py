"""v0.6 Phase 1A — LoraRef dataclass + Style schema extension for the
``loras`` field on user style TOMLs.

Pure-function tests on :class:`imgen.styles.LoraRef` + the
:func:`parse_lora_refs` validator. The actual mflux argv wiring is
covered in Phase 1C tests; here we only pin the data layer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.styles import (
    LoraRef,
    UserStyleError,
    load_user_style_file,
    parse_lora_refs,
)


# ── LoraRef dataclass shape ────────────────────────────────────────────


class TestLoraRefDataclass:
    def test_minimal_construction_uses_defaults(self):
        ref = LoraRef(ref="org/model")
        assert ref.ref == "org/model"
        assert ref.weight == 1.0
        # Default compat group matches the project's primary backend.
        assert ref.compatible_with == ("flux-1",)
        assert ref.trigger is None

    def test_full_construction(self):
        ref = LoraRef(
            ref="strangerzonehf/Flux-Animeo-v1-LoRA",
            weight=0.8,
            compatible_with=("flux-1", "flux-1-dev"),
            trigger="Animeo",
        )
        assert ref.weight == 0.8
        assert "flux-1" in ref.compatible_with
        assert ref.trigger == "Animeo"

    def test_frozen(self):
        ref = LoraRef(ref="x/y")
        with pytest.raises((AttributeError, Exception)):
            ref.weight = 0.5  # type: ignore[misc]

    def test_slots(self):
        ref = LoraRef(ref="x/y")
        assert not hasattr(ref, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            ref.unknown_attr = 1  # type: ignore[attr-defined]


# ── parse_lora_refs validator ──────────────────────────────────────────


class TestParseLoraRefs:
    def test_empty_list_yields_empty_tuple(self):
        out = parse_lora_refs([], Path("x.toml"))
        assert out == ()
        assert isinstance(out, tuple)

    def test_single_entry_minimal_fields(self):
        raw = [{"ref": "alvarobartt/ghibli-characters-flux-lora"}]
        out = parse_lora_refs(raw, Path("x.toml"))
        assert len(out) == 1
        assert out[0].ref == "alvarobartt/ghibli-characters-flux-lora"
        # Defaults filled in.
        assert out[0].weight == 1.0
        assert out[0].compatible_with == ("flux-1",)
        assert out[0].trigger is None

    def test_single_entry_full_fields(self):
        raw = [{
            "ref": "strangerzonehf/Flux-Animeo-v1-LoRA",
            "weight": 0.8,
            "compatible_with": ["flux-1"],
            "trigger": "Animeo",
        }]
        out = parse_lora_refs(raw, Path("x.toml"))
        assert out[0].weight == 0.8
        assert out[0].compatible_with == ("flux-1",)
        assert out[0].trigger == "Animeo"

    def test_multiple_entries_preserve_order(self):
        raw = [
            {"ref": "first/lora", "weight": 0.8},
            {"ref": "second/lora", "weight": 0.4},
            {"ref": "third/lora", "weight": 0.2},
        ]
        out = parse_lora_refs(raw, Path("x.toml"))
        assert [r.ref for r in out] == [
            "first/lora", "second/lora", "third/lora",
        ]
        assert [r.weight for r in out] == [0.8, 0.4, 0.2]

    def test_missing_ref_field_raises(self):
        raw = [{"weight": 0.8}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*'ref'"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_empty_ref_string_rejected(self):
        raw = [{"ref": "   "}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*ref"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_ref_with_control_byte_rejected(self):
        """The ref string ends up in subprocess argv (--lora-paths)
        and in dry-run terminal display. Reject C0/DEL/C1 — symmetric
        with v0.5 scene_suffix defence."""
        raw = [{"ref": "evil\x1b[2J/model"}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*ref"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_oversized_ref_rejected(self):
        raw = [{"ref": "x" * 5000}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*ref"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_weight_out_of_range_rejected(self):
        for bad_weight in (-3.0, 2.5, 100):
            raw = [{"ref": "x/y", "weight": bad_weight}]
            with pytest.raises(UserStyleError, match=r"loras\[0\].*weight"):
                parse_lora_refs(raw, Path("x.toml"))

    def test_weight_bool_rejected(self):
        """Python's bool is a subclass of int — explicit reject so
        ``weight = true`` in TOML doesn't silently mean 1.0."""
        raw = [{"ref": "x/y", "weight": True}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*weight"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_weight_boundary_values_accepted(self):
        for boundary in (-2.0, 0.0, 1.0, 2.0):
            raw = [{"ref": "x/y", "weight": boundary}]
            out = parse_lora_refs(raw, Path("x.toml"))
            assert out[0].weight == boundary

    def test_empty_compatible_with_rejected(self):
        """At least one compat group must be declared — otherwise the
        LoRA matches nothing and silently disappears."""
        raw = [{"ref": "x/y", "compatible_with": []}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*compatible_with"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_compatible_with_non_string_element_rejected(self):
        raw = [{"ref": "x/y", "compatible_with": ["flux-1", 42]}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*compatible_with"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_compatible_with_control_byte_rejected(self):
        raw = [{"ref": "x/y", "compatible_with": ["flux\x1b-1"]}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*compatible_with"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_trigger_with_control_byte_rejected(self):
        raw = [{"ref": "x/y", "trigger": "Pixar\x1b 3D"}]
        with pytest.raises(UserStyleError, match=r"loras\[0\].*trigger"):
            parse_lora_refs(raw, Path("x.toml"))

    def test_compatible_with_list_converted_to_tuple(self):
        """Schema accepts list (TOML's native shape); LoraRef wants
        tuple (frozen-dataclass-friendly)."""
        raw = [{"ref": "x/y", "compatible_with": ["flux-1", "flux-1-dev"]}]
        out = parse_lora_refs(raw, Path("x.toml"))
        assert out[0].compatible_with == ("flux-1", "flux-1-dev")
        assert isinstance(out[0].compatible_with, tuple)


# ── End-to-end via load_user_style_file ───────────────────────────────


class TestLoraInUserStyleTOML:
    def test_toml_with_inline_loras_array_parses(self, tmp_path):
        """Inline ``loras = [...]`` array of dicts in TOML."""
        toml = tmp_path / "anime.toml"
        toml.write_text(
            'prompt = "Restyle this person as anime..."\n'
            'guidance = 4.0\n'
            'strength = 0.6\n'
            'loras = [\n'
            '  { ref = "strangerzonehf/Flux-Animeo-v1-LoRA", '
            'weight = 0.8, trigger = "Animeo" },\n'
            ']\n'
        )
        preset = load_user_style_file(toml)
        assert "loras" in preset
        assert len(preset["loras"]) == 1
        assert preset["loras"][0].ref == "strangerzonehf/Flux-Animeo-v1-LoRA"
        assert preset["loras"][0].weight == 0.8
        assert preset["loras"][0].trigger == "Animeo"

    def test_toml_with_double_bracket_loras_table_parses(self, tmp_path):
        """``[[loras]]`` array-of-tables syntax (TOML's other shape
        for the same data structure)."""
        toml = tmp_path / "anime.toml"
        toml.write_text(
            'prompt = "Restyle this person as anime"\n'
            '\n'
            '[[loras]]\n'
            'ref = "strangerzonehf/Flux-Animeo-v1-LoRA"\n'
            'weight = 0.8\n'
            '\n'
            '[[loras]]\n'
            'ref = "Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style"\n'
            'weight = 0.4\n'
        )
        preset = load_user_style_file(toml)
        assert len(preset["loras"]) == 2
        assert preset["loras"][0].weight == 0.8
        assert preset["loras"][1].weight == 0.4

    def test_toml_without_loras_yields_empty_field(self, tmp_path):
        """Backward compat: a style TOML that doesn't mention loras
        defaults the field to an empty tuple — identical effective
        behaviour to a v0.5 preset (no LoRAs applied).

        v0.6.2: Style dataclass always exposes a ``loras`` attribute
        (default ``()``); the prior "absent key" test was an artefact
        of the dict shape.
        """
        toml = tmp_path / "noir.toml"
        toml.write_text(
            'prompt = "Film noir style"\n'
            'guidance = 4.5\n'
        )
        preset = load_user_style_file(toml)
        assert preset.loras == ()

    def test_toml_with_invalid_loras_raises(self, tmp_path):
        toml = tmp_path / "bad.toml"
        toml.write_text(
            'prompt = "x"\n'
            'loras = [{ weight = 0.8 }]\n'  # missing 'ref'
        )
        with pytest.raises(UserStyleError, match=r"loras\[0\].*'ref'"):
            load_user_style_file(toml)

    def test_toml_with_non_list_loras_raises(self, tmp_path):
        toml = tmp_path / "bad.toml"
        toml.write_text(
            'prompt = "x"\n'
            'loras = "not a list"\n'
        )
        with pytest.raises(UserStyleError, match="loras"):
            load_user_style_file(toml)

"""v0.7.0 (architect §C): `--lora` accepts comma-list AND repeated flag.

Mirrors the `--style anime,ghibli,pixar` shape shipped in v0.2.3+. Each
comma-split element flows through the existing per-element validation
in :func:`_lora_ref_arg` (control bytes, oversized refs, flag-shape
rejection, weight range). The new public helper :func:`_lora_refs_arg`
returns ``list[LoraRef]``; argparse's ``action='append'`` collects
calls into ``list[list[LoraRef]]`` which
:func:`cmd_helpers.resolve_effective_loras` flattens at use-site.

Table-driven on the §C matrix.
"""
from __future__ import annotations

import argparse

import pytest

from imgen.cmd_helpers import _flatten_cli_lora, resolve_effective_loras
from imgen.parser import _lora_ref_arg, _lora_refs_arg
from imgen.styles import LoraRef, Style


class TestLoraRefsArg:
    """Direct test of the new comma-splitting validator."""

    def test_single_ref_no_weight(self):
        """CLI default compatible_with widened to ('flux-1', 'flux-dev')
        in v0.7.0 alongside the FLUX.1-dev t2i backend; see
        test_flux_dev_backend.py for the dedicated coverage."""
        out = _lora_refs_arg("a/b")
        assert out == [
            LoraRef(ref="a/b", weight=1.0, compatible_with=("flux-1", "flux-dev")),
        ]

    def test_single_ref_with_weight(self):
        out = _lora_refs_arg("a/b:0.7")
        assert len(out) == 1
        assert out[0].ref == "a/b"
        assert out[0].weight == 0.7

    def test_comma_two_refs_no_weights(self):
        out = _lora_refs_arg("a/b,c/d")
        assert len(out) == 2
        assert [r.ref for r in out] == ["a/b", "c/d"]
        assert all(r.weight == 1.0 for r in out)

    def test_comma_with_mixed_weights(self):
        out = _lora_refs_arg("a/b:0.7,c/d")
        assert len(out) == 2
        assert out[0].ref == "a/b" and out[0].weight == 0.7
        assert out[1].ref == "c/d" and out[1].weight == 1.0

    def test_comma_three_refs(self):
        out = _lora_refs_arg("a/b,c/d:0.5,e/f")
        assert [r.ref for r in out] == ["a/b", "c/d", "e/f"]
        assert [r.weight for r in out] == [1.0, 0.5, 1.0]

    def test_whitespace_around_comma_elements_stripped(self):
        out = _lora_refs_arg(" a/b , c/d ")
        assert len(out) == 2
        assert out[0].ref == "a/b"
        assert out[1].ref == "c/d"

    def test_empty_value_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_refs_arg("")

    def test_empty_comma_element_rejected(self):
        """``a,,b`` is a typo not a stack of 3 refs."""
        with pytest.raises(argparse.ArgumentTypeError, match="empty comma-element"):
            _lora_refs_arg("a/b,,c/d")

    def test_trailing_comma_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty comma-element"):
            _lora_refs_arg("a/b,")

    def test_leading_comma_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty comma-element"):
            _lora_refs_arg(",a/b")

    def test_flag_shaped_element_rejected(self):
        """Per-element guards still fire — the comma-split layer doesn't
        bypass _lora_ref_arg's argv-injection rejection."""
        with pytest.raises(
            argparse.ArgumentTypeError, match="must not start with '-'"
        ):
            _lora_refs_arg("a/b,--config")

    def test_absolute_path_comma_split_no_weight(self):
        """Absolute paths skip the colon-weight split (v0.6 python IMP-1).
        With comma-split layered on top: each comma-element is a full
        path, the colon inside the path stays intact, weight defaults
        to 1.0."""
        out = _lora_refs_arg("/abs/path/lora.safetensors,a/b")
        assert len(out) == 2
        assert out[0].ref == "/abs/path/lora.safetensors"
        assert out[0].weight == 1.0
        assert out[1].ref == "a/b"

    def test_weight_out_of_range_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="out of range"):
            _lora_refs_arg("a/b:5.0,c/d")

    def test_backward_compat_lora_ref_arg_still_single_ref(self):
        """The single-ref helper survives the comma-split addition.
        Existing callers / tests of _lora_ref_arg keep their contract."""
        out = _lora_ref_arg("a/b:0.7")
        assert isinstance(out, LoraRef)
        assert out.ref == "a/b"
        assert out.weight == 0.7


class TestFlattenCliLora:
    """Normalisation helper between argparse output and
    resolve_effective_loras' precedence logic."""

    def test_none_returns_empty_tuple(self):
        assert _flatten_cli_lora(None) == ()

    def test_empty_list_returns_empty_tuple(self):
        assert _flatten_cli_lora([]) == ()

    def test_list_of_lists_flattens(self):
        """v0.7.0 CLI shape: argparse's action='append' over
        _lora_refs_arg collects list-of-lists."""
        a = LoraRef(ref="a/b", weight=1.0)
        b = LoraRef(ref="c/d", weight=0.5)
        c = LoraRef(ref="e/f", weight=1.0)
        out = _flatten_cli_lora([[a, b], [c]])
        assert out == (a, b, c)

    def test_flat_list_pass_through(self):
        """Legacy / replay shape: list[LoraRef] flat."""
        a = LoraRef(ref="a/b", weight=1.0)
        b = LoraRef(ref="c/d", weight=0.5)
        out = _flatten_cli_lora([a, b])
        assert out == (a, b)

    def test_mixed_shape_handled(self):
        """Defence-in-depth: a future caller that mixes shapes
        doesn't blow up."""
        a = LoraRef(ref="a/b", weight=1.0)
        b = LoraRef(ref="c/d", weight=0.5)
        c = LoraRef(ref="e/f", weight=1.0)
        out = _flatten_cli_lora([a, [b, c]])
        assert out == (a, b, c)


class TestResolveEffectiveLorasWithCommaSplit:
    """resolve_effective_loras handles both legacy + v0.7.0 cli_lora
    shapes via the _flatten_cli_lora normalisation."""

    def test_v070_list_of_lists_appended_to_style_stack(self):
        style = LoraRef(ref="style/lora", weight=0.8)
        cli_a = LoraRef(ref="cli/a", weight=1.0)
        cli_b = LoraRef(ref="cli/b", weight=0.5)
        preset = Style(loras=(style,))
        # v0.7.0 shape: --lora cli/a,cli/b → argparse collects [[cli_a, cli_b]]
        out = resolve_effective_loras(preset, [[cli_a, cli_b]], no_lora=False)
        assert out == (style, cli_a, cli_b)

    def test_legacy_flat_list_still_works(self):
        """replay_entry passes flat list of LoraRefs reconstructed from
        history; the helper accepts that shape unchanged."""
        style = LoraRef(ref="style/lora", weight=0.8)
        stored_a = LoraRef(ref="stored/a", weight=1.0)
        stored_b = LoraRef(ref="stored/b", weight=0.5)
        preset = Style(loras=(style,))
        # no_lora=True is the replay path; cli_lora carries the stored stack.
        out = resolve_effective_loras(
            preset, [stored_a, stored_b], no_lora=True,
        )
        # no_lora=True drops style LoRAs, keeps CLI/stored ones.
        assert out == (stored_a, stored_b)

    def test_repeated_plus_comma_split_combined(self):
        """``--lora a,b --lora c`` → argparse collects [[a, b], [c]]."""
        a = LoraRef(ref="a/b", weight=1.0)
        b = LoraRef(ref="c/d", weight=0.5)
        c = LoraRef(ref="e/f", weight=1.0)
        preset = Style(loras=())
        out = resolve_effective_loras(preset, [[a, b], [c]], no_lora=False)
        assert out == (a, b, c)

    def test_no_lora_with_comma_split_keeps_cli_only(self):
        """The replay-style carve-out (no_lora=True + non-empty cli_lora)
        still works when cli_lora is the list-of-lists shape."""
        style = LoraRef(ref="style/lora", weight=0.8)
        cli_a = LoraRef(ref="cli/a", weight=1.0)
        preset = Style(loras=(style,))
        out = resolve_effective_loras(preset, [[cli_a]], no_lora=True)
        assert out == (cli_a,)

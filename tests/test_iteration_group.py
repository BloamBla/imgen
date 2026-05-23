"""v0.7.0 (architect FL-1 cash-in): IterationGroup Protocol + new
DrawIterationGroup sibling. Lock both concrete shapes against the
Protocol contract, and exercise the Protocol-typed
apply_enhance_results_to_groups helper against a mixed list of i2i +
t2i groups (mirroring what an `imgen draw` + `imgen batch` script
sequence would produce).

v0.7.1: the temporary `apply_enhance_results_to_per_input` alias from
v0.7.0 has been removed. Only `apply_enhance_results_to_groups`
survives. Any external programmatic caller still importing the old
name will get a clean ImportError instead of a silent type mismatch.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from imgen.cmd_helpers import apply_enhance_results_to_groups
from imgen.enhance import EnhanceResult
from imgen.runs import (
    DrawIterationGroup,
    Iteration,
    IterationGroup,
    PerInputBatch,
)


def _make_iter(prompt: str, style: str = "anime") -> Iteration:
    return Iteration(
        style_name=style,
        prompt=prompt,
        negative="",
        final_steps=20,
        final_quantize=4,
        final_guidance=4.0,
        final_strength=0.6,
        output_path=Path(f"/tmp/out-{style}.png"),
        cmd=["fake-mflux", "--prompt", prompt],
    )


def _make_enh_result(original: str, enhanced: str | None = None) -> EnhanceResult:
    if enhanced is None:
        return EnhanceResult(
            final_prompt=original,
            original_prompt=original,
            was_enhanced=False,
            fallback_reason="user_opt_out",
            was_truncated=False,
            raw_llm_output=None,
        )
    return EnhanceResult(
        final_prompt=enhanced,
        original_prompt=original,
        was_enhanced=True,
        fallback_reason=None,
        was_truncated=False,
        raw_llm_output=enhanced,
    )


class TestIterationGroupProtocol:
    """Both concrete shapes satisfy the Protocol; mismatched shapes
    don't pass isinstance."""

    def test_per_input_batch_is_iteration_group(self):
        pib = PerInputBatch(
            input_path=Path("/tmp/photo.jpg"),
            mflux_input=Path("/tmp/photo.jpg"),
            width=1024,
            height=1024,
            iters=(_make_iter("p1"),),
        )
        assert isinstance(pib, IterationGroup)

    def test_draw_iteration_group_is_iteration_group(self):
        dig = DrawIterationGroup(
            width=1024,
            height=1024,
            iters=(_make_iter("a samurai"),),
        )
        assert isinstance(dig, IterationGroup)

    def test_non_group_class_not_iteration_group(self):
        """An arbitrary frozen+slots dataclass without the
        width/height/iters trio fails the Protocol check."""

        class Other:
            def __init__(self):
                self.foo = 1

        assert not isinstance(Other(), IterationGroup)


class TestDrawIterationGroup:
    def test_constructs_with_required_fields(self):
        it = _make_iter("a samurai")
        g = DrawIterationGroup(width=1024, height=1024, iters=(it,))
        assert g.width == 1024
        assert g.height == 1024
        assert g.iters == (it,)

    def test_is_frozen(self):
        g = DrawIterationGroup(width=1024, height=1024, iters=())
        with pytest.raises(FrozenInstanceError):
            g.width = 2048  # type: ignore[misc]

    def test_iters_is_tuple_not_list(self):
        """Mirror PerInputBatch + Iteration.loras precedent: contained
        sequence is a tuple, eliminating the 'mutable element in
        frozen field' gotcha."""
        g = DrawIterationGroup(
            width=1024, height=1024, iters=(_make_iter("a"),),
        )
        assert isinstance(g.iters, tuple)

    def test_hash_disabled(self):
        """Iteration is __hash__ = None (cmd: list[str]); any container
        holding Iterations inherits unhashability."""
        g = DrawIterationGroup(width=1024, height=1024, iters=())
        with pytest.raises(TypeError):
            hash(g)


class TestApplyEnhanceResultsToGroups:
    """v0.7.0 (architect FL-2): the helper sheds its i2i-flavoured
    name. v0.7.1 dropped the temporary backward-compat alias —
    `apply_enhance_results_to_per_input` no longer exists at any
    import path."""

    def test_old_alias_removed(self):
        """v0.7.1 lock-in: importing the v0.6.4–v0.7.0 name raises
        ImportError. Loud-fail beats silent type mismatch on a future
        contributor who copy-pasted from a pre-v0.7.1 doc."""
        import imgen.cmd_helpers as cmd_helpers
        assert not hasattr(cmd_helpers, "apply_enhance_results_to_per_input")

    def test_per_input_batch_round_trip(self):
        """The i2i path through the new helper name. Same behaviour as
        v0.6.5's apply_enhance_results_to_per_input."""
        pib = PerInputBatch(
            input_path=Path("/tmp/photo.jpg"),
            mflux_input=Path("/tmp/photo.jpg"),
            width=1024,
            height=1024,
            iters=(_make_iter("p1"), _make_iter("p2", "ghibli")),
        )
        results = [
            _make_enh_result("p1", "ENH p1"),
            _make_enh_result("p2", "ENH p2"),
        ]
        out = apply_enhance_results_to_groups([pib], results)
        assert len(out) == 1
        assert isinstance(out[0], PerInputBatch)
        assert out[0].iters[0].prompt == "ENH p1"
        assert out[0].iters[1].prompt == "ENH p2"
        # Path + dim fields preserved via dataclasses.replace.
        assert out[0].input_path == pib.input_path
        assert out[0].width == pib.width

    def test_draw_iteration_group_round_trip(self):
        """The new t2i path. Single iter, single result, replace
        carries width/height through."""
        dig = DrawIterationGroup(
            width=1024,
            height=1024,
            iters=(_make_iter("a samurai", "draw"),),
        )
        results = [_make_enh_result("a samurai", "ENH a samurai")]
        out = apply_enhance_results_to_groups([dig], results)
        assert len(out) == 1
        assert isinstance(out[0], DrawIterationGroup)
        assert out[0].iters[0].prompt == "ENH a samurai"
        assert out[0].width == 1024

    def test_mixed_groups_round_trip(self):
        """The Protocol-typed helper accepts both concrete types in
        one list. Real scripts may interleave (cmd_draw + cmd_batch
        sequence) — locks the cross-shape behaviour even though no
        current orchestrator emits this shape today."""
        pib = PerInputBatch(
            input_path=Path("/tmp/photo.jpg"),
            mflux_input=Path("/tmp/photo.jpg"),
            width=1024,
            height=1024,
            iters=(_make_iter("p1"),),
        )
        dig = DrawIterationGroup(
            width=1024, height=1024,
            iters=(_make_iter("draw1"), _make_iter("draw2")),
        )
        results = [
            _make_enh_result("p1", "ENH p1"),
            _make_enh_result("draw1", "ENH draw1"),
            _make_enh_result("draw2", "ENH draw2"),
        ]
        out = apply_enhance_results_to_groups([pib, dig], results)
        assert len(out) == 2
        assert isinstance(out[0], PerInputBatch)
        assert isinstance(out[1], DrawIterationGroup)
        assert out[0].iters[0].prompt == "ENH p1"
        assert out[1].iters[0].prompt == "ENH draw1"
        assert out[1].iters[1].prompt == "ENH draw2"

    def test_count_mismatch_raises(self):
        dig = DrawIterationGroup(
            width=1024, height=1024,
            iters=(_make_iter("a"), _make_iter("b")),
        )
        with pytest.raises(ValueError, match="enhance-result count mismatch"):
            apply_enhance_results_to_groups([dig], [_make_enh_result("a")])

    def test_empty_groups_pass_through(self):
        assert apply_enhance_results_to_groups([], []) == []

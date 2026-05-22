"""Tests for the `Iteration` frozen dataclass in runs.py.

v0.2.4 replaces the stringly-typed dict (9 keys) used inside
`cmd_generate` to pass per-style parameters between the build-iterations
pre-pass and the loop with a typed dataclass.

Frozen because the v0.2.3 pre-build-all-iterations design assumes
iteration values cannot mutate mid-loop; slots because we want the
field set to be a closed contract (typos like `it.style_anme` should be
AttributeError, not silently create a new attribute).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.runs import Iteration


def _make_iteration(**overrides) -> Iteration:
    """Helper: build an Iteration with sane defaults; override per test."""
    defaults: dict = dict(
        style_name="anime",
        prompt="cinematic anime portrait of this person",
        negative="bad anatomy, blurry",
        final_steps=14,
        final_quantize=8,
        final_guidance=2.5,
        final_strength=0.6,
        output_path=Path("/tmp/out.png"),
        cmd=["/venv/bin/mflux-generate", "--prompt", "..."],
    )
    defaults.update(overrides)
    return Iteration(**defaults)


def test_iteration_constructs_with_all_fields():
    it = _make_iteration()
    assert it.style_name == "anime"
    assert it.final_steps == 14
    assert it.final_quantize == 8
    assert it.final_guidance == pytest.approx(2.5)
    assert it.final_strength == pytest.approx(0.6)
    assert it.output_path == Path("/tmp/out.png")
    assert it.cmd[0] == "/venv/bin/mflux-generate"


def test_iteration_is_frozen():
    """v0.2.3 batch design pre-builds all iterations then loops — any
    field mutation mid-loop would be a bug. Catch it at write-time."""
    it = _make_iteration()
    with pytest.raises((AttributeError, TypeError)):  # FrozenInstanceError → TypeError on 3.12
        it.style_name = "ghibli"  # type: ignore[misc]


def test_iteration_has_slots_no_dict():
    """slots=True closes the field set — typos must AttributeError, not
    silently land on __dict__ as a new attribute."""
    it = _make_iteration()
    assert not hasattr(it, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        it.nonexistent_typo_field = "x"  # type: ignore[attr-defined]


def test_iteration_equality_by_fields():
    """Two iterations with identical fields compare equal (default
    dataclass __eq__). Useful for test assertions on _build_iterations
    output."""
    a = _make_iteration()
    b = _make_iteration()
    assert a == b


def test_iteration_inequality_on_field_difference():
    a = _make_iteration(style_name="anime")
    b = _make_iteration(style_name="ghibli")
    assert a != b


def test_iteration_repr_includes_class_name():
    """Default dataclass __repr__ is fine — make sure we get it (helps
    pytest failure diagnostics)."""
    it = _make_iteration(style_name="pixar")
    assert "Iteration" in repr(it)
    assert "pixar" in repr(it)


def test_iteration_field_order_matches_spec():
    """The architect's v0.2.4 spec fixes the field order so callers can
    rely on positional construction in tests (and future code that
    might build via *args). Lock it.

    Order: style_name, prompt, negative, final_steps, final_quantize,
    final_guidance, final_strength, output_path, cmd.
    """
    import dataclasses
    fields = [f.name for f in dataclasses.fields(Iteration)]
    assert fields == [
        "style_name",
        "prompt",
        "negative",
        "final_steps",
        "final_quantize",
        "final_guidance",
        "final_strength",
        "output_path",
        "cmd",
    ]

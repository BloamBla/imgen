"""Tests for the `BatchContext` frozen dataclass in runs.py.

v0.2.5 introduces BatchContext to bundle the 9 batch-invariant kwargs
of `_run_one_iteration` (backend, seed, width, height, input_path,
effective_custom_prompt, args, batch_id, env) into one frozen value.
`_run_one_iteration`'s signature drops from 16 args to 8; v0.3.0's
nested N×M loop in commands/batch.py becomes legible.

Same disciplines as Iteration: frozen so per-iteration mutation can't
happen by accident; slots so a typo on field access raises instead of
silently landing on __dict__.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.runs import BatchContext


def _make_ctx(**overrides) -> BatchContext:
    defaults: dict = dict(
        backend="flux",
        seed=42,
        width=1024,
        height=1024,
        input_path=Path("/tmp/in.jpg"),
        effective_custom_prompt=None,
        args=SimpleNamespace(scope=None, preview=False),
        batch_id=None,
        env={"PATH": "/usr/bin"},
    )
    defaults.update(overrides)
    return BatchContext(**defaults)


def test_batch_context_constructs_with_all_fields():
    ctx = _make_ctx()
    assert ctx.backend == "flux"
    assert ctx.seed == 42
    assert ctx.width == 1024 and ctx.height == 1024
    assert ctx.input_path == Path("/tmp/in.jpg")
    assert ctx.effective_custom_prompt is None
    assert ctx.batch_id is None
    assert ctx.env == {"PATH": "/usr/bin"}


def test_batch_context_is_frozen():
    """v0.3.0 batch.py threads ctx through nested loops — accidental
    mutation would change semantics across iterations. Catch at write."""
    ctx = _make_ctx()
    with pytest.raises((AttributeError, TypeError)):
        ctx.backend = "qwen"  # type: ignore[misc]
    # Belt-and-braces: even if frozen=True were dropped, slots would
    # still reject mutation of the named field — verify value didn't
    # silently change.
    assert ctx.backend == "flux"


def test_batch_context_has_slots_no_dict():
    """slots=True prevents adding new attributes via typo."""
    ctx = _make_ctx()
    assert not hasattr(ctx, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        ctx.nonexistent_field = "x"  # type: ignore[attr-defined]
    assert not hasattr(ctx, "nonexistent_field")


def test_batch_context_field_order_matches_spec():
    """Architect's v0.2.5 backlog (IMP-3) fixed the field order. Lock
    it so positional construction in future tests stays stable.

    Order matches the 9 batch-invariant args of v0.2.4's
    _run_one_iteration."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(BatchContext)]
    assert fields == [
        "backend",
        "seed",
        "width",
        "height",
        "input_path",
        "effective_custom_prompt",
        "args",
        "batch_id",
        "env",
    ]


def test_batch_context_equality_by_fields():
    a = _make_ctx()
    b = _make_ctx()
    assert a == b


def test_batch_context_with_custom_prompt():
    ctx = _make_ctx(effective_custom_prompt="my prompt")
    assert ctx.effective_custom_prompt == "my prompt"


def test_batch_context_with_batch_id():
    ctx = _make_ctx(batch_id="abc123def456")
    assert ctx.batch_id == "abc123def456"


def test_batch_context_is_explicitly_unhashable():
    """frozen=True dataclasses auto-generate __hash__ which would call
    hash() on each field. `env: dict` and `args: Namespace` aren't
    hashable, so the default __hash__ would TypeError on first use.

    We opt out with `__hash__ = None` so callers see the type error
    AT THE SET/DICT INSERTION site, not deep inside the dataclass
    machinery. Lock the contract. (v0.2.5 review IMP-1)"""
    ctx = _make_ctx()
    with pytest.raises(TypeError):
        hash(ctx)
    # Equality still works — locked separately by
    # test_batch_context_equality_by_fields.

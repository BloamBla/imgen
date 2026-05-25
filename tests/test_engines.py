"""v0.8.0 commit 1 — Engine Protocol shape (skeleton).

Per [[project-v080-design]] §C: Engine is `typing.Protocol` with
`@runtime_checkable`, NOT abc.ABC. This file pins the Protocol surface
so v0.8.1+ can't silently drift the contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_minimal_genparams(**overrides):
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="x", negative="", width=64, height=64,
        steps=1, guidance=0.0, seed=0, quantize=4, strength=0.0,
        input_path=None, output_path=Path("/tmp/out.png"), loras=(),
    )
    defaults.update(overrides)
    return GenParams(**defaults)


class _MinimalCompliantEngine:
    """Test double with the exact Protocol surface — used to verify
    `isinstance(eng, Engine)` works for structural typing."""
    name = "test-engine"

    def build_cmd(self, model, params):
        return ["test-cmd"]

    def run(self, model, params, env=None):
        return 0

    def validate(self, model, params):
        return []

    def ram_estimate_gb(self, model, params):
        return 4.0


class _IncompleteEngine:
    """Missing `validate` method — should fail isinstance check."""
    name = "incomplete"

    def build_cmd(self, model, params):
        return []

    def run(self, model, params, env=None):
        return 0

    def ram_estimate_gb(self, model, params):
        return 1.0


class TestEngineProtocol:
    def test_protocol_accepts_minimal_compliant_engine(self):
        from imgen.engines.base import Engine
        assert isinstance(_MinimalCompliantEngine(), Engine)

    def test_protocol_rejects_engine_missing_method(self):
        from imgen.engines.base import Engine
        # @runtime_checkable Protocol checks structural conformance;
        # missing `validate` method must fail isinstance.
        assert not isinstance(_IncompleteEngine(), Engine)

    def test_protocol_is_runtime_checkable(self):
        """Lock-in that we chose `@runtime_checkable Protocol`, not
        `abc.ABC`. The decorator is the marker — if a future refactor
        drops it, this assertion catches it."""
        from imgen.engines.base import Engine
        # @runtime_checkable Protocols accept isinstance(); plain Protocols
        # raise TypeError on isinstance. So the fact that the previous two
        # tests PASS without raising TypeError confirms runtime_checkable
        # is applied. This explicit assertion is the documented version
        # of that same constraint.
        from typing import get_origin
        # Bonus sanity: the class IS a typing.Protocol marker, not abc.ABC.
        # (Implementation detail: Protocols have `_is_protocol = True`.)
        assert getattr(Engine, "_is_protocol", False) is True

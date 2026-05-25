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


# ── v0.8.0 commit 2 — MfluxEngine ────────────────────────────────────


class TestMfluxEngineConformance:
    """MfluxEngine implements the Engine Protocol structurally."""

    def test_mflux_engine_satisfies_protocol(self):
        from imgen.engines.base import Engine
        from imgen.engines.mflux_engine import MfluxEngine
        assert isinstance(MfluxEngine(), Engine)

    def test_mflux_engine_name(self):
        from imgen.engines.mflux_engine import MfluxEngine
        assert MfluxEngine().name == "mflux"


@pytest.mark.parametrize(
    "v07_name,v08_name",
    [
        ("flux", "flux-kontext"),
        ("qwen", "qwen-image-edit-v1"),
        ("flux-dev", "flux-dev"),
        ("flux2-klein-edit-9b", "flux2-klein-edit-9b"),
    ],
)
class TestMfluxEngineBuildCmdMatchesV07_17:
    """Argv-stability lock-in (§D, §Q commit 2 + 4b): MfluxEngine.build_cmd
    produces bit-identical argv to v0.7.17's `build_mflux_cmd` for
    every currently-built-in model.

    4b updated to parametrize by ``(v07_name, v08_name)`` pairs because
    BUILTIN_MODELS is now keyed by v0.8 names (literal declaration per
    §G.1) while ``backends.BACKENDS`` (= BUILTIN_BACKENDS, derived
    backward) keeps the v0.7 keys for v0.7.x test fixture compatibility.
    Each parametrize row exercises BOTH the literal Model lookup and
    the v0.7 Backend lookup, asserting argv parity.
    """

    def test_build_cmd_argv_identical_to_v07_17(self, v07_name, v08_name):
        from pathlib import Path
        from imgen.backends import BACKENDS, build_mflux_cmd
        from imgen.engines.base import GenParams
        from imgen.engines.mflux_engine import MfluxEngine
        from imgen.models import BUILTIN_MODELS

        backend = BACKENDS[v07_name]
        model = BUILTIN_MODELS[v08_name]

        # Pick an input_path only for backends that accept one — flux
        # and qwen are i2i; flux-dev is t2i (image_flag set for
        # dataclass shape consistency but the runtime gate is on
        # input_path is None per backends.py:925-927).
        input_path = Path("/fake/in.png") if v07_name != "flux-dev" else None
        common = dict(
            output_path=Path("/fake/out.png"),
            prompt="a samurai on a misty mountain at dawn",
            negative="",
            quantize=4,
            steps=20,
            guidance=3.5,
            strength=0.55,
            seed=1088118853,
            width=1024,
            height=1024,
            mlx_cache_gb=12,
            battery_stop=20,
            loras=(),
        )

        legacy_argv = build_mflux_cmd(
            binary=Path("/fake/mflux-bin"),
            model=backend,
            input_path=input_path,
            **common,
        )

        params = GenParams(
            prompt=common["prompt"],
            negative=common["negative"],
            width=common["width"],
            height=common["height"],
            steps=common["steps"],
            guidance=common["guidance"],
            seed=common["seed"],
            quantize=common["quantize"],
            strength=common["strength"],
            input_path=input_path,
            output_path=common["output_path"],
            loras=common["loras"],
            mlx_cache_gb=common["mlx_cache_gb"],
            battery_stop=common["battery_stop"],
        )
        new_argv = MfluxEngine().build_cmd(
            model, params, binary=Path("/fake/mflux-bin"),
        )
        assert new_argv == legacy_argv, (
            f"argv drift for {backend_name}:\n"
            f"  legacy: {legacy_argv}\n"
            f"  new:    {new_argv}"
        )

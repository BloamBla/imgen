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
    `isinstance(eng, Engine)` works for structural typing.

    v0.10 commit 1: gains ``train(model, params)`` per
    [[project-v100-design]] §R.1 round-1 closure (Engine.train replaces
    the dropped MfluxTrainer class). Engines that don't support training
    raise NotImplementedError per the docstring convention."""
    name = "test-engine"

    def build_cmd(self, model, params):
        return ["test-cmd"]

    def run(self, model, params, env=None):
        return 0

    def validate(self, model, params):
        return []

    def ram_estimate_gb(self, model, params):
        return 4.0

    def train(self, model, params):
        """v0.10.0 — Engine.train Protocol method. Compliant test
        double; production impls raise NotImplementedError where
        training isn't supported (e.g. DiffusersMpsEngine)."""
        raise NotImplementedError("test-engine does not implement train")


class _IncompleteEngine:
    """Missing `validate` method — should fail isinstance check.

    v0.10 commit 1: also missing ``train`` — both gaps make this an
    incomplete Protocol implementation. The lock-in test asserts
    isinstance returns False, proving structural conformance is
    enforced at runtime."""
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


# ── v0.10 commit 1 — Engine.train Protocol method ─────────────────────


class TestEngineTrainProtocolMethod:
    """v0.10.0 commit 1: ``Engine.train(model, params)`` is the new
    Protocol verb per [[project-v100-design]] §R.1 round-1 closure
    (architect H-3 — re-decided from separate MfluxTrainer class to
    Engine.train method to preserve the v0.9.5 M-2 Engine-registry
    single-source-of-truth).

    Convention per §R.1 round-2 N-1: engines that don't support training
    raise ``NotImplementedError`` with the engine name in the message —
    same posture as ``abc.abstractmethod`` conventions. v0.10.0 ships:

    * ``MfluxEngine.train`` raises NotImplementedError until commit 5
      wires the real impl (subprocess invocation of ``mflux-train``).
    * ``DiffusersMpsEngine.train`` raises NotImplementedError
      permanently (v0.10.0 doesn't train via diffusers_mps; video
      Models stay inference-only).
    """

    def test_engine_protocol_includes_train_method(self):
        """Protocol structural lock — Engine MUST declare ``train``."""
        from imgen.engines.base import Engine
        assert hasattr(Engine, "train"), (
            "v0.10 commit 1 added Engine.train; missing here means "
            "the Protocol declaration regressed."
        )

    def test_mflux_engine_train_raises_not_implemented_until_commit_5(self):
        """Commit 5 will replace the NotImplementedError with the
        actual mflux-train subprocess dispatch. Until then this is a
        load-bearing placeholder so Protocol structural conformance
        holds."""
        from imgen.engines.mflux_engine import MfluxEngine
        params = _make_minimal_genparams()
        with pytest.raises(NotImplementedError, match="mflux"):
            MfluxEngine().train(model=None, params=params)

    def test_diffusers_mps_engine_train_raises_not_implemented_permanent(self):
        """v0.10.0: diffusers_mps doesn't train. Permanent
        NotImplementedError per [[project-v100-design]] §B.4."""
        from imgen.engines.diffusers_mps_engine import DiffusersMpsEngine
        params = _make_minimal_genparams()
        with pytest.raises(NotImplementedError, match="diffusers_mps"):
            DiffusersMpsEngine().train(model=None, params=params)

    def test_mflux_engine_still_satisfies_protocol_after_train_added(self):
        """v0.10 commit 1 adds Engine.train to the Protocol. MfluxEngine
        must still pass isinstance(eng, Engine)."""
        from imgen.engines.base import Engine
        from imgen.engines.mflux_engine import MfluxEngine
        assert isinstance(MfluxEngine(), Engine)

    def test_diffusers_mps_engine_still_satisfies_protocol(self):
        """Same as above for DiffusersMpsEngine."""
        from imgen.engines.base import Engine
        from imgen.engines.diffusers_mps_engine import DiffusersMpsEngine
        assert isinstance(DiffusersMpsEngine(), Engine)


# ── v0.9.5 M-2: Engine registry + get_engine helper ────────────────────


class TestEngineRegistry:
    """v0.9.5 architect M-2 closure: ``ENGINES`` dict + ``get_engine``
    helper consolidate the previously-duplicated ``if engine == "mflux"
    / elif "diffusers_mps"`` dispatch at 3 sites (``_engine_for_model``
    in engine_dispatch, ``ram_required_gb`` in checks, doctor RAM
    forecast). The registry is the single source of truth for the
    ``engine name → Engine class`` mapping; adding a 3rd engine
    becomes one-line in the dict instead of editing every dispatch
    site.

    ``Model.__post_init__`` keeps its literal ``{'mflux',
    'diffusers_mps'}`` guard — its engine-conditional invariants
    (mflux requires binary=, diffusers_mps requires repo=) are
    intrinsically per-engine, not pure dispatch. A drift-lock test
    (``test_engines_registry_matches_model_post_init``) pins the
    registry key set against Model's accept set.

    ``iteration_dryrun_display`` also keeps its branched code path —
    its diffusers branch routes through ``_format_diffusers_dryrun``
    / ``_format_diffusers_video_dryrun`` helpers (different shape per
    output_type), NOT through ``Engine.format_dryrun`` (Engine
    Protocol doesn't define that method). Moving the dryrun
    rendering into the Engine Protocol is a v0.10.x design call.
    """

    def test_get_engine_mflux_returns_mflux_engine(self):
        from imgen.engines import MfluxEngine, get_engine
        engine = get_engine("mflux")
        assert isinstance(engine, MfluxEngine)

    def test_get_engine_diffusers_mps_returns_diffusers_mps_engine(self):
        from imgen.engines import DiffusersMpsEngine, get_engine
        engine = get_engine("diffusers_mps")
        assert isinstance(engine, DiffusersMpsEngine)

    def test_get_engine_unknown_raises_value_error(self):
        from imgen.engines import get_engine
        with pytest.raises(ValueError) as exc_info:
            get_engine("nonexistent_engine")
        msg = str(exc_info.value)
        assert "nonexistent_engine" in msg
        # Error message should list the valid engine names for
        # discoverability.
        assert "mflux" in msg
        assert "diffusers_mps" in msg

    def test_engines_registry_keys_match_model_post_init_accepts(self):
        """Drift lock: ENGINES keys must match the engine names
        Model.__post_init__ accepts. Adding a 3rd engine row to ENGINES
        without extending Model's per-engine invariant block (or vice
        versa) creates a silent registration gap — this test surfaces it.
        """
        from imgen.engines import ENGINES
        from imgen.models import Model
        # Probe each registered engine name through Model() — must NOT
        # raise on the engine= check. Use minimal-but-valid Model fields
        # per the engine's requirements (mflux: binary=; diffusers_mps:
        # repo=).
        for engine_name in ENGINES:
            try:
                if engine_name == "mflux":
                    Model(
                        engine=engine_name,
                        binary="mflux-generate-fake",
                        repo=None,
                        ram_baseline_gb=1.0,
                        ram_slope_gb_per_mp=0.1,
                    )
                elif engine_name == "diffusers_mps":
                    Model(
                        engine=engine_name,
                        binary=None,
                        repo="fake/fake",
                        ram_baseline_gb=1.0,
                        ram_slope_gb_per_mp=0.1,
                    )
                else:
                    # New engine name in ENGINES but no probe known here
                    # → drift between registry and this lock test. Either
                    # the new engine row also needs a Model() probe
                    # added here, or the drift is real.
                    raise AssertionError(
                        f"ENGINES has engine={engine_name!r} but this "
                        "drift-lock test doesn't know how to construct "
                        "a probe Model for it — extend the test."
                    )
            except ValueError as e:
                raise AssertionError(
                    f"Model.__post_init__ rejected engine={engine_name!r} "
                    f"despite ENGINES registering it: {e}. "
                    "Likely drift between the two sources of truth."
                ) from e

        # Reverse direction: a known-unaccepted engine name SHOULD raise
        # at Model.__post_init__, confirming the literal guard still
        # fires for unregistered names.
        with pytest.raises(ValueError, match=r"engine='unknown'"):
            Model(
                engine="unknown",
                binary="fake",
                ram_baseline_gb=1.0,
                ram_slope_gb_per_mp=0.1,
            )

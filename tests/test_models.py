"""v0.8.0 commit 1 — Model dataclass + GenParams.

Schema lock + __post_init__ invariants per
[[project-v080-design]] §F. These tests pin the dataclass field surface
so v0.8.1+ can't silently drop a field, AND prove that every Model
instantiation path (BUILTIN_MODELS at import, user TOMLs, test fixtures)
gets the same engine-conditional invariants enforced.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest


def _minimal_mflux_model(**overrides):
    """Smallest valid mflux Model — every test starts from here so when
    we test a single constraint, the surrounding fields are guaranteed
    valid."""
    from imgen.models import Model
    defaults = dict(
        engine="mflux",
        binary="mflux-generate",
        ram_baseline_gb=9.0,
        ram_slope_gb_per_mp=5.0,
    )
    defaults.update(overrides)
    return Model(**defaults)


def _minimal_diffusers_mps_model(**overrides):
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="Qwen/Qwen-Image-2512",
        ram_baseline_gb=24.0,
        ram_slope_gb_per_mp=8.0,
    )
    defaults.update(overrides)
    return Model(**defaults)


class TestModelDataclassShape:
    """Lock the v0.8.0 field surface — silent drops in v0.8.1+ caught
    by these tests."""

    def test_engine_is_required(self):
        from imgen.models import Model
        with pytest.raises(TypeError):
            Model()  # type: ignore[call-arg]

    def test_minimal_mflux_model_instantiates(self):
        m = _minimal_mflux_model()
        assert m.engine == "mflux"
        assert m.binary == "mflux-generate"

    def test_minimal_diffusers_mps_model_instantiates(self):
        m = _minimal_diffusers_mps_model()
        assert m.engine == "diffusers_mps"
        assert m.repo == "Qwen/Qwen-Image-2512"

    def test_frozen_dataclass(self):
        """frozen=True per §F — attribute reassignment must raise."""
        m = _minimal_mflux_model()
        with pytest.raises(FrozenInstanceError):
            m.engine = "diffusers_mps"  # type: ignore[misc]

    def test_v0_8_0_field_surface_locked(self):
        """Schema lock — every named field is present. If a field gets
        dropped or renamed, this test fails and forces a deliberate
        update. v0.9 commit 1 widens with `video` (nested VideoConfig)
        per [[project-v090-design]] §C."""
        from imgen.models import Model
        names = {f.name for f in fields(Model)}
        expected = {
            "engine", "binary", "repo", "extra_args", "image_flag",
            "cpu_offload_threshold_mp",
            "supports_strength", "supports_negative", "needs_token",
            "lora_compat_group", "hf_gated_repo",
            "default_steps", "default_guidance",
            "min_guidance", "max_guidance",
            "supported_quants", "omit_quantize", "param_overrides",
            "ram_baseline_gb", "ram_slope_gb_per_mp", "encoder_ram_gb",
            "enhance_system_prompt", "enhance_invariants",
            # v0.9 commit 1 — nested video config (None ⇒ image Model)
            "video",
        }
        assert expected == names, (
            f"Field surface drift: missing={expected - names}, "
            f"extra={names - expected}"
        )


class TestModelPostInitInvariants:
    """§F lock-in: __post_init__ enforces engine-conditional invariants
    at every instantiation site (BUILTIN_MODELS dict, user TOMLs, tests).
    """

    def test_mflux_without_binary_raises(self):
        from imgen.models import Model
        with pytest.raises(ValueError, match="engine='mflux'.*binary"):
            Model(
                engine="mflux",
                ram_baseline_gb=9.0,
                ram_slope_gb_per_mp=5.0,
            )

    def test_diffusers_mps_without_repo_raises(self):
        from imgen.models import Model
        with pytest.raises(ValueError, match="engine='diffusers_mps'.*repo"):
            Model(
                engine="diffusers_mps",
                ram_baseline_gb=24.0,
                ram_slope_gb_per_mp=8.0,
            )

    def test_unknown_engine_raises(self):
        from imgen.models import Model
        with pytest.raises(ValueError, match="engine="):
            Model(
                engine="opencv",
                binary="opencv-gen",
                ram_baseline_gb=9.0,
                ram_slope_gb_per_mp=5.0,
            )

    def test_zero_ram_baseline_raises(self):
        """Sentinel 0.0 means 'registry author forgot to declare' — must
        fail loudly per §F rather than silently letting preflight under-
        estimate."""
        from imgen.models import Model
        with pytest.raises(ValueError, match="ram_baseline_gb"):
            Model(
                engine="mflux",
                binary="mflux-generate",
                ram_baseline_gb=0.0,  # sentinel
                ram_slope_gb_per_mp=5.0,
            )

    def test_zero_ram_slope_raises(self):
        from imgen.models import Model
        with pytest.raises(ValueError, match="ram_slope_gb_per_mp"):
            Model(
                engine="mflux",
                binary="mflux-generate",
                ram_baseline_gb=9.0,
                ram_slope_gb_per_mp=0.0,
            )


class TestParamOverridesImmutability:
    """§F lock-in: param_overrides is tuple-of-tuples (immutable), not
    dict (which would have bypassed frozen=True via .update())."""

    def test_default_is_empty_tuple(self):
        m = _minimal_mflux_model()
        assert m.param_overrides == ()
        assert isinstance(m.param_overrides, tuple)

    def test_explicit_tuple_of_pairs(self):
        m = _minimal_mflux_model(
            param_overrides=(("cfg_normalization", False),),
        )
        assert m.param_overrides == (("cfg_normalization", False),)

    def test_tuple_is_immutable(self):
        """tuple.append doesn't exist; mutation attempts raise
        AttributeError or TypeError. Lock-in proves we didn't accidentally
        regress to dict/list."""
        m = _minimal_mflux_model(
            param_overrides=(("true_cfg_scale", 4.0),),
        )
        with pytest.raises((AttributeError, TypeError)):
            m.param_overrides.append(("another", 1))  # type: ignore[attr-defined]


class TestBuiltinModelsLiteral:
    """v0.8.0 commit 4b (§Q): BUILTIN_MODELS becomes the LIVE registry
    via literal declaration with v0.8 canonical keys. The commit-2/3
    ``_model_from_backend``-derived-view contract is gone; BUILTIN_MODELS
    is now its own source-of-truth (per §G.1).

    BUILTIN_BACKENDS becomes the BACKWARD-DERIVED v0.7-keyed view
    (architect 4b pre-vet HIGH-1) so v0.7.x test fixtures asserting
    ``BACKENDS["flux"]`` shape stay green without churn.
    """

    def test_builtin_models_keyed_by_v08_canonical_names(self):
        """4b lock-in: literal declaration uses v0.8 names. Two renames
        (flux→flux-kontext, qwen→qwen-image-edit-v1) per §I; other names
        unchanged. v0.9 commit 7 added ``ltx-video`` (§K, pulled forward
        from commit 9 in lockstep with cmd_video so parser default
        resolves)."""
        from imgen.models import BUILTIN_MODELS
        assert set(BUILTIN_MODELS.keys()) == {
            "flux-kontext",
            "qwen-image-edit-v1",
            "flux-dev",
            "flux2-klein-edit-9b",
            "ltx-video",
        }

    def test_builtin_models_engine_routing(self):
        """Per-row engine routing — mflux for image, diffusers_mps for
        video. Pre-v0.9 this was an "all mflux" invariant; v0.9 added
        the first diffusers_mps built-in (ltx-video) for t2v."""
        from imgen.models import BUILTIN_MODELS
        expected_engines = {
            "flux-kontext": "mflux",
            "qwen-image-edit-v1": "mflux",
            "flux-dev": "mflux",
            "flux2-klein-edit-9b": "mflux",
            "ltx-video": "diffusers_mps",
        }
        for name, m in BUILTIN_MODELS.items():
            assert m.engine == expected_engines[name], (
                f"built-in {name!r} has engine={m.engine!r}, "
                f"expected {expected_engines[name]!r}"
            )

    def test_builtin_models_have_required_v08_fields(self):
        """Literal declaration must populate ram_baseline_gb /
        ram_slope_gb_per_mp (the __post_init__ sentinels) — would have
        raised at module load otherwise; this asserts non-zero values
        per row."""
        from imgen.models import BUILTIN_MODELS
        for name, m in BUILTIN_MODELS.items():
            assert m.ram_baseline_gb > 0.0, name
            assert m.ram_slope_gb_per_mp > 0.0, name

    def test_v08_to_v07_rename_map_inverts_for_renamed_models(self):
        """The inverse of ``_V07_TO_V08_MODEL_RENAMES`` maps v0.8
        canonical names back to v0.7 keys. backends.py uses this
        inversion for the backward-derived BUILTIN_BACKENDS view."""
        from imgen.models import _V07_TO_V08_MODEL_RENAMES
        assert _V07_TO_V08_MODEL_RENAMES["flux"] == "flux-kontext"
        assert _V07_TO_V08_MODEL_RENAMES["qwen"] == "qwen-image-edit-v1"
        # Unchanged names not present in the map
        assert "flux-dev" not in _V07_TO_V08_MODEL_RENAMES
        assert "flux2-klein-edit-9b" not in _V07_TO_V08_MODEL_RENAMES

    def test_builtin_backends_derived_backward_with_v07_keys(self):
        """architect 4b HIGH-1: BUILTIN_BACKENDS is keyed by v0.7
        names (derived backward from BUILTIN_MODELS via the inverse
        rename map). v0.7.x test fixtures like ``BACKENDS["flux"]``
        keep working unchanged. v0.9 commit 7 added ``ltx-video`` — no
        v0.7 alias rename map entry (v0.7 never had a video name);
        passes through as-is."""
        from imgen.backends import BUILTIN_BACKENDS
        assert set(BUILTIN_BACKENDS.keys()) == {
            "flux",
            "qwen",
            "flux-dev",
            "flux2-klein-edit-9b",
            "ltx-video",
        }

    def test_backend_from_model_preserves_v07_shape(self):
        """The conversion helper drops v0.8-only fields (engine, repo,
        ram_*, default_*) but preserves every v0.7 Backend field."""
        from imgen.backends import BACKENDS
        be = BACKENDS["flux"]
        assert be.binary == "mflux-generate-kontext"
        assert be.image_flag == "--image-path"
        assert be.supports_strength is True
        assert be.supports_negative is True
        assert be.extra_args == ("--model", "dev")
        assert be.needs_token is True
        assert be.lora_compat_group == "flux-1"
        assert be.hf_gated_repo == "black-forest-labs/FLUX.1-Kontext-dev"


class TestGenParams:
    """GenParams is the pure-data envelope passed between cli and Engine
    per §C. frozen+slots so engines can rely on identity stability."""

    def test_genparams_instantiates_with_required_fields(self):
        from imgen.engines.base import GenParams
        from pathlib import Path
        p = GenParams(
            prompt="a samurai",
            negative="",
            width=1024,
            height=1024,
            steps=20,
            guidance=3.5,
            seed=42,
            quantize=8,
            strength=0.5,
            input_path=None,
            output_path=Path("/tmp/out.png"),
            loras=(),
        )
        assert p.prompt == "a samurai"
        assert p.loras == ()

    def test_genparams_is_frozen(self):
        from imgen.engines.base import GenParams
        from pathlib import Path
        p = GenParams(
            prompt="x", negative="", width=64, height=64, steps=1,
            guidance=0.0, seed=0, quantize=4, strength=0.0,
            input_path=None, output_path=Path("/tmp/x.png"), loras=(),
        )
        with pytest.raises(FrozenInstanceError):
            p.prompt = "y"  # type: ignore[misc]

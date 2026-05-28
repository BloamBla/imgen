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
        per [[project-v090-design]] §C. v0.10 commit 1 widens with
        ``training`` (nested TrainingConfig) per [[project-v100-design]]
        §C — None ⇒ Model cannot be a LoRA-training target."""
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
            # v0.11.0 — per-Model inference quant default (None ⇒ global)
            "default_quantize",
            "ram_baseline_gb", "ram_slope_gb_per_mp", "encoder_ram_gb",
            "enhance_system_prompt", "enhance_invariants",
            # v0.9 commit 1 — nested video config (None ⇒ image Model)
            "video",
            # v0.10 commit 1 — nested training config (None ⇒ not trainable)
            "training",
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
        resolves). v0.10 commit 2 added ``flux2-klein-4b`` — the first
        inference+training-capable Model per
        [[project-v100-design]] §B.3."""
        from imgen.models import BUILTIN_MODELS
        assert set(BUILTIN_MODELS.keys()) == {
            "flux-kontext",
            "qwen-image-edit-v1",
            "flux-dev",
            "flux2-klein-edit-9b",
            "ltx-video",
            "flux2-klein-4b",
        }

    def test_builtin_models_engine_routing(self):
        """Per-row engine routing — mflux for image, diffusers_mps for
        video. Pre-v0.9 this was an "all mflux" invariant; v0.9 added
        the first diffusers_mps built-in (ltx-video) for t2v. v0.10
        added flux2-klein-4b (mflux, inference+training)."""
        from imgen.models import BUILTIN_MODELS
        expected_engines = {
            "flux-kontext": "mflux",
            "qwen-image-edit-v1": "mflux",
            "flux-dev": "mflux",
            "flux2-klein-edit-9b": "mflux",
            "ltx-video": "diffusers_mps",
            "flux2-klein-4b": "mflux",
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
        passes through as-is. v0.10 commit 2 added ``flux2-klein-4b``
        — also not in the v0.7 rename map (klein-4b didn't exist in
        v0.7); passes through unchanged."""
        from imgen.backends import BUILTIN_BACKENDS
        assert set(BUILTIN_BACKENDS.keys()) == {
            "flux",
            "qwen",
            "flux-dev",
            "flux2-klein-edit-9b",
            "ltx-video",
            "flux2-klein-4b",
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


class TestFlux2Klein4bRow:
    """v0.10.0 commit 2 — ``flux2-klein-4b`` BUILTIN_MODELS row per
    [[project-v100-design]] §B.3 + §R.1 ROUND-1 CLOSURES.

    First inference+training-capable Model. Honest-framed 6 added
    surfaces per architect H-2 closure:
    1. ``--list-models`` entry
    2. doctor RAM forecast row (covered by per-Model RAM math)
    3. parser default propagation (commit 4 for parser stanza)
    4. ``lora_compat_group="flux2-klein-4b"`` for LoRA compat checks
    5. real-mflux-smoke obligation (pre-tag gate per §K)
    6. ``enhance_system_prompt`` decision (v0.10.0: None — klein-4b's
       prompt conventions differ from FLUX.1; defer enhancer to v0.10.x)
    """

    def test_klein_4b_in_registry(self):
        from imgen.models import BUILTIN_MODELS
        assert "flux2-klein-4b" in BUILTIN_MODELS

    def test_klein_4b_uses_mflux_generate_flux2_binary(self):
        """Per colleague's recipe + mflux 0.17.5: klein-4b base t2i
        inference uses ``mflux-generate-flux2``. The edit variant
        (klein-9b) uses ``mflux-generate-flux2-edit``."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.binary == "mflux-generate-flux2"
        assert m.extra_args == ("-m", "flux2-klein-4b")

    def test_klein_4b_is_t2i_not_edit(self):
        """klein-4b is the base distilled t2i model — no input image,
        no strength. Distinct from klein-edit-9b which IS i2i."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        # No image flag for pure t2i — same shape as flux-dev (which
        # has image_flag set for dataclass-shape consistency but the
        # binary doesn't accept input). Per the FLUX.1 precedent in
        # flux-dev's row (image_flag="--image-path" but gated by
        # input_path is None at build_cmd), klein-4b mirrors that
        # pattern. The build_cmd gate fires per existing logic.
        assert m.supports_strength is False, (
            "klein-4b is t2i base, not i2i edit — no strength"
        )

    def test_klein_4b_flux2_family_no_negative_no_cfg(self):
        """FLUX.2 family (klein-4b + klein-9b-edit) deliberately
        dropped negative prompt + CFG support. Per existing klein-9b
        row pattern: min_guidance=max_guidance=1.0, supports_negative=False."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.supports_negative is False, (
            "FLUX.2 family does not support negative prompt"
        )
        assert m.default_guidance == 1.0
        assert m.min_guidance == 1.0
        assert m.max_guidance == 1.0, (
            "klein-4b distilled — mflux pins guidance to 1.0 exactly"
        )

    def test_klein_4b_gated_repo(self):
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.needs_token is True
        assert m.hf_gated_repo == "black-forest-labs/FLUX.2-klein-4B"

    def test_klein_4b_lora_compat_group_distinct(self):
        """§R.1 architect H-5 closure: per-base lora_compat_group so
        LoRAs trained on klein-4b don't silently mis-route to klein-9b
        (architecturally different — 4B vs 9B params, distinct LoRA
        weight tensors)."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.lora_compat_group == "flux2-klein-4b"
        # Distinct from klein-9b-edit row (which is "flux2-klein-9b")
        m9b = BUILTIN_MODELS["flux2-klein-edit-9b"]
        assert m.lora_compat_group != m9b.lora_compat_group, (
            "klein-4b and klein-9b LoRAs are architecturally incompatible "
            "— compat groups MUST differ to prevent silent mis-routing"
        )

    def test_klein_4b_inference_ram_lighter_than_klein_9b(self):
        """Sanity: klein-4b has ~half the params of klein-9b, so
        ram_baseline_gb should be meaningfully lower. Klein-9b row
        sets baseline=27.0 (Q8 1MP); klein-4b should be ~half. Exact
        value calibrated from real-mflux smoke (pre-tag gate per §L)."""
        from imgen.models import BUILTIN_MODELS
        m4b = BUILTIN_MODELS["flux2-klein-4b"]
        m9b = BUILTIN_MODELS["flux2-klein-edit-9b"]
        assert m4b.ram_baseline_gb > 0  # sentinel rule
        assert m4b.ram_slope_gb_per_mp > 0
        assert m4b.ram_baseline_gb < m9b.ram_baseline_gb, (
            f"klein-4b ({m4b.ram_baseline_gb} GB) should have lower "
            f"baseline than klein-9b ({m9b.ram_baseline_gb} GB)"
        )

    def test_klein_4b_training_supported(self):
        """v0.10.0 FIRST training-capable Model. ``Model.training`` is
        a TrainingConfig instance (not None)."""
        from imgen.models import BUILTIN_MODELS, TrainingConfig
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.training_supported is True
        assert isinstance(m.training, TrainingConfig)

    def test_klein_4b_training_uses_module_constant(self):
        """§R.1 architect C-2 closure: klein-4b row references
        ``_KLEIN_4B_TARGET_MODULES`` module constant by NAME, not
        embedded literal. Single source of truth (no schema-vs-content
        drift)."""
        from imgen.models import BUILTIN_MODELS, _KLEIN_4B_TARGET_MODULES
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.training.target_modules is _KLEIN_4B_TARGET_MODULES, (
            "klein-4b row's target_modules MUST be the module-level "
            "_KLEIN_4B_TARGET_MODULES constant (identity check), NOT "
            "an inline copy — B-1 anti-pattern shape per v0.9.3 precedent."
        )

    def test_klein_4b_training_peak_ram_calibrated(self):
        """§M.1 smoke (M2 Pro 32 GB, 2026-05-28) measured klein-4b
        training at ~21 GB resident + ~3 GB swap (q4/512/rank16/low_ram);
        v0.10.0 ships training_peak_ram_gb=22.0. Must stay >0 (sentinel
        rule) and well under 32 so the preflight pings green on the
        primary target without forcing --force unnecessarily."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.training.training_peak_ram_gb > 0
        assert m.training.training_peak_ram_gb < 28.0, (
            "training_peak_ram_gb should reflect the §M.1-measured ~21 GB "
            "peak, not the pre-smoke 28 GB guess"
        )

    def test_klein_4b_no_enhance_system_prompt_at_v0_10_0(self):
        """v0.10.0 ships klein-4b with ``enhance_system_prompt=None`` per
        §B.3 honest framing: klein-4b prompt conventions differ from
        FLUX.1 family; enhancer needs a dedicated per-Model prompt that
        v0.10.0 defers. User can add via models.d/ TOML or v0.10.x can
        ship one."""
        from imgen.models import BUILTIN_MODELS
        m = BUILTIN_MODELS["flux2-klein-4b"]
        assert m.enhance_system_prompt is None


class TestModelTrainingField:
    """v0.10.0 commit 1 — ``Model.training: TrainingConfig | None``
    nested field per [[project-v100-design]] §R.1 ROUND-1 CLOSURES.

    Mirrors the v0.9 ``Model.video`` pattern: None ⇒ image-or-video
    Model that cannot be a LoRA-training target; present ⇒ Model is a
    valid ``imgen train --base <name>`` target. Cross-rule in
    __post_init__ requires engine='mflux' when training is set (v0.10.0
    does not train via diffusers_mps)."""

    def test_training_field_defaults_to_none(self):
        m = _minimal_mflux_model()
        assert m.training is None

    def test_training_supported_property_false_when_training_none(self):
        m = _minimal_mflux_model()
        assert m.training_supported is False

    def test_training_supported_property_true_when_training_set(self):
        from imgen.models import (
            TrainingConfig,
            TrainingTargetSpec,
        )
        tc = TrainingConfig(
            training_peak_ram_gb=28.0,
            target_modules=(
                TrainingTargetSpec(
                    module_path="transformer_blocks.{block}.attn.to_q",
                    blocks=(0, 38),
                    rank=16,
                ),
            ),
        )
        m = _minimal_mflux_model(training=tc)
        assert m.training_supported is True

    def test_training_requires_mflux_engine(self):
        """§R.1 closure: training on diffusers_mps Model raises in
        __post_init__. v0.10.0 ships only mflux-based training; video
        Models stay inference-only."""
        from imgen.models import (
            Model,
            TrainingConfig,
            TrainingTargetSpec,
        )
        tc = TrainingConfig(
            training_peak_ram_gb=28.0,
            target_modules=(
                TrainingTargetSpec(
                    module_path="x.{block}.y", blocks=(0, 10), rank=16,
                ),
            ),
        )
        with pytest.raises(ValueError, match=r"[Tt]raining.*mflux"):
            Model(
                engine="diffusers_mps",
                repo="fake/fake",
                ram_baseline_gb=10.0,
                ram_slope_gb_per_mp=4.0,
                training=tc,
            )

    def test_v09_builtins_still_valid_after_v10_field_addition(self):
        """§R.1 backwards-compatibility lock: every v0.9 BUILTIN_MODELS
        row instantiates cleanly without setting ``training=``.

        flux-kontext / qwen-image-edit-v1 / flux-dev /
        flux2-klein-edit-9b / ltx-video all default training=None and
        the v0.10 cross-rule in __post_init__ is a no-op for them."""
        from imgen.models import BUILTIN_MODELS
        for name, m in BUILTIN_MODELS.items():
            # All v0.9 rows default training=None until commit 2 adds
            # the klein-4b row with training=TrainingConfig(...).
            if name == "flux2-klein-4b":
                # commit 2 will set training= on this row
                continue
            assert m.training is None, (
                f"v0.9 builtin {name!r} unexpectedly has training set"
            )
            assert m.training_supported is False, name


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


class TestModelsForCompatGroups:
    """P2: maps a LoRA's compat group(s) to the --model(s) that load it,
    powering the actionable incompat warn."""

    def test_klein_4b_group_maps_to_klein_4b_model(self):
        from imgen.models import models_for_compat_groups
        assert models_for_compat_groups(("flux2-klein-4b",)) == [
            "flux2-klein-4b",
        ]

    def test_unknown_group_maps_to_empty(self):
        from imgen.models import models_for_compat_groups
        assert models_for_compat_groups(("flux-2",)) == []

    def test_empty_groups_maps_to_empty(self):
        from imgen.models import models_for_compat_groups
        assert models_for_compat_groups(()) == []

    def test_result_is_sorted(self):
        from imgen.models import BUILTIN_MODELS, models_for_compat_groups
        # Every built-in model with a non-empty compat group should be
        # reachable from its own group, and the result is sorted.
        groups = tuple(
            m.lora_compat_group for m in BUILTIN_MODELS.values()
            if m.lora_compat_group
        )
        out = models_for_compat_groups(groups)
        assert out == sorted(out)

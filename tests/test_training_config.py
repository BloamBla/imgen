"""v0.10.0 commit 1 — TrainingConfig + TrainingTargetSpec schema lock.

Per [[project-v100-design]] §R.1 ROUND-1 CLOSURES preamble (canonical
shape). These tests pin the dataclass field surface for the LoRA-training
config nested onto ``Model.training`` so v0.10.x can't silently drift
the contract before mflux-train picks it up.

Canonical schema (post §R.1):
* ``TrainingTargetSpec``: matches real mflux JSON ``lora_layers.targets[]``
  shape — ``{module_path: str, blocks: tuple[int, int] | None, rank: int}``.
  NOT the original draft's ``(block_type, attn_target_keys)`` tuple-shape.
* ``TrainingConfig``: NO ``mflux_train_model`` field (derived from
  registry key in cmd_train per §R.1). NO ``_TRAINING_BASE_ALLOWLIST``
  (``Model.training is not None`` IS the signal). ``default_epochs``
  not ``default_steps_per_image`` (semantically correct: total_steps =
  epochs × len(dataset) × batch_size per colleague's recipe).
* Module constants: ``_VALID_LORA_RANKS`` / ``_VALID_QUANTIZE_TRAIN`` /
  ``_VALID_TRAIN_RESOLUTIONS`` consumed by BOTH ``__post_init__`` AND
  argparse ``choices=`` (commit 4) — no drift bugs at v0.10.x rank=128
  additions.
* ``_KLEIN_4B_TARGET_MODULES`` lifted to module-level constant for the
  klein-4b BUILTIN_MODELS row (commit 2) to reference — closes the B-1
  anti-pattern that v0.9.3 fixed for ``pipeline_class``.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest


# ── TrainingTargetSpec schema lock ─────────────────────────────────────


class TestTrainingTargetSpecShape:
    """Schema lock for the nested-list-of-targets dataclass matching
    real mflux ``lora_layers.targets[]`` JSON. Each entry has its OWN
    rank (NOT a single top-level rank) per mflux's actual JSON shape."""

    def test_field_surface_locked(self):
        from imgen.models import TrainingTargetSpec
        names = {f.name for f in fields(TrainingTargetSpec)}
        expected = {"module_path", "blocks", "rank"}
        assert expected == names, (
            f"TrainingTargetSpec field drift: missing={expected - names}, "
            f"extra={names - expected}"
        )

    def test_minimal_instantiation(self):
        from imgen.models import TrainingTargetSpec
        spec = TrainingTargetSpec(
            module_path="transformer_blocks.{block}.attn.to_q",
            blocks=(0, 38),
            rank=16,
        )
        assert spec.module_path == "transformer_blocks.{block}.attn.to_q"
        assert spec.blocks == (0, 38)
        assert spec.rank == 16

    def test_blocks_none_means_single_module(self):
        """For non-templated module_path (no ``{block}`` placeholder),
        ``blocks=None`` is the valid shape — e.g. a top-level layer
        like ``cap_embedder.1``."""
        from imgen.models import TrainingTargetSpec
        spec = TrainingTargetSpec(
            module_path="cap_embedder.1",
            blocks=None,
            rank=16,
        )
        assert spec.blocks is None

    def test_frozen_immutable(self):
        from imgen.models import TrainingTargetSpec
        spec = TrainingTargetSpec(
            module_path="x.{block}.y", blocks=(0, 10), rank=16,
        )
        with pytest.raises(FrozenInstanceError):
            spec.rank = 32  # type: ignore[misc]


class TestTrainingTargetSpecPostInitInvariants:
    """Validation rules per §R.1 preamble closure of python H-9."""

    def test_empty_module_path_raises(self):
        from imgen.models import TrainingTargetSpec
        with pytest.raises(ValueError, match="module_path"):
            TrainingTargetSpec(module_path="", blocks=None, rank=16)

    def test_module_path_with_control_bytes_raises(self):
        from imgen.models import TrainingTargetSpec
        with pytest.raises(ValueError, match="module_path"):
            TrainingTargetSpec(
                module_path="layers.{block}.attn\x1b.to_q",
                blocks=(0, 10),
                rank=16,
            )

    def test_blocks_template_mismatch_raises(self):
        """``{block}`` placeholder present but ``blocks=None`` → spec is
        unrenderable. Inverse also rejected — ``blocks=(0, N)`` but no
        placeholder."""
        from imgen.models import TrainingTargetSpec
        with pytest.raises(ValueError, match=r"\{block\}"):
            TrainingTargetSpec(
                module_path="transformer_blocks.{block}.attn.to_q",
                blocks=None,
                rank=16,
            )
        with pytest.raises(ValueError, match=r"\{block\}"):
            TrainingTargetSpec(
                module_path="cap_embedder.1",
                blocks=(0, 38),  # no placeholder to substitute into
                rank=16,
            )

    def test_blocks_invalid_range_raises(self):
        from imgen.models import TrainingTargetSpec
        # start > end
        with pytest.raises(ValueError, match="blocks"):
            TrainingTargetSpec(
                module_path="x.{block}.y", blocks=(10, 5), rank=16,
            )
        # negative start
        with pytest.raises(ValueError, match="blocks"):
            TrainingTargetSpec(
                module_path="x.{block}.y", blocks=(-1, 10), rank=16,
            )

    def test_rank_must_be_in_valid_set(self):
        from imgen.models import TrainingTargetSpec, _VALID_LORA_RANKS
        # 7 not in {4, 8, 16, 32, 64}
        with pytest.raises(ValueError, match="rank"):
            TrainingTargetSpec(
                module_path="x.{block}.y", blocks=(0, 10), rank=7,
            )
        # 16 is valid (covered by minimal test above)
        assert 16 in _VALID_LORA_RANKS


# ── Module constants exposed for argparse + post_init reuse ────────────


class TestModuleConstants:
    """§R.1 python H-13 closure: shared constants prevent drift between
    ``__post_init__`` and argparse ``choices=`` at commit 4."""

    def test_valid_lora_ranks_locked(self):
        from imgen.models import _VALID_LORA_RANKS
        assert isinstance(_VALID_LORA_RANKS, frozenset)
        assert _VALID_LORA_RANKS == frozenset({4, 8, 16, 32, 64})

    def test_valid_quantize_train_locked(self):
        """mflux-train --quantize {3,4,5,6,8} — note 0 NOT in set (mflux
        inference accepts 0 = bf16, but mflux-train doesn't)."""
        from imgen.models import _VALID_QUANTIZE_TRAIN
        assert isinstance(_VALID_QUANTIZE_TRAIN, frozenset)
        assert _VALID_QUANTIZE_TRAIN == frozenset({3, 4, 5, 6, 8})

    def test_valid_train_resolutions_locked(self):
        from imgen.models import _VALID_TRAIN_RESOLUTIONS
        assert isinstance(_VALID_TRAIN_RESOLUTIONS, frozenset)
        assert _VALID_TRAIN_RESOLUTIONS == frozenset(
            {256, 384, 512, 768, 1024}
        )


# ── TrainingConfig schema lock ─────────────────────────────────────────


def _minimal_training_config(**overrides):
    """Smallest valid TrainingConfig — every test starts from here.
    Uses klein-4b-shaped defaults from the colleague's validated
    recipe."""
    from imgen.models import (
        TrainingConfig,
        TrainingTargetSpec,
    )
    defaults = dict(
        training_peak_ram_gb=28.0,
        target_modules=(
            TrainingTargetSpec(
                module_path="transformer_blocks.{block}.attn.to_q",
                blocks=(0, 38),
                rank=16,
            ),
        ),
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestTrainingConfigShape:
    """Schema lock for the post-§R.1 canonical field surface. NO
    ``mflux_train_model`` field — derived from registry key in cmd_train
    (§R.1 round-1 closure). NO ``_TRAINING_BASE_ALLOWLIST`` constant
    — ``Model.training is not None`` IS the signal."""

    def test_field_surface_locked(self):
        from imgen.models import TrainingConfig
        names = {f.name for f in fields(TrainingConfig)}
        expected = {
            "training_peak_ram_gb",
            "default_lora_rank",
            "default_max_resolution",
            "default_quantize",
            "default_epochs",  # §R.1: renamed from default_steps_per_image
            "default_low_ram",
            "optimizer_name",
            "optimizer_lr",
            "target_modules",
            "default_preview_frequency",
        }
        assert expected == names, (
            f"TrainingConfig field drift: missing={expected - names}, "
            f"extra={names - expected}"
        )

    def test_no_mflux_train_model_field(self):
        """§R.1 closure: derived from registry key in cmd_train, not a
        TrainingConfig field. Single source of truth = registry key."""
        from imgen.models import TrainingConfig
        names = {f.name for f in fields(TrainingConfig)}
        assert "mflux_train_model" not in names, (
            "§R.1: mflux_train_model field was dropped; derive from "
            "Model registry key in cmd_train."
        )

    def test_no_training_base_allowlist_constant(self):
        """§R.1 closure: ``Model.training is not None`` IS the signal
        for trainability. A separate allowlist constant would duplicate
        the registry."""
        import imgen.models as m
        assert not hasattr(m, "_TRAINING_BASE_ALLOWLIST"), (
            "§R.1: _TRAINING_BASE_ALLOWLIST dropped per architect C-2 "
            "closure (B-1 anti-pattern shape)."
        )

    def test_frozen_immutable(self):
        tc = _minimal_training_config()
        with pytest.raises(FrozenInstanceError):
            tc.default_epochs = 100  # type: ignore[misc]

    def test_defaults_match_colleague_recipe(self):
        """Colleague's M5 Pro 48 GB klein-4b recipe (validated 2026-05-27):
        rank 16, max_res 512, q4, 80 epochs, low_ram, AdamW lr 1e-4,
        preview every 100 steps (sparser than colleague's 10 to keep
        wall time reasonable on M2 Pro 32 GB)."""
        tc = _minimal_training_config()
        assert tc.default_lora_rank == 16
        assert tc.default_max_resolution == 512
        assert tc.default_quantize == 4
        assert tc.default_epochs == 80
        assert tc.default_low_ram is True
        assert tc.optimizer_name == "AdamW"
        assert tc.optimizer_lr == 1e-4
        assert tc.default_preview_frequency == 100


class TestTrainingConfigPostInitInvariants:
    """§R.1 preamble closure: numeric ranges + allowlist sets + sentinel
    catches. Each test pins one rule from the canonical shape."""

    def test_target_modules_empty_raises(self):
        from imgen.models import TrainingConfig
        with pytest.raises(ValueError, match="target_modules"):
            TrainingConfig(
                training_peak_ram_gb=28.0,
                target_modules=(),  # empty
            )

    def test_training_peak_ram_zero_sentinel_raises(self):
        """training_peak_ram_gb=0.0 means registry author forgot to
        declare — preflight gate is load-bearing on this value."""
        from imgen.models import TrainingConfig, TrainingTargetSpec
        with pytest.raises(ValueError, match="training_peak_ram_gb"):
            TrainingConfig(
                training_peak_ram_gb=0.0,  # sentinel
                target_modules=(
                    TrainingTargetSpec(
                        module_path="x.{block}.y", blocks=(0, 10), rank=16,
                    ),
                ),
            )

    def test_default_lora_rank_not_in_valid_set_raises(self):
        with pytest.raises(ValueError, match="default_lora_rank"):
            _minimal_training_config(default_lora_rank=7)

    def test_default_quantize_not_in_valid_set_raises(self):
        with pytest.raises(ValueError, match="default_quantize"):
            _minimal_training_config(default_quantize=0)  # not in {3..8}

    def test_default_max_resolution_not_in_valid_set_raises(self):
        with pytest.raises(ValueError, match="default_max_resolution"):
            _minimal_training_config(default_max_resolution=500)

    def test_default_epochs_floor_raises(self):
        """§R.1 architect M-1 closure: default_epochs (renamed from
        default_steps_per_image). Floor 10 — under-trains below."""
        with pytest.raises(ValueError, match="default_epochs"):
            _minimal_training_config(default_epochs=5)

    def test_optimizer_name_allowlist_raises(self):
        with pytest.raises(ValueError, match="optimizer_name"):
            _minimal_training_config(optimizer_name="SGD")  # not in set

    def test_optimizer_lr_range_tightened_low(self):
        """§R.1 python M-5 closure: tighten lr range to 1e-6..1e-2."""
        with pytest.raises(ValueError, match="optimizer_lr"):
            _minimal_training_config(optimizer_lr=1e-7)  # too low

    def test_optimizer_lr_range_tightened_high(self):
        with pytest.raises(ValueError, match="optimizer_lr"):
            _minimal_training_config(optimizer_lr=0.5)  # too high

    def test_optimizer_lr_canonical_1e_4_accepted(self):
        """Colleague's recipe — 1e-4 is the canonical AdamW lr."""
        tc = _minimal_training_config(optimizer_lr=1e-4)
        assert tc.optimizer_lr == 1e-4

    def test_default_preview_frequency_floor(self):
        with pytest.raises(ValueError, match="default_preview_frequency"):
            _minimal_training_config(default_preview_frequency=0)


# ── _KLEIN_4B_TARGET_MODULES module-level constant ─────────────────────


class TestKlein4bTargetModulesConstant:
    """§R.1 architect C-2 closure: lift to module-level constant so
    BUILTIN_MODELS row (commit 2) references it by NAME, not by literal
    embedding. Single source of truth — closes B-1 anti-pattern shape
    that v0.9.3 fixed for ``pipeline_class``."""

    def test_constant_exposed(self):
        from imgen.models import _KLEIN_4B_TARGET_MODULES
        assert _KLEIN_4B_TARGET_MODULES is not None

    def test_constant_is_tuple_of_specs(self):
        from imgen.models import (
            TrainingTargetSpec,
            _KLEIN_4B_TARGET_MODULES,
        )
        assert isinstance(_KLEIN_4B_TARGET_MODULES, tuple)
        assert len(_KLEIN_4B_TARGET_MODULES) > 0
        for spec in _KLEIN_4B_TARGET_MODULES:
            assert isinstance(spec, TrainingTargetSpec)

    def test_constant_covers_attention_qkv_for_transformer_blocks(self):
        """Klein-4b LoRA targets per colleague's recipe + FLUX.2
        transformer architecture grep (transformer_blocks.{block}.attn.*
        per mflux/models/flux2/weights/flux2_lora_mapping.py:336)."""
        from imgen.models import _KLEIN_4B_TARGET_MODULES
        module_paths = {s.module_path for s in _KLEIN_4B_TARGET_MODULES}
        assert any(
            "transformer_blocks.{block}.attn.to_q" in p
            for p in module_paths
        ), f"missing attn.to_q in klein-4b targets: {module_paths}"
        assert any(
            "transformer_blocks.{block}.attn.to_k" in p
            for p in module_paths
        )
        assert any(
            "transformer_blocks.{block}.attn.to_v" in p
            for p in module_paths
        )

    def test_constant_covers_single_transformer_blocks(self):
        """Per colleague + FLUX.2 transformer.py:20 (num_single_layers=20),
        single_transformer_blocks is the second target set for klein-4b
        identity training."""
        from imgen.models import _KLEIN_4B_TARGET_MODULES
        module_paths = {s.module_path for s in _KLEIN_4B_TARGET_MODULES}
        assert any(
            "single_transformer_blocks" in p for p in module_paths
        ), (
            f"missing single_transformer_blocks target in klein-4b set: "
            f"{module_paths}"
        )

    def test_constant_blocks_ranges_within_klein_4b_layer_count(self):
        """FLUX.2 klein-4b real architecture (transformer.py:19-20):
        num_layers=5 double-stream transformer_blocks + num_single_layers=20
        single_transformer_blocks. mflux BlockRange end is EXCLUSIVE, so the
        constant must stay within (0, 5] double / (0, 20] single. The §M.1
        smoke (2026-05-28) caught the original (0, 38) draft as an IndexError
        at LoRA injection — this test now locks the real counts so a
        regression to the FLUX.1-dev-shaped range is rejected."""
        from imgen.models import _KLEIN_4B_TARGET_MODULES
        for spec in _KLEIN_4B_TARGET_MODULES:
            if spec.blocks is None:
                continue
            start, end = spec.blocks
            if "single_transformer_blocks" in spec.module_path:
                assert end <= 20, (
                    f"single_transformer_blocks range exceeds klein-4b's "
                    f"20-layer limit: {spec.module_path} blocks={spec.blocks}"
                )
            elif "transformer_blocks" in spec.module_path:
                assert end <= 5, (
                    f"transformer_blocks range exceeds klein-4b's 5-layer "
                    f"limit (num_layers=5): {spec.module_path} "
                    f"blocks={spec.blocks}"
                )

    def test_constant_used_with_default_rank_16(self):
        """Colleague's recipe uses rank=16 across all targets. v0.10.0
        defaults match."""
        from imgen.models import _KLEIN_4B_TARGET_MODULES
        ranks = {s.rank for s in _KLEIN_4B_TARGET_MODULES}
        assert ranks == {16}, (
            f"expected all klein-4b targets at rank=16 per colleague "
            f"recipe; got {ranks}"
        )

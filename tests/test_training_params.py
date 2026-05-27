"""v0.10.0 commit 5 — :class:`TrainingParams` envelope.

Covers the pure-data dataclass that carries resolved parameters from
``cmd_train`` to ``MfluxEngine.train`` (and through ``build_config_json``
into the mflux-train JSON config).

Per [[project-v100-design]] §E.1 + §R.1 ROUND-1 CLOSURES — ``MfluxTrainer``
class DROPPED; ``TrainingParams`` lives at module level alongside
``build_config_json`` so the JSON-building logic stays a pure function
that can be unit-tested without an Engine instance.

Tested invariants mirror :class:`TrainingConfig` (commit 1) — same
allowlist shared constants from ``imgen.models``, same shape semantics
for the LoRA targets, plus runtime-only fields (scratch_dir,
output_path, seed, battery_stop).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from imgen.engines._training import TrainingParams
from imgen.models import (
    _KLEIN_4B_TARGET_MODULES,
    _VALID_LORA_RANKS,
    _VALID_QUANTIZE_TRAIN,
    _VALID_TRAIN_RESOLUTIONS,
    TrainingTargetSpec,
)


def _minimal_training_params(**overrides):
    """klein-4b-shaped baseline. Override any field to test invariants."""
    base = dict(
        dataset_dir=Path("/Users/me/.imgen/datasets/alina"),
        scratch_dir=Path("/Users/me/.imgen/loras/.alina.training"),
        lora_name="alina",
        trigger="al1na woman",
        base_model="flux2-klein-4b",
        total_steps=800,
        lora_rank=16,
        max_resolution=512,
        quantize=4,
        low_ram=True,
        optimizer_name="AdamW",
        optimizer_lr=1e-4,
        target_modules=_KLEIN_4B_TARGET_MODULES,
        preview_frequency=100,
        seed=42,
        battery_stop=20,
        output_path=Path("/Users/me/.imgen/loras/alina.safetensors"),
    )
    base.update(overrides)
    return TrainingParams(**base)


# ── shape ────────────────────────────────────────────────────────

class TestTrainingParamsShape:
    def test_minimal_instantiation_succeeds(self):
        params = _minimal_training_params()
        assert params.lora_name == "alina"
        assert params.trigger == "al1na woman"
        assert params.total_steps == 800

    def test_frozen(self):
        """Frozen dataclass — mutation must raise."""
        params = _minimal_training_params()
        with pytest.raises(FrozenInstanceError):
            params.lora_name = "different"

    def test_hashable(self):
        """Hashable + slots — required for set/dict keys + low overhead.
        Mirror :class:`TrainingConfig` discipline (commit 1)."""
        params = _minimal_training_params()
        assert hash(params) == hash(_minimal_training_params())

    def test_slots(self):
        """``__slots__`` defined — instances reject ad-hoc attributes."""
        params = _minimal_training_params()
        with pytest.raises((AttributeError, TypeError)):
            params.extra_field = "x"

    def test_target_modules_is_tuple_not_list(self):
        """The field type is ``tuple[TrainingTargetSpec, ...]`` (hashable)
        — passing a list should TypeError-fail at construction via the
        post-init invariant."""
        with pytest.raises((TypeError, ValueError)):
            _minimal_training_params(target_modules=list(_KLEIN_4B_TARGET_MODULES))


# ── path invariants ──────────────────────────────────────────────

class TestTrainingParamsPathInvariants:
    """Paths must be absolute. cmd_train resolves user paths via
    ``Path.expanduser().resolve()`` before constructing TrainingParams;
    a non-absolute leak here is a bug at the call site."""

    def test_dataset_dir_must_be_absolute(self):
        with pytest.raises(ValueError, match="absolute"):
            _minimal_training_params(dataset_dir=Path("relative/path"))

    def test_scratch_dir_must_be_absolute(self):
        with pytest.raises(ValueError, match="absolute"):
            _minimal_training_params(scratch_dir=Path("relative/scratch"))

    def test_output_path_must_be_absolute(self):
        with pytest.raises(ValueError, match="absolute"):
            _minimal_training_params(output_path=Path("alina.safetensors"))


# ── identifier invariants ────────────────────────────────────────

class TestTrainingParamsIdentifierInvariants:
    def test_lora_name_must_be_nonempty(self):
        with pytest.raises(ValueError, match="lora_name"):
            _minimal_training_params(lora_name="")

    def test_trigger_must_be_nonempty(self):
        with pytest.raises(ValueError, match="trigger"):
            _minimal_training_params(trigger="")

    def test_base_model_must_be_nonempty(self):
        with pytest.raises(ValueError, match="base_model"):
            _minimal_training_params(base_model="")


# ── numeric range invariants (shared with TrainingConfig) ────────

class TestTrainingParamsNumericInvariants:
    def test_total_steps_floor(self):
        with pytest.raises(ValueError, match="total_steps"):
            _minimal_training_params(total_steps=49)

    def test_total_steps_ceiling(self):
        with pytest.raises(ValueError, match="total_steps"):
            _minimal_training_params(total_steps=5001)

    @pytest.mark.parametrize("rank", sorted(_VALID_LORA_RANKS))
    def test_lora_rank_accepts_valid(self, rank):
        params = _minimal_training_params(lora_rank=rank)
        assert params.lora_rank == rank

    @pytest.mark.parametrize("bad", [1, 2, 3, 5, 7, 128])
    def test_lora_rank_rejects_outside_allowlist(self, bad):
        with pytest.raises(ValueError, match="lora_rank"):
            _minimal_training_params(lora_rank=bad)

    @pytest.mark.parametrize("q", sorted(_VALID_QUANTIZE_TRAIN))
    def test_quantize_accepts_valid(self, q):
        params = _minimal_training_params(quantize=q)
        assert params.quantize == q

    @pytest.mark.parametrize("bad", [0, 1, 2, 7, 9, 16])
    def test_quantize_rejects_outside_allowlist(self, bad):
        """``0`` (bf16) excluded — mflux-train doesn't accept it for
        training (§R.1 closure + verified at commit 1)."""
        with pytest.raises(ValueError, match="quantize"):
            _minimal_training_params(quantize=bad)

    @pytest.mark.parametrize("res", sorted(_VALID_TRAIN_RESOLUTIONS))
    def test_max_resolution_accepts_valid(self, res):
        params = _minimal_training_params(max_resolution=res)
        assert params.max_resolution == res

    @pytest.mark.parametrize("bad", [128, 200, 640, 2048])
    def test_max_resolution_rejects_outside_allowlist(self, bad):
        with pytest.raises(ValueError, match="max_resolution"):
            _minimal_training_params(max_resolution=bad)

    def test_optimizer_name_accepts_adamw(self):
        params = _minimal_training_params(optimizer_name="AdamW")
        assert params.optimizer_name == "AdamW"

    def test_optimizer_name_accepts_adafactor(self):
        params = _minimal_training_params(optimizer_name="Adafactor")
        assert params.optimizer_name == "Adafactor"

    def test_optimizer_name_rejects_unknown(self):
        with pytest.raises(ValueError, match="optimizer_name"):
            _minimal_training_params(optimizer_name="SGD")

    @pytest.mark.parametrize("lr", [1e-6, 5e-5, 1e-4, 1e-3, 1e-2])
    def test_optimizer_lr_accepts_in_range(self, lr):
        params = _minimal_training_params(optimizer_lr=lr)
        assert params.optimizer_lr == lr

    @pytest.mark.parametrize("bad", [0.0, 1e-7, 0.1, 1.0])
    def test_optimizer_lr_rejects_outside_range(self, bad):
        """§R.1 python M-5: range tightened to [1e-6, 1e-2] — realistic
        LoRA-finetune envelope."""
        with pytest.raises(ValueError, match="optimizer_lr"):
            _minimal_training_params(optimizer_lr=bad)

    def test_preview_frequency_floor_is_one(self):
        """§M.12 (round-2 N-3): mflux-train rejects
        ``generate_image_frequency=0`` — floor is 1 in TrainingParams
        AND in the argparse ``--preview-every`` validator (commit 4)."""
        with pytest.raises(ValueError, match="preview_frequency"):
            _minimal_training_params(preview_frequency=0)

    def test_preview_frequency_negative_rejected(self):
        with pytest.raises(ValueError, match="preview_frequency"):
            _minimal_training_params(preview_frequency=-1)

    def test_battery_stop_zero_allowed(self):
        """0% = "never stop on battery". Edge of valid range."""
        params = _minimal_training_params(battery_stop=0)
        assert params.battery_stop == 0

    def test_battery_stop_hundred_allowed(self):
        params = _minimal_training_params(battery_stop=100)
        assert params.battery_stop == 100

    def test_battery_stop_negative_rejected(self):
        with pytest.raises(ValueError, match="battery_stop"):
            _minimal_training_params(battery_stop=-1)

    def test_battery_stop_above_100_rejected(self):
        with pytest.raises(ValueError, match="battery_stop"):
            _minimal_training_params(battery_stop=101)

    def test_seed_zero_allowed(self):
        params = _minimal_training_params(seed=0)
        assert params.seed == 0

    def test_seed_max_u32_allowed(self):
        params = _minimal_training_params(seed=2**32 - 1)
        assert params.seed == 2**32 - 1

    def test_seed_negative_rejected(self):
        with pytest.raises(ValueError, match="seed"):
            _minimal_training_params(seed=-1)

    def test_seed_above_u32_rejected(self):
        with pytest.raises(ValueError, match="seed"):
            _minimal_training_params(seed=2**32)


# ── target_modules invariants ────────────────────────────────────

class TestTrainingParamsTargetModules:
    def test_klein_4b_canonical_set_accepted(self):
        params = _minimal_training_params(
            target_modules=_KLEIN_4B_TARGET_MODULES,
        )
        assert params.target_modules == _KLEIN_4B_TARGET_MODULES
        assert len(params.target_modules) == 5

    def test_empty_target_modules_rejected(self):
        """mflux-train requires at least one target — empty tuple is a
        ship-bug-class invariant."""
        with pytest.raises(ValueError, match="target_modules"):
            _minimal_training_params(target_modules=())

    def test_single_target_accepted(self):
        single = (
            TrainingTargetSpec(
                module_path="transformer_blocks.{block}.attn.to_q",
                blocks=(0, 38),
                rank=16,
            ),
        )
        params = _minimal_training_params(target_modules=single)
        assert params.target_modules == single

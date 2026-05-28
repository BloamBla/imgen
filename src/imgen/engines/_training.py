"""v0.10.0 commit 5 — :class:`TrainingParams` + :func:`build_config_json`.

This module owns the **pure** training-side surface that
:meth:`MfluxEngine.train` (commit 7) will consume. Splitting purity
from subprocess dispatch keeps the JSON-shape lock-in testable
without spawning mflux-train.

Per [[project-v100-design]] §E.1 + §R.1 ROUND-1 CLOSURES:

* The original ``MfluxTrainer`` class was DROPPED — training routes
  through ``Engine.train(model, params)`` on the existing Engine
  Protocol (v0.9.5 M-2 registry stays single source of truth).
  ``TrainingParams`` + ``build_config_json`` live as a pure
  dataclass + module-level function pair instead.
* ``lora_layers.targets[]`` JSON shape is
  ``[{module_path, blocks: {start, end} | null, rank}]`` — verified
  against ``mflux/models/common/training/_example/train.json`` +
  ``mflux/models/flux2/training_adapter/flux2_base_training_adapter.py:86-92``.
  NOT the tuple-of-tuples ``[(block_type, target_keys)]`` shape from
  the original draft.
* ``training_loop.num_epochs`` derives from
  ``total_steps // num_entries`` (colleague's recipe: 880 steps / 10
  photos = 88 epochs).
* ``monitoring.generate_image_frequency`` mirrors
  ``params.preview_frequency``; floor 1 (mflux-train rejects 0,
  verified at §M.12).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..models import (
    _VALID_LORA_RANKS,
    _VALID_QUANTIZE_TRAIN,
    _VALID_TRAIN_RESOLUTIONS,
    TrainingTargetSpec,
)

__all__ = ["TrainingParams", "build_config_json"]


# Range constants intentionally mirror :class:`TrainingConfig` floors
# in ``models.py``. Duplicated for clarity at this boundary — the cmd
# layer (commit 8) is responsible for resolving CLI overrides against
# the registry defaults BEFORE constructing TrainingParams, so the
# invariants here are defence-in-depth.
_TOTAL_STEPS_MIN = 50
_TOTAL_STEPS_MAX = 5000
_OPTIMIZER_LR_MIN = 1e-6
_OPTIMIZER_LR_MAX = 1e-2
_OPTIMIZER_NAMES = frozenset({"AdamW", "Adafactor"})
_PREVIEW_FREQUENCY_MIN = 1  # §M.12 — mflux-train rejects 0
_BATTERY_STOP_MIN = 0
_BATTERY_STOP_MAX = 100
_SEED_MIN = 0
_SEED_MAX = 2**32 - 1


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Resolved parameters for one mflux-train invocation. Pure data.

    Built by ``cmd_train`` (commit 8) from CLI args + the base Model's
    :class:`TrainingConfig` defaults; consumed by ``MfluxEngine.train``
    (commit 7) via :func:`build_config_json`.

    All paths MUST be absolute (cmd_train calls
    ``Path.expanduser().resolve()`` upstream). The dataclass is
    frozen + slotted + hashable so it can be stashed in test
    fixtures and dispatch tables without surprise.
    """

    # ── Required identifiers ─────────────────────────────────────
    dataset_dir: Path            # user-supplied source dataset
    scratch_dir: Path            # imgen-materialised mflux-train workspace
    lora_name: str               # ~/.imgen/loras/<lora_name>.safetensors
    trigger: str                 # auto-prepended to inference prompts
    base_model: str              # mflux-train --model value
    output_path: Path            # final .safetensors atomic-rename target

    # ── Training run params ──────────────────────────────────────
    total_steps: int
    lora_rank: int               # in _VALID_LORA_RANKS
    max_resolution: int          # in _VALID_TRAIN_RESOLUTIONS
    quantize: int                # in _VALID_QUANTIZE_TRAIN
    low_ram: bool

    # ── Optimizer (AdamW lr=1e-4 colleague-validated default) ────
    optimizer_name: str          # in _OPTIMIZER_NAMES
    optimizer_lr: float          # 1e-6 .. 1e-2

    # ── Real mflux JSON `lora_layers.targets[]` shape ────────────
    target_modules: tuple[TrainingTargetSpec, ...]

    # ── Monitoring + run-control ─────────────────────────────────
    preview_frequency: int       # >= 1 (mflux-train rejects 0)
    seed: int                    # 0..2^32-1
    battery_stop: int            # 0..100

    def __post_init__(self) -> None:
        # ── Paths absolute ──
        for field_name in ("dataset_dir", "scratch_dir", "output_path"):
            value: Path = getattr(self, field_name)
            if not value.is_absolute():
                raise ValueError(
                    f"TrainingParams.{field_name}={value!r} must be "
                    "an absolute path — cmd_train resolves user paths "
                    "via Path.expanduser().resolve() upstream."
                )

        # ── Non-empty identifiers ──
        if not self.lora_name:
            raise ValueError("TrainingParams.lora_name is empty")
        if not self.trigger:
            raise ValueError("TrainingParams.trigger is empty")
        if not self.base_model:
            raise ValueError("TrainingParams.base_model is empty")

        # ── Numeric ranges + allowlists ──
        if not (_TOTAL_STEPS_MIN <= self.total_steps <= _TOTAL_STEPS_MAX):
            raise ValueError(
                f"TrainingParams.total_steps={self.total_steps} "
                f"out of [{_TOTAL_STEPS_MIN}, {_TOTAL_STEPS_MAX}]"
            )
        if self.lora_rank not in _VALID_LORA_RANKS:
            raise ValueError(
                f"TrainingParams.lora_rank={self.lora_rank} not in "
                f"_VALID_LORA_RANKS={sorted(_VALID_LORA_RANKS)!r}"
            )
        if self.quantize not in _VALID_QUANTIZE_TRAIN:
            raise ValueError(
                f"TrainingParams.quantize={self.quantize} not in "
                f"_VALID_QUANTIZE_TRAIN={sorted(_VALID_QUANTIZE_TRAIN)!r} "
                "(note 0 NOT in set — mflux-train doesn't accept bf16)"
            )
        if self.max_resolution not in _VALID_TRAIN_RESOLUTIONS:
            raise ValueError(
                f"TrainingParams.max_resolution={self.max_resolution} "
                f"not in _VALID_TRAIN_RESOLUTIONS="
                f"{sorted(_VALID_TRAIN_RESOLUTIONS)!r}"
            )

        if self.optimizer_name not in _OPTIMIZER_NAMES:
            raise ValueError(
                f"TrainingParams.optimizer_name={self.optimizer_name!r} "
                f"not in {sorted(_OPTIMIZER_NAMES)!r}"
            )
        if not (_OPTIMIZER_LR_MIN <= self.optimizer_lr <= _OPTIMIZER_LR_MAX):
            raise ValueError(
                f"TrainingParams.optimizer_lr={self.optimizer_lr} "
                f"out of [{_OPTIMIZER_LR_MIN}, {_OPTIMIZER_LR_MAX}] — "
                "realistic LoRA-finetune envelope per §R.1 python M-5."
            )

        # ── target_modules: real mflux shape ──
        # The field type is ``tuple[TrainingTargetSpec, ...]``.
        # Reject lists explicitly so a future call-site mistake doesn't
        # silently work via tuple-isinstance duck-typing.
        if not isinstance(self.target_modules, tuple):
            raise TypeError(
                "TrainingParams.target_modules must be a tuple "
                "(hashable); got "
                f"{type(self.target_modules).__name__}. Convert via "
                "tuple(...) at the call site."
            )
        if not self.target_modules:
            raise ValueError(
                "TrainingParams.target_modules is empty — mflux-train "
                "requires at least one TrainingTargetSpec. "
                "Klein-4b's validated set lives in module constant "
                "imgen.models._KLEIN_4B_TARGET_MODULES."
            )

        # ── Monitoring + run-control ──
        if self.preview_frequency < _PREVIEW_FREQUENCY_MIN:
            raise ValueError(
                f"TrainingParams.preview_frequency={self.preview_frequency} "
                f"must be >= {_PREVIEW_FREQUENCY_MIN}. mflux-train "
                "rejects ``monitoring.generate_image_frequency=0`` "
                "(verified at §M.12); 'disable previews' is expressed "
                "as a high N (e.g. preview_frequency > total_steps)."
            )
        if not (_SEED_MIN <= self.seed <= _SEED_MAX):
            raise ValueError(
                f"TrainingParams.seed={self.seed} out of "
                f"[{_SEED_MIN}, {_SEED_MAX}] (mflux seed range)"
            )
        if not (_BATTERY_STOP_MIN <= self.battery_stop <= _BATTERY_STOP_MAX):
            raise ValueError(
                f"TrainingParams.battery_stop={self.battery_stop} "
                f"out of [{_BATTERY_STOP_MIN}, {_BATTERY_STOP_MAX}]"
            )


def build_config_json(
    params: TrainingParams,
    num_entries: int,
) -> dict:
    """Returns the dict to pass through ``json.dumps`` for
    ``mflux-train --config <FILE>``.

    Pure: no FS I/O, no mutation of ``params``. Each call returns a
    fresh top-level dict (defence-in-depth — callers that mutate the
    returned dict don't surprise subsequent callers).

    The schema is locked by ``tests/test_build_config_json.py`` to
    avoid silent drift. Verified against
    ``mflux/models/common/training/_example/train.json`` and the
    klein-4b training adapter source.

    ``num_entries`` is the dataset size (after validation). It feeds
    ``training_loop.num_epochs = total_steps // num_entries``. A
    zero-entry dataset would normally be rejected upstream by
    :func:`validate_dataset_dir`; the ``max(1, num_entries)`` floor is
    defence-in-depth against ZeroDivisionError if a test fixture
    passes 0.
    """
    safe_num_entries = max(1, num_entries)
    return {
        # Top-level run-shape.
        "model": params.base_model,
        "data": str(params.scratch_dir / "data"),
        "seed": params.seed,
        "steps": params.total_steps,
        "guidance": 0.0,  # klein-4b distilled: training-time CFG off
        "quantize": params.quantize,
        "max_resolution": params.max_resolution,
        "low_ram": params.low_ram,

        # Training loop. num_epochs derives from total_steps /
        # num_entries — colleague's 880/10 = 88-epoch recipe scales
        # linearly.
        "training_loop": {
            # Floor at 1: a small --steps with a large dataset (e.g.
            # --steps 50 over 60 images) would otherwise floor-divide to
            # num_epochs=0 → mflux-train does zero passes / errors deep
            # in setup. At least one epoch always runs.
            "num_epochs": max(1, params.total_steps // safe_num_entries),
            "batch_size": 1,  # v0.10.0 locked; no batching surface
            "timestep_low": 1,
            "timestep_high": params.total_steps,
        },

        # Optimizer (AdamW lr=1e-4 default, colleague-validated).
        "optimizer": {
            "name": params.optimizer_name,
            "learning_rate": params.optimizer_lr,
        },

        # Checkpoint cadence: 8 evenly-spaced saves across the run,
        # floored at 50 steps to avoid thrash on tiny step counts.
        "checkpoint": {
            "save_frequency": max(50, params.total_steps // 8),
            "output_path": str(params.scratch_dir / "checkpoints"),
        },

        # Monitoring. preview_frequency drives BOTH plot_frequency
        # and generate_image_frequency — single CLI knob in imgen.
        "monitoring": {
            "preview_width": 512,
            "preview_height": 512,
            "plot_frequency": params.preview_frequency,
            "generate_image_frequency": params.preview_frequency,
        },

        # Real mflux JSON `lora_layers.targets[]` shape:
        # [{module_path, blocks: {start, end} | null, rank}, ...].
        "lora_layers": {
            "targets": [
                {
                    "module_path": t.module_path,
                    "blocks": (
                        {"start": t.blocks[0], "end": t.blocks[1]}
                        if t.blocks is not None
                        else None
                    ),
                    "rank": t.rank,
                }
                for t in params.target_modules
            ],
        },
    }

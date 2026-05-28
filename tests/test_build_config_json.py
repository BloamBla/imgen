"""v0.10.0 commit 5 — :func:`build_config_json` lock-in.

The dict returned by :func:`build_config_json` flows directly through
``json.dumps(...)`` into the ``mflux-train --config <FILE>`` invocation
at commit 7. A silent schema drift here would either:

1. Break dry-run validation (mflux-train rejects the JSON before
   training starts — recoverable, but a regression for the user).
2. Pass dry-run but produce a different training run (different
   ``num_epochs``, wrong ``lora_layers.targets[]`` shape, missing
   ``low_ram`` flag — non-recoverable; user discovers the deviation
   only after a 10-hour run).

These tests pin the schema by asserting the EXACT dict structure for
the colleague's recipe (the only validated-end-to-end shape we have:
flux2-klein-4b, 10 photos, 880 steps, rank 16, q4, low_ram=True,
preview_frequency=100, AdamW lr=1e-4 — see
[[project-colleague-lora-training-2026-05-27]]). Any future shape
change MUST update the expected dict in lockstep with the schema
change.

Per [[project-v100-design]] §E.1 + §R.1 ROUND-1 CLOSURES — verified
real ``lora_layers.targets[]`` shape from
``mflux/models/common/training/_example/train.json`` +
``mflux/models/flux2/training_adapter/flux2_base_training_adapter.py:86-92``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.engines._training import TrainingParams, build_config_json
from imgen.models import _KLEIN_4B_TARGET_MODULES, TrainingTargetSpec


def _klein_4b_params(**overrides):
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


class TestBuildConfigJsonTopLevelKeys:
    """Top-level mflux-train JSON keys (matches
    ``mflux/models/common/training/_example/train.json``)."""

    def test_returns_dict(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert isinstance(config, dict)

    def test_has_expected_top_level_keys(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        expected = {
            "model", "data", "seed", "steps", "guidance",
            "quantize", "max_resolution", "low_ram",
            "training_loop", "optimizer", "checkpoint",
            "monitoring", "lora_layers",
        }
        assert set(config.keys()) == expected, (
            f"Schema drift: extra={set(config) - expected!r} "
            f"missing={expected - set(config)!r}"
        )


class TestBuildConfigJsonScalarFields:
    def test_model_field(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["model"] == "flux2-klein-4b"

    def test_data_field_points_to_scratch_data_subdir(self):
        """mflux-train discovers images under <data>/; imgen materialises
        a scratch copy so the user's source dataset stays read-only."""
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["data"] == (
            "/Users/me/.imgen/loras/.alina.training/data"
        )

    def test_seed_field(self):
        config = build_config_json(_klein_4b_params(seed=123), num_entries=10)
        assert config["seed"] == 123

    def test_steps_is_schedule_length_decoupled_from_total_steps(self):
        """mflux's ``steps`` = diffusion SCHEDULE length (klein-4b inference
        count, _TRAIN_SCHEDULE_STEPS) — NOT the training length, which is
        num_epochs (driven by total_steps). The original code conflated
        them (steps=total_steps → 800-step monitoring previews, ~34 min
        each, hours wasted). steps must stay the small constant regardless
        of total_steps, and respect mflux's timestep_high <= steps."""
        from imgen.engines._training import _TRAIN_SCHEDULE_STEPS
        for total in (50, 800, 5000):
            config = build_config_json(
                _klein_4b_params(total_steps=total), num_entries=10,
            )
            assert config["steps"] == _TRAIN_SCHEDULE_STEPS
            assert config["training_loop"]["timestep_high"] <= config["steps"]

    def test_guidance_is_zero_for_klein_4b_distilled(self):
        """klein-4b is a distilled model — training-time CFG is OFF
        (guidance=0.0). Locked at v0.10.0 because klein-4b is the only
        training base; a later base may need guidance > 0."""
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["guidance"] == 0.0

    @pytest.mark.parametrize("q", [3, 4, 5, 6, 8])
    def test_quantize_field(self, q):
        config = build_config_json(
            _klein_4b_params(quantize=q), num_entries=10,
        )
        assert config["quantize"] == q

    def test_max_resolution_field(self):
        config = build_config_json(
            _klein_4b_params(max_resolution=768), num_entries=10,
        )
        assert config["max_resolution"] == 768

    def test_low_ram_true(self):
        config = build_config_json(
            _klein_4b_params(low_ram=True), num_entries=10,
        )
        assert config["low_ram"] is True

    def test_low_ram_false(self):
        config = build_config_json(
            _klein_4b_params(low_ram=False), num_entries=10,
        )
        assert config["low_ram"] is False


class TestBuildConfigJsonTrainingLoop:
    """``training_loop.num_epochs`` derives from
    ``total_steps // num_entries`` so the colleague's recipe of
    880 steps × 10 photos lands at num_epochs=88 in the JSON
    (mflux-train then iterates that many times across the dataset)."""

    def test_training_loop_keys(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert set(config["training_loop"]) == {
            "num_epochs", "batch_size", "timestep_low", "timestep_high",
        }

    def test_num_epochs_derives_from_steps_div_entries(self):
        config = build_config_json(
            _klein_4b_params(total_steps=880), num_entries=10,
        )
        assert config["training_loop"]["num_epochs"] == 88

    def test_num_epochs_floor_uses_max_one_entry(self):
        """Guard against ZeroDivisionError on empty dataset (the upstream
        validator rejects empty datasets, so this is defence-in-depth)."""
        config = build_config_json(
            _klein_4b_params(total_steps=800), num_entries=0,
        )
        assert config["training_loop"]["num_epochs"] == 800

    def test_num_epochs_floored_to_one_when_steps_below_entries(self):
        """§R.3 python MEDIUM: --steps below the dataset size (e.g. the
        50-step floor over a 60-image dataset) would floor-divide to
        num_epochs=0 → mflux-train does zero passes / errors deep in
        setup. The max(1, ...) numerator floor keeps at least one epoch."""
        config = build_config_json(
            _klein_4b_params(total_steps=50), num_entries=60,
        )
        assert config["training_loop"]["num_epochs"] == 1

    def test_batch_size_is_one(self):
        """v0.10.0: batch_size locked to 1 — colleague-validated; no
        batching surface on the CLI."""
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["training_loop"]["batch_size"] == 1

    def test_timestep_low_is_one(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["training_loop"]["timestep_low"] == 1

    def test_timestep_high_equals_schedule_steps(self):
        """timestep_high = the schedule length (_TRAIN_SCHEDULE_STEPS),
        NOT total_steps — full range [1, steps], and satisfies mflux's
        timestep_high <= steps regardless of the user's --steps."""
        from imgen.engines._training import _TRAIN_SCHEDULE_STEPS
        config = build_config_json(
            _klein_4b_params(total_steps=800), num_entries=10,
        )
        assert config["training_loop"]["timestep_high"] == _TRAIN_SCHEDULE_STEPS
        assert config["training_loop"]["timestep_high"] <= config["steps"]


class TestBuildConfigJsonOptimizer:
    def test_optimizer_keys(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert set(config["optimizer"]) == {"name", "learning_rate"}

    def test_optimizer_name_propagates(self):
        config = build_config_json(
            _klein_4b_params(optimizer_name="AdamW"), num_entries=10,
        )
        assert config["optimizer"]["name"] == "AdamW"

    def test_optimizer_lr_propagates(self):
        config = build_config_json(
            _klein_4b_params(optimizer_lr=5e-5), num_entries=10,
        )
        assert config["optimizer"]["learning_rate"] == 5e-5


class TestBuildConfigJsonCheckpoint:
    def test_checkpoint_keys(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert set(config["checkpoint"]) == {
            "save_frequency", "output_path",
        }

    def test_save_frequency_derives_from_total_steps(self):
        """8 evenly-spaced saves over the run, floored at 50 to avoid
        thrash on tiny step counts (50-step minimum bracket)."""
        config = build_config_json(
            _klein_4b_params(total_steps=800), num_entries=10,
        )
        assert config["checkpoint"]["save_frequency"] == 100

    def test_save_frequency_floor_fifty(self):
        config = build_config_json(
            _klein_4b_params(total_steps=200), num_entries=10,
        )
        # 200 // 8 = 25, floor 50
        assert config["checkpoint"]["save_frequency"] == 50

    def test_output_path_points_to_scratch_checkpoints_dir(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["checkpoint"]["output_path"] == (
            "/Users/me/.imgen/loras/.alina.training/checkpoints"
        )


class TestBuildConfigJsonMonitoring:
    """mflux-train's ``monitoring`` section drives preview-image
    generation during training. ``generate_image_frequency`` is what
    drives wall-time inflation (each preview is a full inference pass)
    — colleague's recipe of 10 doubled wall-time vs imgen default 100.

    §M.12 (round-2 N-3) verified mflux-train REJECTS
    ``generate_image_frequency=0`` — the TrainingParams invariant
    floors it at 1."""

    def test_monitoring_keys(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert set(config["monitoring"]) == {
            "preview_width", "preview_height",
            "plot_frequency", "generate_image_frequency",
        }

    def test_preview_dimensions_locked_at_512(self):
        """v0.10.0: preview dims locked at 512×512 — small enough to
        complete in seconds, large enough to judge LoRA quality."""
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert config["monitoring"]["preview_width"] == 512
        assert config["monitoring"]["preview_height"] == 512

    def test_plot_frequency_mirrors_preview_frequency(self):
        config = build_config_json(
            _klein_4b_params(preview_frequency=50), num_entries=10,
        )
        assert config["monitoring"]["plot_frequency"] == 50

    def test_generate_image_frequency_mirrors_preview_frequency(self):
        config = build_config_json(
            _klein_4b_params(preview_frequency=50), num_entries=10,
        )
        assert config["monitoring"]["generate_image_frequency"] == 50


class TestBuildConfigJsonLoraLayersShape:
    """§R.1 closure: ``lora_layers.targets[]`` shape verified against
    real mflux: each entry is
    ``{module_path, blocks: {start, end} | null, rank}`` —
    NOT the tuple-of-tuples ``[(block_type, target_keys)]`` shape from
    the original draft. Schema correctness is the lock-in here."""

    def test_lora_layers_has_targets_key_only(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert set(config["lora_layers"]) == {"targets"}

    def test_targets_is_list(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert isinstance(config["lora_layers"]["targets"], list)

    def test_target_count_matches_klein_4b_constant(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        assert len(config["lora_layers"]["targets"]) == len(
            _KLEIN_4B_TARGET_MODULES,
        )

    def test_target_entry_keys_real_shape(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        for t in config["lora_layers"]["targets"]:
            assert set(t) == {"module_path", "blocks", "rank"}

    def test_target_module_paths_preserve_constant_order(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        emitted = [t["module_path"] for t in config["lora_layers"]["targets"]]
        expected = [t.module_path for t in _KLEIN_4B_TARGET_MODULES]
        assert emitted == expected

    def test_blocks_as_dict_when_present(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        first = config["lora_layers"]["targets"][0]
        # First target = transformer_blocks (double-stream); klein-4b has
        # num_layers=5 so the range is (0, 5), end exclusive.
        assert first["blocks"] == {"start": 0, "end": 5}

    def test_blocks_as_none_when_target_has_no_blocks(self):
        """A target without ``{block}`` placeholder + ``blocks=None``
        renders ``"blocks": null`` in the JSON."""
        blockless = (
            TrainingTargetSpec(
                module_path="all_final_layer.linear",
                blocks=None,
                rank=16,
            ),
        )
        config = build_config_json(
            _klein_4b_params(target_modules=blockless), num_entries=10,
        )
        first = config["lora_layers"]["targets"][0]
        assert first["blocks"] is None

    def test_rank_propagates_per_target(self):
        config = build_config_json(_klein_4b_params(), num_entries=10)
        for emitted, source in zip(
            config["lora_layers"]["targets"],
            _KLEIN_4B_TARGET_MODULES,
        ):
            assert emitted["rank"] == source.rank


class TestBuildConfigJsonPurity:
    """Pure function — no FS writes, no mutation of inputs, callable
    repeatedly."""

    def test_repeated_calls_return_equal_dicts(self):
        params = _klein_4b_params()
        c1 = build_config_json(params, num_entries=10)
        c2 = build_config_json(params, num_entries=10)
        assert c1 == c2

    def test_returns_new_dict_each_call(self):
        """Distinct object identity — caller can mutate without
        affecting subsequent calls (defence-in-depth even if we never
        actually mutate)."""
        params = _klein_4b_params()
        c1 = build_config_json(params, num_entries=10)
        c2 = build_config_json(params, num_entries=10)
        assert c1 is not c2

    def test_does_not_mutate_target_modules_tuple(self):
        params = _klein_4b_params()
        before = params.target_modules
        build_config_json(params, num_entries=10)
        assert params.target_modules == before


class TestBuildConfigJsonRoundTripsThroughJsonDumps:
    """Spot-check JSON serialisability — every value must be a primitive
    that ``json.dumps`` accepts without ``default=`` shim."""

    def test_json_dumps_succeeds(self):
        import json

        config = build_config_json(_klein_4b_params(), num_entries=10)
        text = json.dumps(config)
        re_parsed = json.loads(text)
        # Top-level structure preserved after round-trip.
        assert re_parsed["model"] == "flux2-klein-4b"
        assert re_parsed["training_loop"]["num_epochs"] == 80
        assert (
            re_parsed["lora_layers"]["targets"][0]["module_path"]
            == "transformer_blocks.{block}.attn.to_q"
        )

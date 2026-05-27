"""v0.10.0 commit 8 — `cmd_train` 12-step pipeline integration tests.

Covers the orchestration tying together commits 3-7 + the helpers
landing here (preflight, dry-run, history append). Heavy mocking
strategy:

* ``MfluxEngine.train`` patched to skip subprocess; tests assert on
  whether it was called + with what.
* ``prompt_yes_no`` patched per scenario (accept / decline).
* ``checks.get_memory_gb`` / ``checks.get_battery`` patched to feed
  RAM / battery state through the preflight gate.
* ``tokens.load_token`` patched to a sentinel so the env-build path
  is exercised without depending on a real ``~/.imgen/hf_token``.

Tests use a temporary ``STATE_DIR`` (via ``monkeypatch.setattr``) so
each test starts with a clean ``~/.imgen/loras/`` and ``history.jsonl``
slate.

Per [[project-v100-design]] §G + §R.1 ROUND-1 CLOSURES.
"""
from __future__ import annotations

import json
import os
import shutil
import stat as stat_module
from pathlib import Path
from types import SimpleNamespace

import pytest


# ── Test fixtures ────────────────────────────────────────────────


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR + dependent paths into a fresh tmp_path
    so each test starts with a clean ~/.imgen/ slate."""
    fake_state = tmp_path / "fake_state"
    fake_state.mkdir(mode=0o700)
    from imgen import paths as paths_module
    monkeypatch.setattr(paths_module, "STATE_DIR", fake_state)
    monkeypatch.setattr(
        paths_module, "HISTORY_FILE", fake_state / "history.jsonl",
    )
    # cmd_train imports STATE_DIR via `from ..paths import STATE_DIR`
    # so we also patch the re-bound name on the commands module.
    from imgen.commands import train as train_module
    # The module imports STATE_DIR inside cmd_train via late import,
    # so the patch on paths_module is sufficient (the late import
    # re-resolves at call time).
    # Also patch history.HISTORY_FILE used by append_history.
    from imgen import history as history_module
    monkeypatch.setattr(
        history_module, "HISTORY_FILE", fake_state / "history.jsonl",
    )
    return fake_state


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a 5-image dataset with caption sidecars."""
    from PIL import Image
    d = tmp_path / "alina_dataset"
    d.mkdir()
    for i in range(5):
        Image.new("RGB", (256, 256), (i * 20, 100, 200)).save(
            d / f"photo{i}.jpg",
        )
        (d / f"photo{i}.txt").write_text(
            f"al1na woman in pose {i}", encoding="utf-8",
        )
    return d


@pytest.fixture
def fake_mflux_train_bin(tmp_path, monkeypatch):
    """Provide a fake mflux-train binary at a patched VENV_BIN so
    MfluxEngine.train's binary-presence check passes."""
    bin_dir = tmp_path / "venv_bin"
    bin_dir.mkdir()
    fake = bin_dir / "mflux-train"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    from imgen import paths
    monkeypatch.setattr(paths, "VENV_BIN", bin_dir)
    return fake


@pytest.fixture
def mock_subprocess(monkeypatch):
    """Patch run_with_stderr_redaction to skip the real spawn.
    Returns a dict that tests can inspect for the spawn shape +
    can override the rc via the ``rc`` key."""
    recorded = {"argv": None, "env": None, "calls": 0, "rc": 0}

    def fake_run(cmd, env, log_file=None, *, stdin_data=None):
        recorded["argv"] = list(cmd)
        recorded["env"] = dict(env)
        recorded["calls"] += 1
        # Simulate what mflux-train would do on success — write a
        # checkpoint that _promote_final_safetensors can pick up.
        if recorded["rc"] == 0:
            # Find the --config arg to locate the scratch dir.
            i = cmd.index("--config")
            config_path = Path(cmd[i + 1])
            scratch = config_path.parent
            ckpt_dir = scratch / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "0000800_adapter.safetensors").write_bytes(
                b"fake-trained-lora-weights",
            )
        return recorded["rc"]

    from imgen import subprocess_helpers as sh
    monkeypatch.setattr(sh, "run_with_stderr_redaction", fake_run)
    return recorded


@pytest.fixture
def patch_preflight(monkeypatch):
    """Default preflight: ample RAM, on AC. Tests override per scenario."""
    # klein-4b training_peak_ram_gb=28.0 + 3.0 safety = 31.0 GB needed.
    # Default fixture provides 32 GB available so preflight passes;
    # individual tests override available_gb to exercise the die path.
    state = {
        "total_gb": 64.0,
        "available_gb": 32.0,
        "battery_pct": 100,
        "on_ac": True,
    }
    from imgen import checks

    def fake_mem():
        return (state["total_gb"], state["available_gb"])

    def fake_bat():
        return (state["battery_pct"], state["on_ac"])

    monkeypatch.setattr(checks, "get_memory_gb", fake_mem)
    monkeypatch.setattr(checks, "get_battery", fake_bat)
    # Also patch the v0.8.2 RAM safety gate inside
    # run_with_stderr_redaction's _assert_safe_ram_or_raise. Tests
    # bypass that gate by patching the whole function via mock_subprocess.
    return state


@pytest.fixture
def patch_token(monkeypatch):
    """Stub HF token loader so build_mflux_env(token=...) sees a
    deterministic value."""
    from imgen import tokens
    monkeypatch.setattr(tokens, "load_token", lambda: "hf_test_token")


@pytest.fixture
def patch_prompt(monkeypatch):
    """Default: confirm gate ACCEPTS. Tests override per scenario."""
    state = {"answer": True, "calls": 0}
    from imgen import cmd_helpers

    def fake_prompt(question="Continue? [y/N] "):
        state["calls"] += 1
        return state["answer"]

    monkeypatch.setattr(cmd_helpers, "prompt_yes_no", fake_prompt)
    return state


def _make_args(
    dataset, name="alina", trigger="al1na woman", **overrides,
):
    """Build a Namespace mirroring the parser stanza output."""
    base = dict(
        dataset=str(dataset),
        name=name,
        trigger=trigger,
        base="flux2-klein-4b",
        steps=None,
        rank=None,
        quantize=None,
        max_resolution=None,
        preview_every=None,
        seed=42,
        battery_stop=20,
        overwrite=False,
        no_open=False,
        yes=False,
        dry_run=False,
        force=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Dry-run branch ───────────────────────────────────────────────


class TestCmdTrainDryRun:
    def test_dry_run_returns_zero(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
        capsys,
    ):
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir, dry_run=True))
        assert rc == 0

    def test_dry_run_no_subprocess_spawned(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
        capsys,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, dry_run=True))
        assert mock_subprocess["calls"] == 0

    def test_dry_run_no_scratch_dir_created(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
        capsys,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, dry_run=True))
        scratch = state_dir / "loras" / ".alina.training"
        assert not scratch.exists()

    def test_dry_run_prints_json_config(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
        capsys,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, dry_run=True))
        out = capsys.readouterr().out
        assert "flux2-klein-4b" in out
        assert "al1na woman" in out
        assert "mflux-train" in out


# ── Collision check ──────────────────────────────────────────────


class TestCmdTrainCollision:
    def test_existing_lora_without_overwrite_dies(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        loras = state_dir / "loras"
        loras.mkdir(mode=0o700)
        (loras / "alina.safetensors").write_bytes(b"old")
        from imgen.commands.train import cmd_train
        with pytest.raises(SystemExit) as exc:
            cmd_train(_make_args(dataset_dir))
        assert exc.value.code == 2

    def test_existing_lora_with_overwrite_proceeds(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        loras = state_dir / "loras"
        loras.mkdir(mode=0o700)
        (loras / "alina.safetensors").write_bytes(b"old")
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir, overwrite=True))
        assert rc == 0
        assert mock_subprocess["calls"] == 1
        # New content from the fake subprocess wrote 0000800 ckpt.
        assert (loras / "alina.safetensors").read_bytes() == (
            b"fake-trained-lora-weights"
        )


# ── Model resolution ─────────────────────────────────────────────


class TestCmdTrainModelResolution:
    def test_non_training_model_rejected(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """flux-dev (inference-only) has model.training=None — die."""
        from imgen.commands.train import cmd_train
        with pytest.raises(SystemExit) as exc:
            cmd_train(_make_args(dataset_dir, base="flux-dev"))
        assert exc.value.code == 2


# ── Preflight ────────────────────────────────────────────────────


class TestCmdTrainPreflight:
    def test_low_ram_dies(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """klein-4b needs 28 + 3 = 31 GB headroom. Provide 20 GB → die."""
        patch_preflight["available_gb"] = 20.0
        from imgen.commands.train import cmd_train
        with pytest.raises(SystemExit) as exc:
            cmd_train(_make_args(dataset_dir))
        assert exc.value.code == 2

    def test_low_ram_with_force_proceeds(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """--force bypasses headroom check (but NOT the 4 GB floor)."""
        patch_preflight["available_gb"] = 20.0
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir, force=True))
        assert rc == 0

    def test_below_absolute_4gb_floor_dies_even_with_force(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        patch_preflight["available_gb"] = 3.0
        from imgen.commands.train import cmd_train
        with pytest.raises(SystemExit) as exc:
            cmd_train(_make_args(dataset_dir, force=True))
        assert exc.value.code == 2

    def test_on_battery_warns_not_dies(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
        capsys,
    ):
        patch_preflight["on_ac"] = False
        patch_preflight["battery_pct"] = 75
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir))
        assert rc == 0
        # ``colors.warn`` writes to stdout (mirror of ``ok`` / ``step``);
        # only ``err`` writes to stderr.
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "battery" in combined.lower()


# ── Confirm gate ─────────────────────────────────────────────────


class TestCmdTrainConfirmGate:
    def test_decline_returns_one(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        patch_prompt["answer"] = False
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir))
        assert rc == 1
        assert mock_subprocess["calls"] == 0

    def test_yes_flag_skips_prompt(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, yes=True))
        assert patch_prompt["calls"] == 0
        assert mock_subprocess["calls"] == 1


# ── Full success path ────────────────────────────────────────────


class TestCmdTrainSuccessPath:
    def test_returns_zero_on_success(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir))
        assert rc == 0

    def test_safetensors_landed_at_output_path(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        output = state_dir / "loras" / "alina.safetensors"
        assert output.is_file()
        assert output.read_bytes() == b"fake-trained-lora-weights"

    def test_meta_json_written_with_lora_compat_group(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        meta_path = state_dir / "loras" / "alina.meta.json"
        assert meta_path.is_file()
        with meta_path.open() as f:
            meta = json.load(f)
        # §R.1 architect H-5 closure
        assert meta["lora_compat_group"] == "flux2-klein-4b"
        assert meta["lora_name"] == "alina"
        assert meta["trigger"] == "al1na woman"
        assert meta["dataset_image_count"] == 5

    def test_scratch_dir_cleaned_on_success(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        scratch = state_dir / "loras" / ".alina.training"
        assert not scratch.exists()

    def test_loras_dir_mode_is_0o700(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """Security C-2: ~/.imgen/loras/ holds PII-bearing trained
        weights; restrict to user."""
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        loras = state_dir / "loras"
        mode = loras.stat().st_mode & 0o777
        assert mode == 0o700

    def test_env_carries_hf_token(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """§R.1 security H-4: env from build_mflux_env(token=load_token())
        — HF_TOKEN must reach the subprocess for klein-4b gated fetch."""
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        assert mock_subprocess["env"]["HF_TOKEN"] == "hf_test_token"

    def test_history_entry_recorded(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        history_file = state_dir / "history.jsonl"
        assert history_file.is_file()
        lines = history_file.read_text().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["command"] == "train"
        assert entry["lora_name"] == "alina"
        assert entry["status"] == "success"
        assert entry["dataset_image_count"] == 5


# ── Failure paths ────────────────────────────────────────────────


class TestCmdTrainFailurePath:
    def test_mflux_train_nonzero_rc_propagates(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        mock_subprocess["rc"] = 42
        from imgen.commands.train import cmd_train
        rc = cmd_train(_make_args(dataset_dir))
        assert rc == 42

    def test_failed_run_keeps_scratch_dir(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        mock_subprocess["rc"] = 42
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        # Scratch dir + data subdir survive for diagnosis.
        scratch = state_dir / "loras" / ".alina.training"
        assert scratch.exists()
        assert (scratch / "data").is_dir()

    def test_failed_run_no_safetensors_at_output(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        mock_subprocess["rc"] = 42
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        output = state_dir / "loras" / "alina.safetensors"
        assert not output.exists()

    def test_failed_run_no_meta_json_at_output(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        mock_subprocess["rc"] = 42
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        meta_path = state_dir / "loras" / "alina.meta.json"
        assert not meta_path.exists()

    def test_failed_run_history_status_fail(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        mock_subprocess["rc"] = 42
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir))
        history_file = state_dir / "history.jsonl"
        lines = history_file.read_text().splitlines()
        entry = json.loads(lines[-1])
        assert entry["status"] == "fail"


# ── KeyboardInterrupt path ───────────────────────────────────────


class TestCmdTrainKeyboardInterrupt:
    def test_keyboard_interrupt_keeps_scratch_and_records_cancelled(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        patch_preflight, patch_token, patch_prompt, monkeypatch,
    ):
        """User Ctrl-C mid-train: scratch kept, history recorded as
        cancelled, exception re-raised so shell sees signal exit."""

        def boom(cmd, env, log_file=None, *, stdin_data=None):
            # Materialise scratch checkpoint dir too so the cleanup
            # path through the finally block is exercised fully —
            # but DON'T write a checkpoint (would fail promote anyway).
            raise KeyboardInterrupt("user pressed Ctrl-C")

        from imgen import subprocess_helpers as sh
        monkeypatch.setattr(sh, "run_with_stderr_redaction", boom)

        from imgen.commands.train import cmd_train
        with pytest.raises(KeyboardInterrupt):
            cmd_train(_make_args(dataset_dir))

        # Scratch must survive for restart.
        scratch = state_dir / "loras" / ".alina.training"
        assert scratch.exists()

        # History entry recorded with cancelled status.
        history_file = state_dir / "history.jsonl"
        lines = history_file.read_text().splitlines()
        entry = json.loads(lines[-1])
        assert entry["status"] == "cancelled"


# ── CLI param resolution ─────────────────────────────────────────


class TestCmdTrainParamResolution:
    def test_default_steps_from_training_config(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        """args.steps=None → total_steps = TrainingConfig.default_epochs
        × num_entries. klein-4b default_epochs=80, 5 photos → 400 steps."""
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, steps=None))
        # The config.json written into scratch (now removed) is the
        # only direct lock-in we have; assert via the history entry
        # total_steps instead.
        history_file = state_dir / "history.jsonl"
        entry = json.loads(history_file.read_text().splitlines()[-1])
        assert entry["total_steps"] == 80 * 5  # 400

    def test_explicit_steps_override(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, steps=1000))
        history_file = state_dir / "history.jsonl"
        entry = json.loads(history_file.read_text().splitlines()[-1])
        assert entry["total_steps"] == 1000

    def test_explicit_rank_override(
        self, state_dir, dataset_dir, fake_mflux_train_bin,
        mock_subprocess, patch_preflight, patch_token, patch_prompt,
    ):
        from imgen.commands.train import cmd_train
        cmd_train(_make_args(dataset_dir, rank=32))
        history_file = state_dir / "history.jsonl"
        entry = json.loads(history_file.read_text().splitlines()[-1])
        assert entry["lora_rank"] == 32


# ── _estimate_wall_hours pure helper ─────────────────────────────


class TestEstimateWallHours:
    def test_returns_float(self):
        from imgen.commands.train import _estimate_wall_hours

        params = SimpleNamespace(total_steps=880)
        assert isinstance(_estimate_wall_hours(params), float)

    def test_scales_linearly_with_steps(self):
        from imgen.commands.train import _estimate_wall_hours

        small = _estimate_wall_hours(SimpleNamespace(total_steps=100))
        big = _estimate_wall_hours(SimpleNamespace(total_steps=1000))
        # 10× steps → 10× wall (within float rounding).
        assert big / small == pytest.approx(10.0, rel=1e-6)

    def test_colleague_recipe_at_880_steps_is_about_seven_hours(self):
        """§K.3 anchor: ~880 steps × 27.5 sec / 3600 ≈ 6.7 h on
        M2 Pro 32 GB (colleague's M5 Pro 48 GB was ~10h with dense
        previews; imgen default sparse previews bring it down)."""
        from imgen.commands.train import _estimate_wall_hours

        hours = _estimate_wall_hours(SimpleNamespace(total_steps=880))
        # 880 × 27.5 / 3600 ≈ 6.722
        assert 6.5 < hours < 7.0

"""v0.10.0 commit 7 — MfluxEngine.train real subprocess invocation.

Covers the engine-level training-side dispatch. The cmd-level
orchestration (materialise scratch → invoke → promote → meta-json)
lands at commit 8; this file pins ONLY the subprocess shape.

Per [[project-v100-design]] §E.1 + §R.1 ROUND-1 CLOSURES:

* Security H-4: subprocess MUST run with ``build_mflux_env(token=...)``
  (NOT ``env=None`` which would inherit DYLD_*/LD_*/PYTHONPATH from
  the parent).
* mflux-train binary at ``VENV_BIN / "mflux-train"`` — lstat-check
  via ``stat.S_ISLNK`` (refuse symlinks) and ``stat.S_ISREG`` (refuse
  dirs/specials). Mirrors v0.4 backends.d + v0.9 .venv-diffusers
  binary-validation discipline.
* ``--battery-percentage-stop-limit`` flag emitted ONLY when
  ``params.battery_stop != 5`` (mflux-train's own default) — keeps
  the argv shape minimal.
* num_entries derived from the materialised scratch dir (count
  non-.txt files in ``<scratch>/data/``). Caller responsible for
  materialising the scratch dataset BEFORE calling Engine.train.
* config.json written to ``<scratch>/config.json`` with mode 0o600
  (PII-bearing dataset_path + trigger live there).
* KeyboardInterrupt propagates unwrapped (mirror MfluxEngine.run
  v0.8.2 architect HIGH-2).
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from imgen.engines._training import TrainingParams, build_config_json
from imgen.engines.mflux_engine import MfluxEngine
from imgen.models import _KLEIN_4B_TARGET_MODULES, BUILTIN_MODELS


def _make_fake_train_binary(tmp_path: Path) -> Path:
    """Create a fake mflux-train binary in a fake VENV_BIN under
    tmp_path. Used for symlink-reject / non-regular-file tests."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "mflux-train"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    return fake


def _materialised_scratch(
    tmp_path: Path, n_images: int = 3,
) -> Path:
    """Create a fake materialised scratch dir matching what
    _materialise_scratch_dataset (commit 6) produces. Each image has
    a companion .txt sidecar so the non-.txt count == n_images."""
    scratch = tmp_path / ".alina.training"
    scratch.mkdir(parents=True)
    data = scratch / "data"
    data.mkdir()
    for i in range(n_images):
        (data / f"photo{i}.png").write_bytes(b"\x00")
        (data / f"photo{i}.txt").write_text(f"caption {i}")
    return scratch


def _klein_4b_params(tmp_path: Path, **overrides) -> TrainingParams:
    scratch = _materialised_scratch(tmp_path, n_images=10)
    defaults = dict(
        dataset_dir=tmp_path / "datasets" / "alina",
        scratch_dir=scratch,
        lora_name="alina",
        trigger="al1na woman",
        base_model="flux2-klein-4b",
        total_steps=880,
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
        output_path=tmp_path / "loras" / "alina.safetensors",
    )
    defaults.update(overrides)
    return TrainingParams(**defaults)


def _install_fake_run(monkeypatch, tmp_path, *, rc: int = 0):
    """Patch run_with_stderr_redaction to record argv + env without
    spawning a real subprocess. Returns the recording dict so tests
    can assert the call shape."""
    recorded = {"argv": None, "env": None, "calls": 0}

    def fake_run(cmd, env, log_file=None, *, stdin_data=None):
        recorded["argv"] = list(cmd)
        recorded["env"] = dict(env)
        recorded["calls"] += 1
        return rc

    # Patch both binding sites: the source module AND any late-import
    # binding inside mflux_engine.
    from imgen import subprocess_helpers as sh
    monkeypatch.setattr(sh, "run_with_stderr_redaction", fake_run)
    from imgen.engines import mflux_engine as me
    # Late import in MfluxEngine.train: also patch the engines module
    # in case it caches the binding.
    monkeypatch.setattr(
        me, "run_with_stderr_redaction", fake_run, raising=False,
    )

    # Point VENV_BIN at a fresh dir with a fake binary.
    fake_train = _make_fake_train_binary(tmp_path)
    from imgen import paths
    monkeypatch.setattr(paths, "VENV_BIN", fake_train.parent)
    return recorded, fake_train


# ── subprocess argv shape ────────────────────────────────────────

class TestMfluxEngineTrainArgvShape:
    def test_returns_rc_zero_on_success(self, tmp_path, monkeypatch):
        recorded, _ = _install_fake_run(monkeypatch, tmp_path, rc=0)
        rc = MfluxEngine().train(
            BUILTIN_MODELS["flux2-klein-4b"],
            _klein_4b_params(tmp_path),
        )
        assert rc == 0

    def test_propagates_nonzero_rc(self, tmp_path, monkeypatch):
        _install_fake_run(monkeypatch, tmp_path, rc=42)
        rc = MfluxEngine().train(
            BUILTIN_MODELS["flux2-klein-4b"],
            _klein_4b_params(tmp_path),
        )
        assert rc == 42

    def test_argv_starts_with_mflux_train_binary(
        self, tmp_path, monkeypatch,
    ):
        recorded, fake_train = _install_fake_run(monkeypatch, tmp_path)
        MfluxEngine().train(
            BUILTIN_MODELS["flux2-klein-4b"],
            _klein_4b_params(tmp_path),
        )
        assert recorded["argv"][0] == str(fake_train)

    def test_argv_passes_config_via_config_flag(
        self, tmp_path, monkeypatch,
    ):
        recorded, _ = _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        config_path = params.scratch_dir / "config.json"
        assert "--config" in recorded["argv"]
        i = recorded["argv"].index("--config")
        assert recorded["argv"][i + 1] == str(config_path)

    def test_battery_stop_default_20_emits_flag(
        self, tmp_path, monkeypatch,
    ):
        """battery_stop=20 (imgen default, ≠ mflux's 5) MUST emit the
        flag so the training run uses imgen's overnight-safer floor."""
        recorded, _ = _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path, battery_stop=20)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        assert "--battery-percentage-stop-limit" in recorded["argv"]
        i = recorded["argv"].index("--battery-percentage-stop-limit")
        assert recorded["argv"][i + 1] == "20"

    def test_battery_stop_5_omits_flag(self, tmp_path, monkeypatch):
        """battery_stop=5 == mflux-train's own default — keep argv
        minimal by omitting the flag."""
        recorded, _ = _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path, battery_stop=5)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        assert "--battery-percentage-stop-limit" not in recorded["argv"]


# ── env construction (security H-4) ─────────────────────────────

class TestMfluxEngineTrainEnv:
    def test_default_env_is_build_mflux_env(
        self, tmp_path, monkeypatch,
    ):
        """§R.1 security H-4: env=None falls back to build_mflux_env()
        (allowlisted keys only) — NEVER inherits parent env wholesale.
        Defence against DYLD_*/LD_*/PYTHONPATH poisoning."""
        recorded, _ = _install_fake_run(monkeypatch, tmp_path)
        MfluxEngine().train(
            BUILTIN_MODELS["flux2-klein-4b"],
            _klein_4b_params(tmp_path),
        )
        # build_mflux_env always sets PATH (allowlisted base) +
        # COLUMNS/LINES (terminal forwarding).
        env = recorded["env"]
        assert "PATH" in env
        assert "COLUMNS" in env
        assert "LINES" in env
        # And does NOT forward DYLD_* / LD_* / PYTHONPATH.
        assert "DYLD_LIBRARY_PATH" not in env
        assert "LD_LIBRARY_PATH" not in env
        assert "PYTHONPATH" not in env

    def test_explicit_env_overrides_default(
        self, tmp_path, monkeypatch,
    ):
        """cmd_train (commit 8) passes
        env=build_mflux_env(token=hf_token) so the gated klein-4b
        repo fetch works on first run. Engine.train must pass it
        through verbatim."""
        recorded, _ = _install_fake_run(monkeypatch, tmp_path)
        custom_env = {
            "PATH": "/usr/bin",
            "HF_TOKEN": "hf_test_token",
            "HOME": "/Users/test",
        }
        MfluxEngine().train(
            BUILTIN_MODELS["flux2-klein-4b"],
            _klein_4b_params(tmp_path),
            env=custom_env,
        )
        assert recorded["env"]["HF_TOKEN"] == "hf_test_token"
        assert recorded["env"]["PATH"] == "/usr/bin"


# ── config.json materialisation ─────────────────────────────────

class TestMfluxEngineTrainConfigJson:
    def test_writes_config_to_scratch_dir(self, tmp_path, monkeypatch):
        _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        config_path = params.scratch_dir / "config.json"
        assert config_path.is_file()

    def test_config_matches_build_config_json(
        self, tmp_path, monkeypatch,
    ):
        """The written config MUST equal build_config_json(params,
        num_entries) — schema lock-in from commit 5 reaches the
        subprocess via this path."""
        _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        with (params.scratch_dir / "config.json").open() as f:
            written = json.load(f)
        # n_images=10 via _materialised_scratch default.
        expected = build_config_json(params, num_entries=10)
        assert written == expected

    def test_config_file_mode_is_0o600(self, tmp_path, monkeypatch):
        """Security C-2: config.json contains dataset_path +
        trigger word (potential PII). Restrict to user."""
        _install_fake_run(monkeypatch, tmp_path)
        params = _klein_4b_params(tmp_path)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        config_path = params.scratch_dir / "config.json"
        mode = config_path.stat().st_mode & 0o777
        assert mode == 0o600, f"config.json mode {oct(mode)} != 0o600"

    def test_num_entries_derived_from_scratch_data_dir(
        self, tmp_path, monkeypatch,
    ):
        """num_entries = count of non-.txt files in scratch_dir/data —
        the materialised dataset is authoritative (caller may have
        rejected images at materialise time we don't know about
        from here)."""
        _install_fake_run(monkeypatch, tmp_path)
        # Override the scratch with a different image count.
        scratch = _materialised_scratch(
            tmp_path / "alt", n_images=7,
        )
        params = _klein_4b_params(tmp_path, scratch_dir=scratch)
        MfluxEngine().train(BUILTIN_MODELS["flux2-klein-4b"], params)
        with (scratch / "config.json").open() as f:
            written = json.load(f)
        # build_config_json: num_epochs = total_steps // num_entries.
        # 880 // 7 = 125.
        assert written["training_loop"]["num_epochs"] == 125


# ── binary validation (security mirror v0.4/v0.9) ───────────────

class TestMfluxEngineTrainBinaryValidation:
    def test_missing_binary_raises(self, tmp_path, monkeypatch):
        """No mflux-train at VENV_BIN — clean error pointing to
        bootstrap.sh."""
        _install_fake_run(monkeypatch, tmp_path)
        # Remove the fake binary.
        (tmp_path / "bin" / "mflux-train").unlink()
        with pytest.raises(SystemExit):
            MfluxEngine().train(
                BUILTIN_MODELS["flux2-klein-4b"],
                _klein_4b_params(tmp_path),
            )

    def test_symlink_binary_rejected(self, tmp_path, monkeypatch):
        """§R.1 security: mflux-train must be a real regular file,
        not a symlink. Symlink would let an attacker hijack the
        binary via a writeable parent dir without being detected by
        an inode/permissions check on the symlink target."""
        _install_fake_run(monkeypatch, tmp_path)
        bin_dir = tmp_path / "bin"
        # Replace the regular file with a symlink to a real binary.
        real = bin_dir / "mflux-train-real"
        (bin_dir / "mflux-train").rename(real)
        (bin_dir / "mflux-train").symlink_to(real)
        with pytest.raises(SystemExit):
            MfluxEngine().train(
                BUILTIN_MODELS["flux2-klein-4b"],
                _klein_4b_params(tmp_path),
            )

    def test_directory_at_binary_path_rejected(
        self, tmp_path, monkeypatch,
    ):
        """If mflux-train is somehow a directory (broken install,
        archive extraction mishap), refuse to exec."""
        _install_fake_run(monkeypatch, tmp_path)
        bin_dir = tmp_path / "bin"
        (bin_dir / "mflux-train").unlink()
        (bin_dir / "mflux-train").mkdir()
        with pytest.raises(SystemExit):
            MfluxEngine().train(
                BUILTIN_MODELS["flux2-klein-4b"],
                _klein_4b_params(tmp_path),
            )


# ── KeyboardInterrupt propagation ────────────────────────────────

class TestMfluxEngineTrainKeyboardInterrupt:
    def test_keyboard_interrupt_propagates_unwrapped(
        self, tmp_path, monkeypatch,
    ):
        """v0.8.2 architect HIGH-2 mirror: when the user Ctrl-Cs
        mid-training, the exception must propagate unwrapped so
        cmd_train (commit 8) can write the cancel marker to history."""
        from imgen import subprocess_helpers as sh

        def boom(*args, **kwargs):
            raise KeyboardInterrupt("user pressed Ctrl-C")

        monkeypatch.setattr(sh, "run_with_stderr_redaction", boom)
        from imgen.engines import mflux_engine as me
        monkeypatch.setattr(
            me, "run_with_stderr_redaction", boom, raising=False,
        )
        # Still need VENV_BIN with valid binary for the pre-spawn
        # checks to pass.
        fake_train = _make_fake_train_binary(tmp_path)
        from imgen import paths
        monkeypatch.setattr(paths, "VENV_BIN", fake_train.parent)

        with pytest.raises(KeyboardInterrupt):
            MfluxEngine().train(
                BUILTIN_MODELS["flux2-klein-4b"],
                _klein_4b_params(tmp_path),
            )

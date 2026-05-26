"""v0.9 commit 6 — ensure_video_deps_or_die + audit marker.

Per [[project-v090-design]] §E.5. The function gates the first
``imgen video`` invocation, lazy-installing the three pinned packages
(imageio, imageio-ffmpeg, sentencepiece) into ``.venv-diffusers/``
the first time the user runs video. Tests pin every safety guard:

* Sentinel file blocks execution if a previous install was interrupted
  (security §R.1 HIGH-3).
* pip / python paths must NOT be symlinks (security §R.1 HIGH-2 —
  same-uid attacker could plant a replacement).
* Missing pip / python → die with bootstrap hint, not raw FileNotFoundError.
* Non-interactive shell without IMGEN_INSTALL_VIDEO_DEPS=1 → die
  rather than block on stdin.
* Env-var bypass → install with audit log, never silent.
* Deps already present → skip install.
* pip install failure → sentinel preserved + die (next invocation
  surfaces the inconsistent state).
* Audit log content includes pinned versions + timestamp.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _setup_fake_venv(tmp_path, monkeypatch, install_pip=True, install_python=True):
    """Build a fake ``.venv-diffusers/`` layout under tmp_path and
    monkeypatch IMGEN_INSTALL_ROOT to point at it. Returns (pip_path,
    python_path, state_dir)."""
    from imgen import paths

    install_root = tmp_path / "install_root"
    venv_bin = install_root / ".venv-diffusers" / "bin"
    venv_bin.mkdir(parents=True)
    pip_path = venv_bin / "pip"
    python_path = venv_bin / "python"
    if install_pip:
        pip_path.write_text("#!/usr/bin/env bash\n")
        pip_path.chmod(0o755)
    if install_python:
        python_path.write_text("#!/usr/bin/env bash\n")
        python_path.chmod(0o755)

    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)
    # tmp_state_dir autouse already monkeypatches STATE_DIR to tmp_path/.imgen
    state_dir = paths.STATE_DIR
    return pip_path, python_path, state_dir


def _stub_subprocess_run(monkeypatch, *, deps_present_rc=0, install_rc=0):
    """Replace subprocess.run with a fake. Returns the list of recorded
    call tuples for inspection.

    Distinguishes between probe calls (``python -c "import ..."``) and
    install calls (``pip install ...``) by argv shape. Returns
    deps_present_rc for probes, install_rc for installs.
    """
    import imgen.commands.video as video_mod
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        # Probe vs install distinguished by `-c` flag (probe) vs
        # `install` arg (pip install).
        if "-c" in argv:
            return subprocess.CompletedProcess(argv, deps_present_rc)
        if "install" in argv:
            return subprocess.CompletedProcess(argv, install_rc)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(video_mod, "subprocess", MagicMock(
        run=fake_run,
        CompletedProcess=subprocess.CompletedProcess,
        DEVNULL=subprocess.DEVNULL,
    ))
    return calls


# ── Sentinel guard (security §R.1 HIGH-3) ─────────────────────────────


class TestSentinelGuard:
    """A leftover sentinel file from a previous interrupted install
    must block further runs — .venv-diffusers/ may be in an
    inconsistent state."""

    def test_sentinel_present_dies(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _, _, state_dir = _setup_fake_venv(tmp_path, monkeypatch)
        state_dir.mkdir(exist_ok=True)
        (state_dir / ".video_deps_installing").touch()

        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "interrupted" in stderr.lower() or "sentinel" in stderr.lower()

    def test_sentinel_message_includes_recovery_hint(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _, _, state_dir = _setup_fake_venv(tmp_path, monkeypatch)
        state_dir.mkdir(exist_ok=True)
        (state_dir / ".video_deps_installing").touch()

        with pytest.raises(SystemExit):
            ensure_video_deps_or_die()
        stderr = capsys.readouterr().err
        assert "bootstrap.sh" in stderr or ".venv-diffusers" in stderr


# ── Symlink guard (security §R.1 HIGH-2) ──────────────────────────────


class TestSymlinkGuard:
    """Same-uid attacker could plant a symlink at pip / python to
    redirect exec. Refuse to proceed."""

    def test_pip_is_symlink_dies(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        pip_path, _, _ = _setup_fake_venv(tmp_path, monkeypatch)
        # Replace pip with a symlink
        pip_path.unlink()
        target = tmp_path / "evil_pip"
        target.write_text("#!/bin/bash\n")
        target.chmod(0o755)
        pip_path.symlink_to(target)

        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "symlink" in stderr.lower()

    def test_python_is_symlink_dies(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _, python_path, _ = _setup_fake_venv(tmp_path, monkeypatch)
        python_path.unlink()
        target = tmp_path / "evil_python"
        target.write_text("#!/bin/bash\n")
        target.chmod(0o755)
        python_path.symlink_to(target)

        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "symlink" in stderr.lower()


# ── Missing venv guard ────────────────────────────────────────────────


class TestMissingVenvGuard:
    """pip / python not found → user hasn't opted into .venv-diffusers/
    via bootstrap. Direct them to bootstrap rather than crashing."""

    def test_pip_missing_dies_with_bootstrap_hint(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _setup_fake_venv(tmp_path, monkeypatch, install_pip=False)
        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "bootstrap" in stderr.lower()

    def test_python_missing_dies_with_bootstrap_hint(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _setup_fake_venv(tmp_path, monkeypatch, install_python=False)
        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "bootstrap" in stderr.lower()


# ── Happy path: deps already present ──────────────────────────────────


class TestDepsAlreadyPresent:
    """If imageio + imageio_ffmpeg + sentencepiece all import cleanly
    in .venv-diffusers/, ensure_video_deps_or_die returns silently —
    no prompt, no install, no audit write (the audit was written by
    the first install)."""

    def test_returns_without_prompt_when_deps_present(
        self, tmp_path, monkeypatch, capsys,
    ):
        from imgen.commands.video import ensure_video_deps_or_die
        _setup_fake_venv(tmp_path, monkeypatch)
        _stub_subprocess_run(monkeypatch, deps_present_rc=0)
        # No stdin needed — we shouldn't reach the prompt
        ensure_video_deps_or_die()
        # No "Install now" message in stderr/stdout
        out = capsys.readouterr()
        assert "Install now" not in out.out
        assert "Install now" not in out.err


# ── Non-TTY guard (no stdin to prompt) ────────────────────────────────


class TestNonTTYGuard:
    """A pipe / cron / CI invocation has no TTY — must NOT block on
    stdin.read(). Either env-var-opt-in OR die."""

    def test_non_tty_no_env_var_dies(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.video import ensure_video_deps_or_die
        _setup_fake_venv(tmp_path, monkeypatch)
        _stub_subprocess_run(monkeypatch, deps_present_rc=1)  # deps missing

        monkeypatch.delenv("IMGEN_INSTALL_VIDEO_DEPS", raising=False)
        # Force non-TTY
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2
        stderr = capsys.readouterr().err
        assert "IMGEN_INSTALL_VIDEO_DEPS" in stderr


# ── Env-var bypass (always with audit) ────────────────────────────────


class TestEnvVarBypass:
    """IMGEN_INSTALL_VIDEO_DEPS=1 → install without prompt BUT print
    audit line on stderr (never silent). Mirrors bootstrap.sh
    IMGEN_INSTALL_DIFFUSERS pattern."""

    def test_env_var_set_installs_without_prompt(
        self, tmp_path, monkeypatch, capsys,
    ):
        from imgen.commands.video import ensure_video_deps_or_die
        pip_path, _, _ = _setup_fake_venv(tmp_path, monkeypatch)
        # First probe returns "missing" (rc=1); after install, the
        # post-install verify also probes — return 0 then.
        calls = []
        import imgen.commands.video as video_mod
        probe_count = {"n": 0}

        def fake_run(argv, **kwargs):
            calls.append(tuple(argv))
            if "-c" in argv:
                probe_count["n"] += 1
                # First probe: deps missing. Second probe (post-install
                # verify): deps present.
                return subprocess.CompletedProcess(
                    argv, 1 if probe_count["n"] == 1 else 0,
                )
            if "install" in argv:
                return subprocess.CompletedProcess(argv, 0)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(video_mod, "subprocess", MagicMock(
            run=fake_run,
            CompletedProcess=subprocess.CompletedProcess,
            DEVNULL=subprocess.DEVNULL,
        ))

        monkeypatch.setenv("IMGEN_INSTALL_VIDEO_DEPS", "1")
        # isatty doesn't matter for env-var bypass but pin to False to
        # prove the bypass works in non-interactive contexts too.
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        ensure_video_deps_or_die()
        stderr = capsys.readouterr().err
        # Audit line on stderr — never silent
        assert "IMGEN_INSTALL_VIDEO_DEPS" in stderr or "auto-installing" in stderr.lower()
        # An install call was made
        install_calls = [c for c in calls if "install" in c]
        assert install_calls, "no pip install call made under env-var bypass"


class TestPinnedVersions:
    """The exact pinned versions land in the pip argv — no wildcards
    that could let a typo-squat sneak through."""

    def test_pinned_versions_constant_shape(self):
        from imgen.commands.video import _VIDEO_DEPS_PINNED
        assert "imageio==2.37.3" in _VIDEO_DEPS_PINNED
        assert "imageio-ffmpeg==0.6.0" in _VIDEO_DEPS_PINNED
        assert "sentencepiece==0.2.1" in _VIDEO_DEPS_PINNED

    def test_pinned_versions_exact_equality_no_wildcards(self):
        """Every pin uses `==`, never `>=` / `~=` / wildcards."""
        from imgen.commands.video import _VIDEO_DEPS_PINNED
        for pin in _VIDEO_DEPS_PINNED:
            assert "==" in pin, f"pin {pin!r} not exactly versioned"
            assert "*" not in pin
            assert ">=" not in pin
            assert "~=" not in pin


# ── Audit marker content + sentinel removal on success ───────────────


class TestAuditAndSentinelLifecycle:

    def _stub_install_success(self, monkeypatch):
        """Probe rc=1 first (missing), then rc=0 (verify); install rc=0."""
        import imgen.commands.video as video_mod
        probe_count = {"n": 0}

        def fake_run(argv, **kwargs):
            if "-c" in argv:
                probe_count["n"] += 1
                return subprocess.CompletedProcess(
                    argv, 1 if probe_count["n"] == 1 else 0,
                )
            if "install" in argv:
                return subprocess.CompletedProcess(argv, 0)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(video_mod, "subprocess", MagicMock(
            run=fake_run,
            CompletedProcess=subprocess.CompletedProcess,
            DEVNULL=subprocess.DEVNULL,
        ))

    def test_audit_marker_written_on_success(self, tmp_path, monkeypatch):
        from imgen.commands.video import ensure_video_deps_or_die
        _, _, state_dir = _setup_fake_venv(tmp_path, monkeypatch)
        self._stub_install_success(monkeypatch)
        monkeypatch.setenv("IMGEN_INSTALL_VIDEO_DEPS", "1")

        ensure_video_deps_or_die()

        marker = state_dir / "video_deps_installed_at.txt"
        assert marker.exists(), "audit marker not written"
        content = marker.read_text()
        assert "imageio==2.37.3" in content
        assert "imageio-ffmpeg==0.6.0" in content
        assert "sentencepiece==0.2.1" in content

    def test_sentinel_removed_after_success(self, tmp_path, monkeypatch):
        from imgen.commands.video import ensure_video_deps_or_die
        _, _, state_dir = _setup_fake_venv(tmp_path, monkeypatch)
        self._stub_install_success(monkeypatch)
        monkeypatch.setenv("IMGEN_INSTALL_VIDEO_DEPS", "1")

        ensure_video_deps_or_die()

        sentinel = state_dir / ".video_deps_installing"
        assert not sentinel.exists(), (
            "sentinel must be removed after successful install"
        )


class TestPipInstallFailure:

    def test_pip_install_failure_preserves_sentinel(self, tmp_path, monkeypatch, capsys):
        """Per §E.5.5: if pip install returns non-zero, sentinel stays
        so the next invocation surfaces the inconsistent state."""
        from imgen.commands.video import ensure_video_deps_or_die
        _, _, state_dir = _setup_fake_venv(tmp_path, monkeypatch)

        import imgen.commands.video as video_mod

        def fake_run(argv, **kwargs):
            if "-c" in argv:
                # deps missing, never verified successful
                return subprocess.CompletedProcess(argv, 1)
            if "install" in argv:
                return subprocess.CompletedProcess(argv, 1)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(video_mod, "subprocess", MagicMock(
            run=fake_run,
            CompletedProcess=subprocess.CompletedProcess,
            DEVNULL=subprocess.DEVNULL,
        ))

        monkeypatch.setenv("IMGEN_INSTALL_VIDEO_DEPS", "1")

        with pytest.raises(SystemExit) as exc:
            ensure_video_deps_or_die()
        assert exc.value.code == 2

        sentinel = state_dir / ".video_deps_installing"
        assert sentinel.exists(), (
            "sentinel must remain after pip install failure"
        )


# ── Bypass v0.8.2 RAM safety net for pip install (§E.5.6) ──────────────


class TestRamSafetyNetBypass:
    """§E.5.6: pip install must NOT route through
    subprocess_helpers.run_with_stderr_redaction (which has the
    v0.8.2 < 4 GB safety net). pip install has a fundamentally
    different RAM profile and the safety net would block legitimate
    installs on a memory-tight Mac."""

    def test_pip_install_uses_plain_subprocess_run(
        self, tmp_path, monkeypatch, capsys,
    ):
        from imgen.commands.video import ensure_video_deps_or_die
        _setup_fake_venv(tmp_path, monkeypatch)

        # Pin the RAM safety net to refuse anything — if pip install
        # routed through it, the install would die. Plain
        # subprocess.run bypasses this gate entirely.
        from imgen import subprocess_helpers
        called = {"safety_invoked": False}

        def trip_wire(*a, **k):
            called["safety_invoked"] = True
            raise AssertionError(
                "pip install must NOT route through "
                "run_with_stderr_redaction (RAM safety net bypass)"
            )

        monkeypatch.setattr(
            subprocess_helpers, "run_with_stderr_redaction", trip_wire,
        )

        import imgen.commands.video as video_mod
        probe_count = {"n": 0}

        def fake_run(argv, **kwargs):
            if "-c" in argv:
                probe_count["n"] += 1
                return subprocess.CompletedProcess(
                    argv, 1 if probe_count["n"] == 1 else 0,
                )
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr(video_mod, "subprocess", MagicMock(
            run=fake_run,
            CompletedProcess=subprocess.CompletedProcess,
            DEVNULL=subprocess.DEVNULL,
        ))

        monkeypatch.setenv("IMGEN_INSTALL_VIDEO_DEPS", "1")
        ensure_video_deps_or_die()
        assert not called["safety_invoked"]

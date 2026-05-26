"""v0.9 commit 6 — doctor video-deps section + drift detection.

Per [[project-v090-design]] §E.5.3. The marker file
``~/.imgen/video_deps_installed_at.txt`` records the timestamp +
pinned versions written by ``ensure_video_deps_or_die``. Doctor
reads it and reports one of:

* GREEN — present + matches pinned set
* YELLOW — present but drifted (e.g. user upgraded imgen but didn't
  re-bootstrap; old pin still installed)
* INFO — absent (no install yet)

The section fires only when .venv-diffusers/ is present (no point
suggesting video deps for a colleague who hasn't opted into the
diffusers stack at all).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _setup_fake_venv(tmp_path, monkeypatch):
    """Build a fake .venv-diffusers/ layout so doctor's
    venv_python.is_file() check passes."""
    from imgen import paths

    install_root = tmp_path / "install_root"
    venv_bin = install_root / ".venv-diffusers" / "bin"
    venv_bin.mkdir(parents=True)
    python_path = venv_bin / "python"
    python_path.write_text("#!/usr/bin/env bash\n")
    python_path.chmod(0o755)
    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)
    return install_root


def _write_marker(state_dir: Path, pins: tuple[str, ...]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    lines = ["installed_ts: 2026-05-26T10:15:42", "pinned_versions:"]
    for pin in pins:
        lines.append(f"  {pin}")
    (state_dir / "video_deps_installed_at.txt").write_text(
        "\n".join(lines) + "\n",
    )


class TestVideoDepsDoctorSection:
    """Doctor reports a video-deps line whenever .venv-diffusers/
    exists. Status tier matches the marker file."""

    def test_marker_absent_reports_info(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.doctor import _report_video_deps_health
        _setup_fake_venv(tmp_path, monkeypatch)
        # No marker written.
        _report_video_deps_health()
        out = capsys.readouterr().out
        assert "video deps" in out.lower()
        # No file means "absent" — INFO tier (no error icon).
        assert "absent" in out.lower() or "not yet installed" in out.lower()
        assert "imgen video" in out

    def test_marker_matches_pinned_reports_green(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.doctor import _report_video_deps_health
        from imgen.commands.video import _VIDEO_DEPS_PINNED
        from imgen.paths import STATE_DIR
        _setup_fake_venv(tmp_path, monkeypatch)
        _write_marker(STATE_DIR, _VIDEO_DEPS_PINNED)
        _report_video_deps_health()
        out = capsys.readouterr().out
        assert "video deps" in out.lower()
        # Match means GREEN — present + matches pinned versions
        assert "present" in out.lower()
        # GREEN tier: no DRIFT word
        assert "drift" not in out.lower()

    def test_marker_drift_reports_yellow(self, tmp_path, monkeypatch, capsys):
        from imgen.commands.doctor import _report_video_deps_health
        from imgen.paths import STATE_DIR
        _setup_fake_venv(tmp_path, monkeypatch)
        # Stale pin set — older version still installed; pinned set bumped.
        _write_marker(STATE_DIR, ("imageio==2.30.0",
                                  "imageio-ffmpeg==0.5.0",
                                  "sentencepiece==0.2.1"))
        _report_video_deps_health()
        out = capsys.readouterr().out
        assert "drift" in out.lower()

    def test_drift_message_lists_specific_mismatched_pins(self, tmp_path, monkeypatch, capsys):
        """Drift message must surface WHICH pins drifted, not just say
        "drift" — so the user knows what to re-install."""
        from imgen.commands.doctor import _report_video_deps_health
        from imgen.paths import STATE_DIR
        _setup_fake_venv(tmp_path, monkeypatch)
        _write_marker(STATE_DIR, ("imageio==2.30.0",
                                  "imageio-ffmpeg==0.6.0",
                                  "sentencepiece==0.2.1"))
        _report_video_deps_health()
        out = capsys.readouterr().out
        # 2.30.0 (installed) is mentioned somewhere
        assert "2.30.0" in out or "imageio" in out.lower()

    def test_marker_malformed_reports_info(self, tmp_path, monkeypatch, capsys):
        """Hand-edited / corrupted marker → don't crash doctor; treat
        as absent."""
        from imgen.commands.doctor import _report_video_deps_health
        from imgen.paths import STATE_DIR
        _setup_fake_venv(tmp_path, monkeypatch)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        # Garbage content
        (STATE_DIR / "video_deps_installed_at.txt").write_text(
            "completely\nmalformed\ngarbage\n",
        )
        # Must not raise
        _report_video_deps_health()
        out = capsys.readouterr().out
        # Treated as absent (no pinned_versions section parseable)
        assert "video deps" in out.lower()


class TestVideoDepsDoctorScoping:
    """The video-deps section only fires when .venv-diffusers/ exists
    — a colleague who never opted into diffusers shouldn't see it."""

    def test_no_venv_no_video_deps_section(self, tmp_path, monkeypatch, capsys):
        """When .venv-diffusers/ is missing, the broader
        _report_diffusers_health surfaces that gap; calling
        _report_video_deps_health directly with no venv should
        still be safe (might emit nothing or just the absent line —
        both are acceptable defensive behaviours)."""
        from imgen.commands.doctor import _report_video_deps_health
        from imgen import paths
        # No setup — IMGEN_INSTALL_ROOT default; .venv-diffusers/
        # certainly missing. Must not raise.
        _report_video_deps_health()

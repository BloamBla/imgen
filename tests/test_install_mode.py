"""Detect mixed install — `pipx install imgen` AND `bootstrap.sh` present.

If a colleague installs both ways, `~/imgen/imgen` shim wins on their
alias but `~/.local/bin/imgen` (from pipx) is also on PATH. Confusing.
`cmd_doctor` warns when it sees both, with a hint to pick one.
"""
from __future__ import annotations

from pathlib import Path

from imgen.commands.doctor import detect_install_collision


def test_no_collision_when_only_pipx(tmp_path):
    """pipx user (IMGEN_HOME is None) — no collision possible."""
    home = tmp_path
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "imgen").write_text("# pipx entry")
    assert detect_install_collision(home=home, imgen_home=None) is None


def test_no_collision_when_only_bootstrap(tmp_path):
    """bootstrap user, no pipx — IMGEN_HOME set but no ~/.local/bin/imgen."""
    home = tmp_path
    imgen_home = tmp_path / "imgen"
    (imgen_home / ".venv" / "bin").mkdir(parents=True)
    (imgen_home / ".venv" / "bin" / "imgen").write_text("# bootstrap entry")
    assert detect_install_collision(home=home, imgen_home=imgen_home) is None


def test_collision_when_both_install_paths_present(tmp_path):
    """Both paths exist → warning text returned."""
    home = tmp_path
    imgen_home = tmp_path / "imgen"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "imgen").write_text("# pipx entry")
    (imgen_home / ".venv" / "bin").mkdir(parents=True)
    (imgen_home / ".venv" / "bin" / "imgen").write_text("# bootstrap entry")
    warning = detect_install_collision(home=home, imgen_home=imgen_home)
    assert warning is not None
    assert "both" in warning.lower() or "two" in warning.lower() \
        or "collision" in warning.lower() or "mixed" in warning.lower()


def test_no_collision_pipx_path_missing_for_bootstrap_user(tmp_path):
    """Bootstrap user with ~/.local/bin existing but no `imgen` in it."""
    home = tmp_path
    imgen_home = tmp_path / "imgen"
    (home / ".local" / "bin").mkdir(parents=True)
    (imgen_home / ".venv" / "bin").mkdir(parents=True)
    (imgen_home / ".venv" / "bin" / "imgen").write_text("# bootstrap entry")
    # No ~/.local/bin/imgen file → no collision
    assert detect_install_collision(home=home, imgen_home=imgen_home) is None

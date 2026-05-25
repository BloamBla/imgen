"""v0.8.0 commit 1 — IMGEN_INSTALL_ROOT path constant + fallback chain.

Per [[project-v080-design]] §E: diffusers_mps engine resolves
`.venv-diffusers/` relative to a stable install-root anchor, NEVER
cwd-relative. The fallback chain handles canonical bootstrap.sh layout
AND pipx/uv/Homebrew installs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestImgenInstallRoot:
    def test_imgen_install_root_resolves_for_current_install(self):
        """In the live test process, IMGEN_INSTALL_ROOT must resolve to
        a directory containing src/imgen/__init__.py — proves both
        probes work on the actual canonical install."""
        from imgen.paths import IMGEN_INSTALL_ROOT
        assert (IMGEN_INSTALL_ROOT / "src" / "imgen" / "__init__.py").is_file()

    def test_compute_uses_sys_prefix_parent_when_valid(self, tmp_path, monkeypatch):
        """Primary probe: sys.prefix.parent contains src/imgen/. Build
        a fake venv-layout tree and assert the function returns that."""
        from imgen.paths import _compute_imgen_install_root
        # Fake imgen install root with a venv inside it.
        fake_root = tmp_path / "imgen"
        (fake_root / "src" / "imgen").mkdir(parents=True)
        (fake_root / "src" / "imgen" / "__init__.py").touch()
        fake_venv = fake_root / ".venv"
        fake_venv.mkdir()
        monkeypatch.setattr(sys, "prefix", str(fake_venv))
        assert _compute_imgen_install_root() == fake_root

    def test_compute_falls_back_to_file_relative(self, tmp_path, monkeypatch):
        """Fallback probe: sys.prefix.parent doesn't contain imgen, so
        function uses __file__-relative resolution."""
        from imgen import paths as paths_module
        from imgen.paths import _compute_imgen_install_root
        # Point sys.prefix at a directory whose parent does NOT contain
        # src/imgen/.
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        monkeypatch.setattr(sys, "prefix", str(unrelated / "subdir"))
        (unrelated / "subdir").mkdir()
        # __file__ is the real paths.py — its parents[2] IS the real
        # imgen install root (by construction; paths.py lives at
        # src/imgen/paths.py).
        expected = Path(paths_module.__file__).resolve().parents[2]
        assert _compute_imgen_install_root() == expected

    def test_compute_dies_when_neither_probe_resolves(self, tmp_path, monkeypatch):
        """Last-resort die: both probes fail (sys.prefix not pointing at
        imgen, AND __file__ artificially relocated). Mock the module
        attribute __file__ so parents[2] also misses src/imgen/."""
        from imgen import paths as paths_module
        # Sys.prefix probe: definitively wrong path.
        wrong = tmp_path / "nope" / "venv"
        wrong.mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(wrong))
        # __file__ probe: point at a tmp path with no src/imgen.
        fake_file = tmp_path / "elsewhere" / "src" / "fake_pkg" / "paths.py"
        fake_file.parent.mkdir(parents=True)
        fake_file.touch()
        monkeypatch.setattr(paths_module, "__file__", str(fake_file))
        with pytest.raises(SystemExit):
            paths_module._compute_imgen_install_root()

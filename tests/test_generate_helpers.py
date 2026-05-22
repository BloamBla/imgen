"""Tests for private helpers extracted from cmd_generate during v0.2.4.

cmd_generate grew to ~590 lines / 18 phases by v0.2.3. v0.2.4 extracts
its pre-build phases into named helpers (architect item I1). These
tests lock in the behaviour of each helper so the extraction is a
pure mechanical move with no semantic shift.

Patterns:
- `die()` exits via sys.exit; helpers calling it are caught with
  `pytest.raises(SystemExit)` + `exc_info.value.code`.
- `args` mimicked via `types.SimpleNamespace` so tests don't need to
  build a full argparse Namespace.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.commands.generate import _validate_input_path


# ── _validate_input_path ────────────────────────────────────────────────

def test_validate_input_path_existing_file_returns_resolved(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake jpeg bytes")

    result = _validate_input_path(str(img))

    assert result == img.resolve()
    assert result.is_absolute()


def test_validate_input_path_expands_tilde(tmp_path, monkeypatch):
    """`~/photo.jpg` must expand — colleagues drop pictures into ~/Desktop
    and reference them with ~ in shell."""
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")

    result = _validate_input_path("~/photo.jpg")

    assert result == img.resolve()


def test_validate_input_path_resolves_relative(tmp_path, monkeypatch):
    """Relative paths must become absolute so downstream `mflux` invoke
    doesn't depend on cwd."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    result = _validate_input_path("photo.jpg")

    assert result == img.resolve()
    assert result.is_absolute()


def test_validate_input_path_missing_file_exits_code_2(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jpg"

    with pytest.raises(SystemExit) as exc_info:
        _validate_input_path(str(missing))

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Image not found" in err
    assert str(missing) in err


def test_validate_input_path_directory_exits_code_2(tmp_path, capsys):
    """If the user points at a folder we reject — only file inputs make
    sense for image→style transfer."""
    folder = tmp_path / "not-a-file"
    folder.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        _validate_input_path(str(folder))

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Not a file" in err

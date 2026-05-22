"""argparse validators — bounded ranges + safe output paths.

These run at parse time (CLI level), so they're the first line of
defense against bad user input. Off-by-one in a range or a missing
extension in the allowlist could land bad values in cmd_generate.
"""
from __future__ import annotations

import argparse

import pytest

from imgen.parser import _float_range, _int_range, _safe_output_path


# ── _int_range ────────────────────────────────────────────────────────

def test_int_range_accepts_in_range():
    v = _int_range(1, 100)
    assert v("50") == 50


@pytest.mark.parametrize("boundary", ["1", "100"])
def test_int_range_accepts_inclusive_boundaries(boundary):
    v = _int_range(1, 100)
    assert v(boundary) == int(boundary)


@pytest.mark.parametrize("bad", ["0", "101", "-1", "1000"])
def test_int_range_rejects_out_of_range(bad):
    v = _int_range(1, 100)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


@pytest.mark.parametrize("bad", ["abc", "1.5", "", "1e2"])
def test_int_range_rejects_non_integer(bad):
    v = _int_range(1, 100)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


# ── _float_range ──────────────────────────────────────────────────────

def test_float_range_accepts_in_range():
    v = _float_range(0.0, 1.0)
    assert v("0.55") == 0.55


@pytest.mark.parametrize("boundary", ["0.0", "1.0"])
def test_float_range_accepts_inclusive_boundaries(boundary):
    v = _float_range(0.0, 1.0)
    assert v(boundary) == float(boundary)


@pytest.mark.parametrize("bad", ["-0.1", "1.1", "2.0", "-1"])
def test_float_range_rejects_out_of_range(bad):
    v = _float_range(0.0, 1.0)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


def test_float_range_rejects_non_float():
    v = _float_range(0.0, 1.0)
    with pytest.raises(argparse.ArgumentTypeError):
        v("not-a-number")


# ── _safe_output_path ─────────────────────────────────────────────────

@pytest.mark.parametrize("good", ["out.png", "out.jpg", "out.jpeg", "out.webp",
                                  "/abs/path/x.PNG", "x.JPEG"])
def test_safe_output_path_accepts_known_image_extensions(good):
    """Allowlist enforced case-insensitively."""
    assert _safe_output_path(good) == good


@pytest.mark.parametrize("bad", [
    "out.terminal",   # macOS would launch Terminal.app
    "out.command",    # macOS would execute as shell
    "out.sh",         # shell script
    "out.app",        # would launch the .app bundle
    "out",            # no extension
    "out.gif",        # not in allowlist
    "out.bmp",
])
def test_safe_output_path_rejects_non_image_extensions(bad):
    """The auto-`open` path would launch the registered app for the
    suffix; restricting to image-only suffixes is defence-in-depth.
    Pins security #8 v0.1.1 fix."""
    with pytest.raises(argparse.ArgumentTypeError):
        _safe_output_path(bad)


# ── --scope default (v0.3.2) ───────────────────────────────────────────


from imgen.parser import build_parser


def test_generate_scope_defaults_to_scene():
    """v0.3.2: ``--scope`` defaults to ``scene`` (was ``None`` in v0.3.1
    and earlier). Most photos colleagues batch are scenes / group shots;
    person-focus is the special case the user opts into explicitly."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg"])
    assert args.scope == "scene"


def test_batch_scope_defaults_to_scene():
    """Same default applies to ``imgen batch <dir>``."""
    parser = build_parser()
    args = parser.parse_args(["batch", "/tmp/dir"])
    assert args.scope == "scene"


def test_generate_scope_person_explicit():
    """Person-focus requires explicit opt-in."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg", "--scope", "person"])
    assert args.scope == "person"


def test_batch_scope_person_explicit():
    parser = build_parser()
    args = parser.parse_args(
        ["batch", "/tmp/dir", "--scope", "person"]
    )
    assert args.scope == "person"


def test_generate_scope_scene_explicit_still_works():
    """Passing --scope scene explicitly resolves to the same default
    (back-compat with users who already typed it before v0.3.2)."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg", "--scope", "scene"])
    assert args.scope == "scene"


# ── -v short flag for --version (v0.3.5) ───────────────────────────────


def test_short_v_prints_version_and_exits(capsys):
    """v0.3.5: `imgen -v` mirrors `--version`. node/npm/pip ergonomics
    — every user types `imgen -v` first; previously got "unrecognized
    arguments"."""
    from imgen import __version__

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["-v"])
    # argparse's version action exits 0
    assert exc.value.code == 0
    captured = capsys.readouterr()
    # argparse writes version output to stdout
    assert __version__ in captured.out


def test_short_v_and_long_version_both_print_same(capsys):
    """`-v` and `--version` must produce identical output."""
    from imgen import __version__

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-v"])
    short_out = capsys.readouterr().out

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])
    long_out = capsys.readouterr().out

    assert short_out == long_out
    assert __version__ in short_out


# ── v0.4: --backend choices include user backends from backends.d/ ──────


def test_parser_loads_user_backends_before_choices(tmp_path, monkeypatch):
    """v0.4 design decision 3: --backend choices are loaded at parse
    time via list_backends(), so a TOML in ~/.imgen/backends.d/ shows
    up as a valid --backend argument without code changes.

    Without this, `imgen --backend custom_thing` died with "invalid
    choice" even when the TOML was valid — defeating the registry."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    (backends_dir / "mythical.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    backends_mod.reset_backends_cache()
    try:
        parser = build_parser()
        args = parser.parse_args(
            ["generate", "photo.jpg", "--backend", "mythical"]
        )
        assert args.backend == "mythical"
    finally:
        backends_mod.reset_backends_cache()


def test_parser_rejects_unknown_backend(tmp_path, monkeypatch):
    """Sanity: a string that doesn't match any built-in or user
    backend still dies with argparse's "invalid choice"."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    (state / "backends.d").mkdir()  # empty
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    backends_mod.reset_backends_cache()
    try:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["generate", "photo.jpg", "--backend", "totally_unknown_xyz"]
            )
    finally:
        backends_mod.reset_backends_cache()

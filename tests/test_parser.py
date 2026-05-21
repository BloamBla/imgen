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

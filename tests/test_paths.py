"""Tests for paths.py helpers added in v0.2.3.

`auto_run_dirname()` and `next_available_run_dir()` back the new
folder-per-invocation output layout: every `imgen` run drops its
artefacts into `~/Desktop/imgen/<start-ts>/` instead of a flat
`~/Desktop/imgen/<basename>_<style>_<id>.png`.

Format is locked at all-dashes (no colons — macOS sometimes refuses
them in file dialogs / older toolchains), second precision (we run
serially, no concurrent generations finishing in the same second).
"""
from __future__ import annotations

import datetime as dt

import pytest

from imgen.paths import auto_run_dirname, next_available_run_dir


# ── auto_run_dirname format ─────────────────────────────────────────────

def test_auto_run_dirname_default_uses_now():
    """No argument → current local time. Smoke-test the shape, not the value."""
    name = auto_run_dirname()
    # YYYY-MM-DD-HH-MM-SS → 19 chars, 5 dashes between digit groups.
    assert len(name) == 19
    assert name.count("-") == 5
    assert all(part.isdigit() for part in name.split("-"))


def test_auto_run_dirname_explicit_datetime_formats_predictably():
    when = dt.datetime(2026, 5, 21, 14, 30, 12)
    assert auto_run_dirname(when) == "2026-05-21-14-30-12"


def test_auto_run_dirname_pads_single_digit_components():
    """Jan 3rd 09:05:07 → '2026-01-03-09-05-07', not '2026-1-3-9-5-7'."""
    when = dt.datetime(2026, 1, 3, 9, 5, 7)
    assert auto_run_dirname(when) == "2026-01-03-09-05-07"


def test_auto_run_dirname_no_colons():
    """macOS / older tooling sometimes chokes on `:` in filenames; we don't
    use ISO-8601 `T` either to keep the whole thing one separator."""
    when = dt.datetime(2026, 5, 21, 14, 30, 12)
    name = auto_run_dirname(when)
    assert ":" not in name
    assert "T" not in name


def test_auto_run_dirname_sortable_alphabetically():
    """File-managers sort by filename — same chronological order must hold."""
    earlier = auto_run_dirname(dt.datetime(2026, 5, 21, 14, 30, 12))
    later = auto_run_dirname(dt.datetime(2026, 5, 21, 14, 30, 13))
    assert earlier < later


# ── next_available_run_dir collision suffix ─────────────────────────────

def test_next_available_run_dir_returns_plain_when_free(tmp_path):
    target = next_available_run_dir(tmp_path, "2026-05-21-14-30-12")
    assert target == tmp_path / "2026-05-21-14-30-12"


def test_next_available_run_dir_suffixes_when_exists(tmp_path):
    """Sub-second collision (rare — only via scripted double-invoke) → `_2`."""
    (tmp_path / "2026-05-21-14-30-12").mkdir()
    target = next_available_run_dir(tmp_path, "2026-05-21-14-30-12")
    assert target == tmp_path / "2026-05-21-14-30-12_2"


def test_next_available_run_dir_increments_until_free(tmp_path):
    (tmp_path / "2026-05-21-14-30-12").mkdir()
    (tmp_path / "2026-05-21-14-30-12_2").mkdir()
    (tmp_path / "2026-05-21-14-30-12_3").mkdir()
    target = next_available_run_dir(tmp_path, "2026-05-21-14-30-12")
    assert target == tmp_path / "2026-05-21-14-30-12_4"


def test_next_available_run_dir_does_not_create(tmp_path):
    """Helper is pure — returns a Path, caller mkdir's. Tests that rely on
    'this is the path that *would* be used' shouldn't get a side effect."""
    target = next_available_run_dir(tmp_path, "2026-05-21-14-30-12")
    assert not target.exists()

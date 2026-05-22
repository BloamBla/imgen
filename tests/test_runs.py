"""Tests for runs.py — folder-per-invocation + per-batch log helpers.

`auto_run_dirname()` and `next_available_run_dir()` back the folder-per-
invocation output layout: every `imgen` run drops its artefacts into
`~/Desktop/imgen/<start-ts>/` instead of a flat
`~/Desktop/imgen/<basename>_<style>_<id>.png`.

Format is locked at all-dashes (no colons — macOS sometimes refuses
them in file dialogs / older toolchains), second precision (we run
serially, no concurrent generations finishing in the same second).

Lived in test_paths.py until v0.2.4 when the helpers were split out
into a dedicated runs.py module.
"""
from __future__ import annotations

import datetime as dt

import pytest

from imgen.runs import (
    auto_run_dirname,
    ensure_logs_dir,
    next_available_run_dir,
    open_log_file_append,
)


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


# ── ensure_logs_dir ─────────────────────────────────────────────────────

def test_ensure_logs_dir_creates_with_0o700(tmp_path, monkeypatch):
    """Per-batch logs dir is restrictive — stderr can contain prompts that
    were redacted-via-substring (not bullet-proof) so don't leak even
    that to co-tenants."""
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    monkeypatch.setattr(paths_mod, "STATE_DIR", tmp_path / ".imgen")
    monkeypatch.setattr(runs_mod, "LOGS_DIR", tmp_path / ".imgen" / "logs")

    ensure_logs_dir()

    state = tmp_path / ".imgen"
    logs = state / "logs"
    assert state.is_dir()
    assert logs.is_dir()
    assert (state.stat().st_mode & 0o777) == 0o700
    assert (logs.stat().st_mode & 0o777) == 0o700


def test_ensure_logs_dir_idempotent(tmp_path, monkeypatch):
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    state = tmp_path / ".imgen"
    state.mkdir(mode=0o700)
    logs = state / "logs"
    logs.mkdir(mode=0o700)
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)

    ensure_logs_dir()  # should not raise on pre-existing dirs

    assert logs.is_dir()
    assert (logs.stat().st_mode & 0o777) == 0o700


def test_ensure_logs_dir_tightens_loose_perms(tmp_path, monkeypatch):
    """If logs dir somehow exists with 0o755, chmod it back to 0o700."""
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    state = tmp_path / ".imgen"
    state.mkdir(mode=0o700)
    logs = state / "logs"
    logs.mkdir(mode=0o755)
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)

    ensure_logs_dir()

    assert (logs.stat().st_mode & 0o777) == 0o700


def test_ensure_logs_dir_refuses_to_chmod_through_symlink(
    tmp_path, monkeypatch, capsys
):
    """If ~/.imgen/logs is a symlink pointing elsewhere, ensure_logs_dir
    must NOT follow it and chmod 0o700 on whatever the target is —
    that target may be a directory the user uses for unrelated purposes
    (or worse, some shared/system dir if mounting was weird).

    Refuse with a warn() instead, so the user finds out their LOGS_DIR
    is a symlink and decides what to do. (v0.2.5 security NIT-2)"""
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    state = tmp_path / ".imgen"
    state.mkdir(mode=0o700)
    # Target the symlink resolves to — give it a deliberately
    # different mode so we can detect a stray chmod.
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o755)
    logs = state / "logs"
    logs.symlink_to(target)
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)

    ensure_logs_dir()

    # Target's mode untouched — we did not follow the symlink.
    assert (target.stat().st_mode & 0o777) == 0o755
    # User informed.
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "symlink" in combined.lower()


def test_ensure_logs_dir_symlink_does_not_create_under_target(
    tmp_path, monkeypatch
):
    """Symlink at LOGS_DIR → nonexistent path. Without the guard,
    `LOGS_DIR.exists()` returns False (symlink target missing), and
    `LOGS_DIR.mkdir(0o700)` would fail with FileNotFoundError or
    even create a directory at the symlink's target location. The
    guard catches it before mkdir."""
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    state = tmp_path / ".imgen"
    state.mkdir(mode=0o700)
    nonexistent_target = tmp_path / "phantom"
    logs = state / "logs"
    logs.symlink_to(nonexistent_target)
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)

    # Should NOT raise (we refuse, not crash).
    ensure_logs_dir()

    # Nothing got materialised at the phantom target.
    assert not nonexistent_target.exists()


# ── open_log_file_append (v0.2.3 review fix: security I1) ───────────────

def test_open_log_file_append_creates_with_0o600(tmp_path):
    """Default umask on macOS gives 0o644 — world-readable. Force 0o600
    from the syscall so batch logs aren't readable by co-tenants on a
    shared Mac, matching how ~/.imgen/hf_token is handled."""
    log = tmp_path / "batch.log"

    with open_log_file_append(log) as f:
        f.write(b"hello\n")

    assert log.exists()
    assert (log.stat().st_mode & 0o777) == 0o600
    assert log.read_bytes() == b"hello\n"


def test_open_log_file_append_appends_not_truncates(tmp_path):
    """Re-opening the same path must not wipe earlier content. The whole
    point of per-batch logs is that each iteration appends — truncation
    on re-open would lose markers."""
    log = tmp_path / "batch.log"
    with open_log_file_append(log) as f:
        f.write(b"first\n")
    with open_log_file_append(log) as f:
        f.write(b"second\n")

    assert log.read_bytes() == b"first\nsecond\n"


def test_open_log_file_append_preserves_existing_perms(tmp_path):
    """If a log already exists with 0o600, a re-open does not clobber
    the mode. (Re-open uses O_CREAT but the file is already there;
    POSIX semantics keep the existing perms.)"""
    log = tmp_path / "batch.log"
    with open_log_file_append(log) as f:
        f.write(b"first\n")
    # User loosened perms manually — re-open should not silently fix it,
    # but also shouldn't break.
    log.chmod(0o644)
    with open_log_file_append(log) as f:
        f.write(b"second\n")
    # We don't FIX wider perms on existing files — that would surprise
    # the user. The 0o600 invariant only applies at creation.
    assert (log.stat().st_mode & 0o777) == 0o644

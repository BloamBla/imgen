"""Tests for BatchLogger + prune_old_batch_logs in runs.py.

BatchLogger owns the per-batch log lifecycle that used to be split
across commands/generate.py (header + iteration markers),
subprocess_helpers.py (the redacted-stderr tee), and commands/clean.py
(retention). Splitting it out makes v0.3.0's per-image-log or per-row
rotation feasible without editing three modules. Lives in runs.py.

Two surfaces under test:
  - BatchLogger: header, iteration start/end/cancelled markers, append-
    semantics, 0o600 file perms inherited from open_log_file_append.
  - prune_old_batch_logs: mtime cutoff, dry-run mode, .log-only glob,
    no-dir noop.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from imgen.runs import BatchLogger, prune_old_batch_logs


# ── BatchLogger construction + path ─────────────────────────────────────


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    """Redirect LOGS_DIR + STATE_DIR to a tmp dir so the suite never
    touches the user's real ~/.imgen/logs/."""
    state = tmp_path / ".imgen"
    state.mkdir(mode=0o700)
    logs = state / "logs"

    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)
    return logs


def test_batch_logger_path_uses_batch_id(logs_dir):
    """Log file name is <batch_id>.log under LOGS_DIR."""
    logger = BatchLogger("abc123def456")
    assert logger.path == logs_dir / "abc123def456.log"


def test_batch_logger_init_ensures_logs_dir_exists(logs_dir):
    """Constructor side-effect: LOGS_DIR is created if missing (so
    write_header doesn't ENOENT). Mirrors the v0.2.3 ensure_logs_dir
    call site in cmd_generate."""
    assert not logs_dir.exists()
    BatchLogger("abc123")
    assert logs_dir.is_dir()
    assert (logs_dir.stat().st_mode & 0o777) == 0o700


# ── BatchLogger.write_header ────────────────────────────────────────────


def _read(path: Path) -> str:
    return path.read_bytes().decode()


def test_batch_logger_write_header_includes_batch_id(logs_dir):
    logger = BatchLogger("abc123def456")
    logger.write_header(
        input_path=Path("/photos/vacation.jpg"),
        styles=["anime", "ghibli"],
        run_dir=Path("/desktop/imgen/2026-05-22-10-00-00"),
        backend="flux",
        quant=8,
        preview=False,
        scope=None,
        seed=42,
    )
    content = _read(logger.path)
    assert "abc123def456" in content
    assert "vacation.jpg" in content


def test_batch_logger_write_header_lists_styles(logs_dir):
    logger = BatchLogger("abc")
    logger.write_header(
        input_path=Path("/x.jpg"),
        styles=["anime", "ghibli", "pixar"],
        run_dir=Path("/out"),
        backend="flux",
        quant=8,
        preview=False,
        scope=None,
        seed=1,
    )
    content = _read(logger.path)
    assert "anime, ghibli, pixar" in content


def test_batch_logger_write_header_includes_backend_quant_seed(logs_dir):
    logger = BatchLogger("abc")
    logger.write_header(
        input_path=Path("/x.jpg"),
        styles=["a"],
        run_dir=Path("/out"),
        backend="qwen",
        quant=4,
        preview=True,
        scope="person",
        seed=99,
    )
    content = _read(logger.path)
    assert "qwen q4" in content
    assert "preview=True" in content
    assert "scope=person" in content
    assert "seed=99" in content


def test_batch_logger_write_header_creates_file_0o600(logs_dir):
    """Inherits the 0o600-from-creation behaviour of open_log_file_append."""
    logger = BatchLogger("abc")
    logger.write_header(
        input_path=Path("/x"),
        styles=["a"],
        run_dir=Path("/o"),
        backend="qwen",
        quant=4,
        preview=False,
        scope=None,
        seed=1,
    )
    assert (logger.path.stat().st_mode & 0o777) == 0o600


# ── iteration markers ───────────────────────────────────────────────────


def test_batch_logger_iteration_start_marker(logs_dir):
    logger = BatchLogger("abc")
    ts = dt.datetime(2026, 5, 22, 10, 0, 12)
    logger.iteration_start(idx=2, total=3, style="anime", ts=ts)
    content = _read(logger.path)
    assert "[2/3] anime" in content
    assert "2026-05-22T10:00:12" in content


def test_batch_logger_iteration_end_ok(logs_dir):
    logger = BatchLogger("abc")
    logger.iteration_end(idx=1, total=2, style="anime", returncode=0, duration=42)
    content = _read(logger.path)
    assert "[1/2] anime" in content
    assert " ok " in content
    assert "in 42s" in content


def test_batch_logger_iteration_end_failed_includes_exit_code(logs_dir):
    logger = BatchLogger("abc")
    logger.iteration_end(idx=1, total=2, style="anime", returncode=7, duration=10)
    content = _read(logger.path)
    assert "FAILED exit=7" in content
    assert "in 10s" in content


def test_batch_logger_iteration_cancelled_marker(logs_dir):
    logger = BatchLogger("abc")
    logger.iteration_cancelled(idx=2, total=3, style="ghibli", duration=5)
    content = _read(logger.path)
    assert "[2/3] ghibli" in content
    assert "CANCELLED" in content
    assert "in 5s" in content


def test_batch_logger_markers_append_not_truncate(logs_dir):
    """Each marker writes to a fresh open_log_file_append handle —
    earlier content must survive."""
    logger = BatchLogger("abc")
    logger.write_header(
        input_path=Path("/x"), styles=["a"], run_dir=Path("/o"),
        backend="qwen", quant=4, preview=False, scope=None, seed=1,
    )
    ts = dt.datetime(2026, 5, 22, 10, 0, 0)
    logger.iteration_start(idx=1, total=2, style="a", ts=ts)
    logger.iteration_end(idx=1, total=2, style="a", returncode=0, duration=3)
    logger.iteration_start(idx=2, total=2, style="b", ts=ts)
    logger.iteration_end(idx=2, total=2, style="b", returncode=1, duration=4)

    content = _read(logger.path)
    # All four markers + header survive in order.
    assert content.find("imgen batch") < content.find("[1/2] a")
    assert content.find("[1/2] a") < content.find("[2/2] b")
    assert "FAILED" in content


# ── prune_old_batch_logs ────────────────────────────────────────────────


def _make_log(path: Path, days_ago: float, content: bytes = b"x") -> None:
    path.write_bytes(content)
    past = path.stat().st_mtime - days_ago * 86400
    os.utime(path, (past, past))


def test_prune_no_logs_dir_returns_zero(logs_dir):
    """Fresh user (no batches yet) → no LOGS_DIR → silent no-op."""
    # logs_dir fixture monkeypatched but didn't mkdir
    assert not logs_dir.exists()

    removed, removed_bytes = prune_old_batch_logs(days=30)

    assert removed == 0
    assert removed_bytes == 0


def test_prune_empty_logs_dir_returns_zero(logs_dir):
    logs_dir.mkdir(mode=0o700)
    removed, _ = prune_old_batch_logs(days=30)
    assert removed == 0


def test_prune_removes_old_logs_keeps_recent(logs_dir):
    logs_dir.mkdir(mode=0o700)
    old = logs_dir / "old.log"
    new = logs_dir / "new.log"
    _make_log(old, days_ago=35)
    _make_log(new, days_ago=5)

    removed, _ = prune_old_batch_logs(days=30)

    assert not old.exists()
    assert new.exists()
    assert removed == 1


def test_prune_reports_bytes_removed(logs_dir):
    logs_dir.mkdir(mode=0o700)
    payload = b"x" * 1024  # 1 KB
    _make_log(logs_dir / "old.log", days_ago=35, content=payload)

    _, removed_bytes = prune_old_batch_logs(days=30)

    assert removed_bytes == 1024


def test_prune_dry_run_counts_but_does_not_delete(logs_dir):
    logs_dir.mkdir(mode=0o700)
    old = logs_dir / "old.log"
    _make_log(old, days_ago=35)

    removed, _ = prune_old_batch_logs(days=30, dry_run=True)

    assert old.exists(), "--dry-run must not delete"
    assert removed == 1


def test_prune_only_targets_log_extension(logs_dir):
    """Non-.log files (e.g. user dropped notes.txt under LOGS_DIR) are
    not touched — the glob is `*.log` exactly to avoid surprises."""
    logs_dir.mkdir(mode=0o700)
    _make_log(logs_dir / "old.log", days_ago=35)
    _make_log(logs_dir / "old.txt", days_ago=35)

    prune_old_batch_logs(days=30)

    assert not (logs_dir / "old.log").exists()
    assert (logs_dir / "old.txt").exists()


def test_prune_boundary_kept(logs_dir):
    """File slightly fresher than cutoff stays; strict `< cutoff`
    comparison means an exact-cutoff file would also survive."""
    logs_dir.mkdir(mode=0o700)
    boundary = logs_dir / "boundary.log"
    _make_log(boundary, days_ago=29.9)

    prune_old_batch_logs(days=30)

    assert boundary.exists()

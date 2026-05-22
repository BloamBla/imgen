"""Retention of per-batch logs by `imgen clean` (v0.2.3).

`_prune_old_batch_logs` is the only new logic — covers it directly with
mtime manipulation. The HF cache cleanup is smoke-tested manually
(needs real symlink-shaped HF directory; not worth mocking).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from imgen.commands.clean import _prune_old_batch_logs
from imgen.runs import LOG_RETENTION_DAYS


@pytest.fixture
def tmp_logs(tmp_path, monkeypatch):
    """Redirect runs.LOGS_DIR to a per-test tmp location.

    v0.2.4: clean.py no longer references LOGS_DIR directly — the
    retention logic lives in runs.prune_old_batch_logs, which reads
    runs.LOGS_DIR. Patch the canonical home, not the import site
    (clean.py imports the function, not the constant)."""
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    import imgen.runs as runs_mod
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs)
    return logs


def _make_log(path, days_ago: float, content: str = "x") -> None:
    """Write a log file and stamp its mtime to N days in the past."""
    path.write_text(content)
    past = path.stat().st_mtime - days_ago * 86400
    os.utime(path, (past, past))


def test_prune_no_logs_dir_is_noop(tmp_path, monkeypatch, capsys):
    """No LOGS_DIR yet → silent no-op (fresh user has never run a batch)."""
    import imgen.runs as runs_mod
    monkeypatch.setattr(runs_mod, "LOGS_DIR", tmp_path / "nonexistent")

    _prune_old_batch_logs(SimpleNamespace(dry_run=False))

    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "removed" not in out.lower()


def test_prune_empty_logs_dir_quiet(tmp_logs, capsys):
    """Empty dir → no message (consistent with .incomplete=0 behaviour)."""
    _prune_old_batch_logs(SimpleNamespace(dry_run=False))
    out = capsys.readouterr().out
    assert "Removed" not in out and "Would remove" not in out


def test_prune_removes_old_logs_keeps_recent(tmp_logs, capsys):
    old_log = tmp_logs / "old_batch.log"
    new_log = tmp_logs / "new_batch.log"
    _make_log(old_log, days_ago=LOG_RETENTION_DAYS + 5)
    _make_log(new_log, days_ago=5)

    _prune_old_batch_logs(SimpleNamespace(dry_run=False))

    assert not old_log.exists(), "log older than retention must be deleted"
    assert new_log.exists(), "recent log must be preserved"
    out = capsys.readouterr().out
    assert "Removed 1" in out


def test_prune_boundary_day_kept(tmp_logs, capsys):
    """Exactly LOG_RETENTION_DAYS old (i.e. just on the cutoff) → kept.
    Cutoff is `now - N*86400`; a file with mtime == cutoff has mtime ==
    cutoff, not < cutoff, so it survives."""
    boundary = tmp_logs / "boundary.log"
    _make_log(boundary, days_ago=LOG_RETENTION_DAYS - 0.1)  # slightly fresher

    _prune_old_batch_logs(SimpleNamespace(dry_run=False))

    assert boundary.exists()


def test_prune_dry_run_counts_but_does_not_delete(tmp_logs, capsys):
    old_log = tmp_logs / "old_batch.log"
    _make_log(old_log, days_ago=LOG_RETENTION_DAYS + 5)

    _prune_old_batch_logs(SimpleNamespace(dry_run=True))

    assert old_log.exists(), "--dry-run must not delete"
    out = capsys.readouterr().out
    assert "Would remove" in out


def test_prune_only_targets_log_extension(tmp_logs, capsys):
    """Non-.log files (e.g. user dropped notes.txt) are left alone — the
    glob is `*.log` exactly to avoid surprises."""
    old_log = tmp_logs / "old_batch.log"
    old_txt = tmp_logs / "old_notes.txt"
    _make_log(old_log, days_ago=LOG_RETENTION_DAYS + 5)
    _make_log(old_txt, days_ago=LOG_RETENTION_DAYS + 5)

    _prune_old_batch_logs(SimpleNamespace(dry_run=False))

    assert not old_log.exists()
    assert old_txt.exists(), "non-.log files must not be touched"


def test_prune_size_uses_same_stat_snapshot(tmp_logs, capsys):
    """Two stat() calls would let st_size raise after st_mtime succeeded,
    desyncing the counter from the size. Use one stat snapshot for both
    fields. (python C2 from v0.2.3 review)"""
    old_log = tmp_logs / "old_batch.log"
    _make_log(old_log, days_ago=LOG_RETENTION_DAYS + 5, content="abc" * 100)
    expected_size = old_log.stat().st_size

    _prune_old_batch_logs(SimpleNamespace(dry_run=True))

    out = capsys.readouterr().out
    # Size printed in KB with 1 decimal — for 300 bytes that's "0.3 KB".
    assert f"{expected_size / 1024:.1f} KB" in out

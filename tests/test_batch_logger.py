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
        input_paths=[Path("/photos/vacation.jpg")],
        styles=["anime", "ghibli"],
        run_dir=Path("/desktop/imgen/2026-05-22-10-00-00"),
        model="flux",
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
        input_paths=[Path("/x.jpg")],
        styles=["anime", "ghibli", "pixar"],
        run_dir=Path("/out"),
        model="flux",
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
        input_paths=[Path("/x.jpg")],
        styles=["a"],
        run_dir=Path("/out"),
        model="qwen",
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
        input_paths=[Path("/x")],
        styles=["a"],
        run_dir=Path("/o"),
        model="qwen",
        quant=4,
        preview=False,
        scope=None,
        seed=1,
    )
    assert (logger.path.stat().st_mode & 0o777) == 0o600


# ── v0.3.0 header shape: N inputs + counted styles ──────────────────────


def test_batch_logger_write_header_inputs_line_counts_and_names(logs_dir):
    """v0.3.0 header format: `# inputs (N):  name1, name2, name3`.
    Counts in the prefix make N×M batches scannable at a glance — user
    can verify "yes I queued 17 photos" without re-counting commas.
    Names are basenames only (no leading path) so logs stay readable
    when input dir is deep."""
    logger = BatchLogger("hdr-multi")
    logger.write_header(
        input_paths=[
            Path("/photos/IMG_1234.heic"),
            Path("/photos/IMG_5678.heic"),
            Path("/photos/vacation.jpg"),
        ],
        styles=["anime", "ghibli", "pixar"],
        run_dir=Path("/out"),
        model="flux", quant=8, preview=False, scope=None, seed=42,
    )
    content = _read(logger.path)
    assert "# inputs (3):  IMG_1234.heic, IMG_5678.heic, vacation.jpg" in content


def test_batch_logger_write_header_inputs_line_single(logs_dir):
    """Single-input multi-style still uses the unified `# inputs (N):`
    shape (with N=1) — keeps the v0.3.0 log format uniform whether the
    invocation is `imgen generate` or `imgen batch`."""
    logger = BatchLogger("hdr-single")
    logger.write_header(
        input_paths=[Path("/photos/vacation.jpg")],
        styles=["anime", "ghibli"],
        run_dir=Path("/out"),
        model="flux", quant=8, preview=False, scope=None, seed=1,
    )
    content = _read(logger.path)
    assert "# inputs (1):  vacation.jpg" in content


def test_batch_logger_write_header_styles_line_counts(logs_dir):
    """`# styles (M):  a, b, c` — same count-in-prefix shape as inputs."""
    logger = BatchLogger("hdr-styles")
    logger.write_header(
        input_paths=[Path("/x.jpg")],
        styles=["anime", "ghibli", "pixar"],
        run_dir=Path("/out"),
        model="flux", quant=8, preview=False, scope=None, seed=1,
    )
    content = _read(logger.path)
    assert "# styles (3):  anime, ghibli, pixar" in content


def test_batch_logger_write_header_rejects_empty_input_paths(logs_dir):
    """Empty list of inputs is a caller bug — would render
    `# inputs (0):  ` which is meaningless. cmd_batch / cmd_generate
    have already verified the list is non-empty before reaching here;
    explicit guard so a future regression surfaces here instead of
    silently writing a confusing header."""
    logger = BatchLogger("hdr-empty")
    with pytest.raises(ValueError, match="input_paths"):
        logger.write_header(
            input_paths=[],
            styles=["a"],
            run_dir=Path("/o"),
            model="qwen", quant=4, preview=False, scope=None, seed=1,
        )


# ── v0.3.0 per-input section markers ────────────────────────────────────


def test_batch_logger_input_section_start_marker(logs_dir):
    """`=== INPUT <name> (k/N) ===` opens each input's block in the
    log. Name is the basename only — full path is in the header's
    `# inputs (N):` line."""
    logger = BatchLogger("section-start")
    logger.input_section_start(idx_input=2, total_inputs=3, name="IMG_5678.heic")
    content = _read(logger.path)
    assert "=== INPUT IMG_5678.heic (2/3) ===" in content


def test_batch_logger_input_section_end_marker_all_ok(logs_dir):
    """`=== INPUT <name> → <ok>/<total> ok in <dur> ===` closes the
    block. <ok>/<total> always reports successful/all-attempted so the
    user can spot per-input partial failures at a glance."""
    logger = BatchLogger("section-end-ok")
    logger.input_section_end(
        idx_input=1, total_inputs=2, name="vacation.jpg",
        ok_count=3, fail_count=0, duration=552,
    )
    content = _read(logger.path)
    assert "=== INPUT vacation.jpg → 3/3 ok in 552s ===" in content


def test_batch_logger_input_section_end_marker_partial(logs_dir):
    """When `fail_count > 0`, the marker reports `<ok>/<total>` (e.g.
    `2/3 ok` for 2 successes + 1 failure) and embeds the failure count
    so grepping the log for `INPUT.*fail=` finds every partial input."""
    logger = BatchLogger("section-end-partial")
    logger.input_section_end(
        idx_input=2, total_inputs=3, name="IMG_5678.heic",
        ok_count=2, fail_count=1, duration=400,
    )
    content = _read(logger.path)
    assert "INPUT IMG_5678.heic" in content
    assert "2/3 ok" in content
    assert "fail=1" in content


def test_batch_logger_input_section_end_marker_all_failed(logs_dir):
    """All-failed input still produces a closing marker (so per-input
    sections are balanced in the log file). `0/N ok fail=N` shape."""
    logger = BatchLogger("section-end-allfail")
    logger.input_section_end(
        idx_input=1, total_inputs=2, name="bad.heic",
        ok_count=0, fail_count=3, duration=120,
    )
    content = _read(logger.path)
    assert "0/3 ok" in content
    assert "fail=3" in content


def test_batch_logger_input_section_markers_after_close_raise(logs_dir):
    """Consistent with iteration_start / write_header: after close,
    section markers must raise instead of silently re-opening."""
    logger = BatchLogger("section-closed")
    logger.input_section_start(idx_input=1, total_inputs=1, name="x.jpg")
    logger.close()
    with pytest.raises(ValueError, match="closed"):
        logger.input_section_start(idx_input=1, total_inputs=1, name="x.jpg")
    with pytest.raises(ValueError, match="closed"):
        logger.input_section_end(
            idx_input=1, total_inputs=1, name="x.jpg",
            ok_count=1, fail_count=0, duration=1,
        )


def test_batch_logger_full_batch_log_order(logs_dir):
    """End-to-end: header → input_section_start(1/2) → iter_start →
    iter_end → input_section_end(1/2) → input_section_start(2/2) → ...
    All markers in chronological order in the file."""
    logger = BatchLogger("full-batch")
    logger.write_header(
        input_paths=[Path("/p/a.heic"), Path("/p/b.jpg")],
        styles=["anime"],
        run_dir=Path("/o"),
        model="flux", quant=8, preview=False, scope=None, seed=1,
    )
    ts = dt.datetime(2026, 5, 22, 10, 0, 0)
    logger.input_section_start(1, 2, "a.heic")
    logger.iteration_start(idx=1, total=2, style="anime", ts=ts)
    logger.iteration_end(idx=1, total=2, style="anime", returncode=0, duration=3)
    logger.input_section_end(
        idx_input=1, total_inputs=2, name="a.heic",
        ok_count=1, fail_count=0, duration=3,
    )
    logger.input_section_start(2, 2, "b.jpg")
    logger.iteration_start(idx=2, total=2, style="anime", ts=ts)
    logger.iteration_end(idx=2, total=2, style="anime", returncode=0, duration=4)
    logger.input_section_end(
        idx_input=2, total_inputs=2, name="b.jpg",
        ok_count=1, fail_count=0, duration=4,
    )
    content = _read(logger.path)
    positions = [
        content.find("# imgen batch"),
        content.find("INPUT a.heic (1/2)"),
        content.find("[1/2] anime →"),
        content.find("INPUT a.heic → 1/1 ok"),
        content.find("INPUT b.jpg (2/2)"),
        content.find("[2/2] anime →"),
        content.find("INPUT b.jpg → 1/1 ok"),
    ]
    assert all(p != -1 for p in positions), \
        f"missing marker(s); positions={positions}, content={content!r}"
    assert positions == sorted(positions), \
        f"markers out of order; positions={positions}"


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
    """Each marker writes to the persistent fd opened on first write;
    earlier content must survive (v0.2.5+ holds the fd for the whole
    batch instead of open/close per marker)."""
    logger = BatchLogger("abc")
    logger.write_header(
        input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
        model="qwen", quant=4, preview=False, scope=None, seed=1,
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


# ── BatchLogger ctxmgr + persistent fd (v0.2.5 — IMP-4) ────────────────


def test_batch_logger_works_as_context_manager(logs_dir):
    """`with BatchLogger(...) as logger:` is the v0.2.5 recommended
    usage. After __exit__ the fd is closed; writes after __exit__
    would fail."""
    with BatchLogger("ctxmgr1") as logger:
        logger.write_header(
            input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
            model="qwen", quant=4, preview=False, scope=None, seed=1,
        )
        assert "imgen batch" in _read(logger.path)
    # fd should be closed now.
    assert logger._fd is None


def test_batch_logger_lazy_open_no_file_until_first_write(logs_dir):
    """Constructor mkdir's LOGS_DIR but should NOT create the log file
    itself. A BatchLogger that's instantiated then abandoned (e.g.
    preflight die before write_header) must not leave an empty .log."""
    logger = BatchLogger("lazy1")

    assert not logger.path.exists(), \
        "log file must not exist until first write"

    # First write triggers open.
    logger.iteration_start(idx=1, total=1, style="x", ts=dt.datetime.now())
    assert logger.path.exists()


def test_batch_logger_close_is_idempotent(logs_dir):
    """close() called twice (e.g. via __exit__ then explicit) must not
    raise. Closes a closed fd → error."""
    logger = BatchLogger("idem1")
    logger.write_header(
        input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
        model="qwen", quant=4, preview=False, scope=None, seed=1,
    )
    logger.close()
    logger.close()  # must not raise


def test_batch_logger_close_safe_with_no_writes(logs_dir):
    """If a batch errored between construction and any write_*, the fd
    was never opened. close() should be a silent no-op, not crash."""
    logger = BatchLogger("noop1")
    logger.close()  # must not raise


def test_batch_logger_writes_after_close_raise(logs_dir):
    """After close(), any further write_* / borrow_fd MUST raise
    instead of silently re-opening the fd against a finalised batch.

    Pre-v0.2.5 review IMP-2: a stale BatchLogger reference would
    silently open a new fd on next write — relevant for v0.3.0
    batch.py's nested loops where a forgotten reference could append
    to a finished log."""
    logger = BatchLogger("closedwrite")
    logger.write_header(
        input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
        model="qwen", quant=4, preview=False, scope=None, seed=1,
    )
    logger.close()

    with pytest.raises(ValueError, match="closed"):
        logger.iteration_start(
            idx=1, total=1, style="a", ts=dt.datetime.now()
        )
    with pytest.raises(ValueError, match="closed"):
        logger.write_header(
            input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
            model="qwen", quant=4, preview=False, scope=None, seed=1,
        )
    with pytest.raises(ValueError, match="closed"):
        logger.borrow_fd()


def test_batch_logger_borrow_fd_opens_lazily(logs_dir):
    """borrow_fd() is the entry point for the subprocess stderr-tee.
    First borrow opens the fd; subsequent borrows return the same one."""
    logger = BatchLogger("borrow1")
    assert logger._fd is None

    fd1 = logger.borrow_fd()
    assert logger._fd is fd1
    assert logger.path.exists()

    fd2 = logger.borrow_fd()
    assert fd2 is fd1, "borrow_fd must return the same fd while open"


def test_batch_logger_borrowed_fd_writes_interleave_with_markers(logs_dir):
    """Critical coherence: subprocess writes via borrow_fd() and
    BatchLogger writes via iteration_* must land in temporal order
    in the file. Same Python file object, same kernel fd, same
    append-position → ordered."""
    logger = BatchLogger("interleave1")
    logger.write_header(
        input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
        model="qwen", quant=4, preview=False, scope=None, seed=1,
    )
    logger.iteration_start(
        idx=1, total=1, style="a",
        ts=dt.datetime(2026, 5, 22, 10, 0, 0),
    )
    # Simulate subprocess writing redacted bytes mid-iteration.
    fd = logger.borrow_fd()
    fd.write(b"some stderr line\n")
    fd.flush()
    logger.iteration_end(idx=1, total=1, style="a", returncode=0, duration=2)
    logger.close()

    content = _read(logger.path)
    h_pos = content.find("imgen batch")
    s_pos = content.find("[1/1] a → 2026")
    stderr_pos = content.find("some stderr line")
    e_pos = content.find(" → ok in 2s")
    # All four blocks present and in expected order.
    assert -1 < h_pos < s_pos < stderr_pos < e_pos


def test_batch_logger_marker_writes_reuse_borrowed_fd(logs_dir):
    """Tighter coherence check: the v0.2.5 persistent-fd design says
    one fd is held for the whole batch. Verify by capturing the fd
    object from borrow_fd() and asserting subsequent iteration_*
    methods write to the SAME object (not a freshly-opened one).

    Without this, a regression that re-opened per marker (the v0.2.4
    pattern) would still pass test_..._interleave_with_markers because
    POSIX O_APPEND keeps writes ordered across separate fds against
    the same path. (v0.2.5 review IMP-3)"""
    logger = BatchLogger("samefd")
    fd_at_borrow = logger.borrow_fd()

    logger.iteration_start(
        idx=1, total=1, style="a",
        ts=dt.datetime(2026, 5, 22, 10, 0, 0),
    )
    assert logger._fd is fd_at_borrow, \
        "iteration_start must write through the borrowed fd, not a new one"

    logger.iteration_end(idx=1, total=1, style="a", returncode=0, duration=1)
    assert logger._fd is fd_at_borrow

    logger.iteration_cancelled(idx=1, total=1, style="a", duration=1)
    assert logger._fd is fd_at_borrow

    logger.write_header(
        input_paths=[Path("/x")], styles=["a"], run_dir=Path("/o"),
        model="qwen", quant=4, preview=False, scope=None, seed=1,
    )
    assert logger._fd is fd_at_borrow

    logger.close()


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


# ── symlink hardening (v0.2.5 — security N2 from v0.2.4 review) ────────


def test_prune_skips_symlinks_even_if_target_is_old(logs_dir, tmp_path):
    """A user-dropped symlink under LOGS_DIR must NOT be unlinked by
    prune, regardless of the symlink target's mtime.

    Scenario: malicious or accidental `~/.imgen/logs/foo.log →
    /some/old/file`. Without lstat() the old target's stat would
    drive `removed_size` (inflating freed-bytes reporting) AND
    `log.unlink()` would silently delete the symlink entry. With
    lstat() + S_ISREG, symlinks are skipped entirely."""
    logs_dir.mkdir(mode=0o700)
    # Target is a real file outside LOGS_DIR, made old enough to fail
    # the < cutoff check if it were ever stat()'d.
    target = tmp_path / "real_target.txt"
    target.write_bytes(b"x")
    past = target.stat().st_mtime - 60 * 86400
    os.utime(target, (past, past))
    link = logs_dir / "evil.log"
    link.symlink_to(target)

    removed, _ = prune_old_batch_logs(days=30)

    assert link.is_symlink(), "symlink must survive prune"
    assert link.exists(), "and its target chain must remain intact"
    assert removed == 0, "symlinks must not be counted as pruned"


def test_prune_skips_symlinks_pointing_at_recent_target(logs_dir, tmp_path):
    """Even if the symlink itself is "old" (its target is fresh), we
    skip it on principle — only regular files belong in LOGS_DIR."""
    logs_dir.mkdir(mode=0o700)
    target = tmp_path / "fresh_target.txt"
    target.write_bytes(b"y")
    link = logs_dir / "link.log"
    link.symlink_to(target)
    # Make the symlink entry itself "old" by adjusting its lstat times.
    # On macOS lutimes is exposed via os.utime with follow_symlinks=False.
    past = link.lstat().st_mtime - 60 * 86400
    os.utime(link, (past, past), follow_symlinks=False)

    removed, _ = prune_old_batch_logs(days=30)

    assert link.is_symlink()
    assert removed == 0


def test_prune_skips_hardlinks(logs_dir, tmp_path):
    """A hardlink under LOGS_DIR (`ln /some/file ~/.imgen/logs/x.log`)
    passes S_ISREG — it IS a regular file (same inode flags as the
    target). v0.2.6 adds an `st.st_nlink == 1` check matching the
    intent "only files we created" — our own batch logs always have
    nlink=1 because nothing else links to them.

    Removing a hardlink doesn't escalate (only removes the dir entry,
    target inode survives via the other link), but the user's
    explicit hardlink-into-LOGS_DIR is an intent signal we should
    respect. (security NIT-3 from v0.2.5 review)"""
    logs_dir.mkdir(mode=0o700)
    # Real file elsewhere — we hardlink it into LOGS_DIR.
    source = tmp_path / "source.log"
    source.write_bytes(b"hardlinked content")
    past = source.stat().st_mtime - 60 * 86400
    os.utime(source, (past, past))
    hardlink = logs_dir / "linked.log"
    os.link(source, hardlink)

    # Sanity: it's truly a hardlink (nlink == 2 on both).
    assert hardlink.stat().st_nlink == 2

    removed, _ = prune_old_batch_logs(days=30)

    # Hardlink survives — neither the LOGS_DIR entry nor the source
    # file is touched.
    assert hardlink.exists()
    assert source.exists()
    assert removed == 0


def test_prune_refuses_when_logs_dir_itself_is_symlink(
    tmp_path, monkeypatch, capsys
):
    """The v0.2.5 symlink-as-log-file fix catches symlinks INSIDE
    LOGS_DIR. v0.2.6 closes the next layer: LOGS_DIR itself being a
    symlink pointing elsewhere. Without this guard, `LOGS_DIR.glob`
    walks through the symlink and could unlink files in the target
    directory. (v0.2.5 security NIT-2)

    Warns explicitly so a clean-only workflow surfaces the misconfig
    (v0.2.6 review — symmetry with ensure_logs_dir warn)."""
    import imgen.runs as runs_mod
    target = tmp_path / "target_dir"
    target.mkdir(mode=0o700)
    # Drop a fake old log into the TARGET — if the guard fails, prune
    # would walk through the symlink and remove this file.
    victim = target / "old.log"
    _make_log(victim, days_ago=60)

    logs_link = tmp_path / "logs_symlink"
    logs_link.symlink_to(target)
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs_link)

    removed, removed_bytes = prune_old_batch_logs(days=30)

    # Refusal: nothing removed, victim file in target survives.
    assert removed == 0
    assert removed_bytes == 0
    assert victim.exists()
    # User informed.
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "symlink" in combined.lower()

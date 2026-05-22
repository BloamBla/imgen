"""Per-invocation run + log helpers.

A "run" is one `imgen` invocation; in v0.2.3+ a single invocation can
produce many output files (multi-style), all dropped into the same
folder `<output-dir>/<auto_run_dirname()>/`. v0.3.0's `imgen batch <dir>`
will keep this folder-per-invocation contract, just with an N*M loop
instead of 1*M.

What lives here:

- ``auto_run_dirname`` / ``next_available_run_dir`` — folder-naming
  for the run.
- ``LOGS_DIR`` / ``LOG_RETENTION_DAYS`` / ``ensure_logs_dir`` /
  ``open_log_file_append`` — per-batch log file handling for
  multi-style runs.

These used to live in ``paths.py`` (v0.2.3); split out in v0.2.4 because
``paths.py`` was meant for pure filesystem-path constants and the run/
log helpers had grown date-formatting + retention policy + state-dir
initialisers beyond that mandate.

``paths.py`` still owns ``STATE_DIR`` and ``ensure_state_dir`` — the
run/log code here depends on those, never the other way round, keeping
the import graph acyclic (``runs`` → ``paths``, never reversed).
"""
from __future__ import annotations

import datetime as _dt
import os
import stat as _stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .paths import STATE_DIR, ensure_state_dir

__all__ = [
    "LOG_RETENTION_DAYS",
    "LOGS_DIR",
    "BatchContext",
    "BatchLogger",
    "Iteration",
    "auto_run_dirname",
    "ensure_logs_dir",
    "next_available_run_dir",
    "open_log_file_append",
    "prune_old_batch_logs",
]


@dataclass(frozen=True, slots=True)
class BatchContext:
    """Batch-wide constants threaded into every iteration.

    cmd_generate (and v0.3.0 commands/batch.py) builds this once before
    the run loop and passes it whole into _run_one_iteration. Replaces
    the 9 individual kwargs that were threaded through v0.2.4's
    16-arg signature — keeps the call site legible and makes nested
    N×M loops in batch.py tractable. (architect IMP-3 from v0.2.4 review)

    Frozen because every iteration sees the same values; slots so
    typos on field access raise instead of silently registering on
    __dict__. ``__hash__`` is explicitly disabled — `env: dict` and
    `args: Namespace` aren't hashable, so the dataclass-auto-generated
    __hash__ would crash on first `hash(ctx)` (v0.2.5 review IMP-1).

    `args` is the parsed argparse Namespace; typed Any to avoid
    importing argparse here just for an annotation.
    `env` is a dict snapshot — frozen=True prevents reassignment of
    the field, not mutation of the dict itself, but callers treat it
    as read-only by convention.
    """
    backend: str
    seed: int
    width: int
    height: int
    input_path: Path
    effective_custom_prompt: str | None
    args: Any
    batch_id: str | None
    env: dict[str, str]

    # Opt out of hashing — dict/Namespace fields make the auto-generated
    # __hash__ blow up on any caller that tries to use BatchContext as
    # a set member or dict key. Equality (__eq__) still works.
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class Iteration:
    """One (input, style) generation slot inside a single CLI invocation.

    cmd_generate pre-builds the whole list before any subprocess work so
    dry-run can show every entry and resource preflight runs against the
    heaviest quant in the batch. Frozen so the post-build loop can't
    accidentally mutate an entry mid-iteration; slots so typos in field
    access raise instead of landing on __dict__.

    Field order matches the legacy dict-of-strings shape (style_name,
    prompt, negative, final_steps, ...) so the v0.2.4 extraction is a
    mechanical replace with no semantic shift.
    """
    style_name: str
    prompt: str
    negative: str
    final_steps: int
    final_quantize: int
    final_guidance: float
    final_strength: float
    output_path: Path
    cmd: list[str]

    # Same opt-out reasoning as BatchContext: `cmd: list[str]` is not
    # hashable, so a caller using Iteration as a set/dict-key would
    # hit TypeError at first hash. Equality still works for test
    # assertions on _build_iterations output.
    __hash__ = None  # type: ignore[assignment]

# Per-batch logs (v0.2.3+) — one .log file per multi-style invocation,
# named after batch_id. Single-style generations don't write here.
# Retention is enforced by `imgen clean` (30 days).
LOGS_DIR = STATE_DIR / "logs"
LOG_RETENTION_DAYS = 30


def auto_run_dirname(now: _dt.datetime | None = None) -> str:
    """Folder name for one CLI invocation: '2026-05-21-14-30-12'.

    All-dashes, second precision. Sortable alphabetically = sortable
    chronologically. No colons (`:`) so the path survives macOS quirks
    and copy-paste into terminals that quote-mangle `:`. We run mflux
    serially so no two generations within one invocation finish in the
    same second; folder-level collisions only arise from scripted
    double-invokes, handled by `next_available_run_dir`.
    """
    if now is None:
        now = _dt.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def next_available_run_dir(parent: Path, dirname: str) -> Path:
    """Return parent/dirname, suffixing `_2`, `_3` if it already exists.

    Pure: does NOT create the directory. Caller mkdir's the returned
    path after the user passes any confirm gates (so a cancel doesn't
    orphan an empty dir).

    Probe-then-caller-mkdir has a tiny race window between this call
    and the eventual `mkdir(parents=True, exist_ok=True)`. For
    single-user serial CLI usage two `imgen` invocations would have to
    start within the same second AND target the same auto_run_dirname()
    to collide — and even then `mkdir(exist_ok=True)` makes both
    succeed and share the run folder (files inside still collide on
    `<basename>-<style>.png` only if both use the same input + style).
    Documented limitation, not a target for atomic-claim today.
    """
    target = parent / dirname
    if not target.exists():
        return target
    i = 2
    while (parent / f"{dirname}_{i}").exists():
        i += 1
    return parent / f"{dirname}_{i}"


def open_log_file_append(path: Path) -> BinaryIO:
    """Open a log file for binary append with 0o600 perms from creation.

    Used by per-batch logs (multi-style runs). Default umask on macOS
    would give 0o644 — world-readable by other users on a shared Mac.
    LOGS_DIR is already 0o700, but defence-in-depth: keep the files
    themselves restrictive too, matching how ~/.imgen/hf_token is
    handled. Token redaction in subprocess_helpers covers HF tokens
    in the content; this guards against everything else in mflux's
    stderr (paths, model traces, scope hints).

    Returns a buffered binary file-like object — callers must encode()
    any strings they want to write.

    **DO NOT remove `O_APPEND`.** It's the kernel-level atomicity
    guarantee that lets BatchLogger.borrow_fd() share the underlying
    fd with subprocess_helpers' stderr-tee — both writers append at
    "current end of file" with no interleaving mid-write, regardless
    of internal Python-buffer offsets. Without O_APPEND, the borrowed
    fd's writes would race with BatchLogger's marker writes for file
    position. (architect NIT-3 from v0.2.5 review)
    """
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "ab")


def _logs_dir_is_safe() -> bool:
    """Return False (and warn) if LOGS_DIR is a symlink.

    Centralizes the v0.2.5/v0.2.6 LOGS_DIR symlink guard shared by
    ``ensure_logs_dir`` and ``prune_old_batch_logs``. (architect NIT-2
    from v0.2.6 review — extracted ahead of the third caller, since
    the runs/logs surface is going to keep growing and re-rolling
    the same guard inline guarantees wording drift.)

    Why we guard at all:
      * ``ensure_logs_dir`` would otherwise chmod 0o700 on whatever the
        symlink targets — could be an unrelated user dir or a shared
        mount.
      * ``prune_old_batch_logs`` would otherwise let ``LOGS_DIR.glob``
        walk into the target tree and unlink files there.
      * Both gaps require an explicit `ln -s` over ``~/.imgen/logs/`` —
        won't happen in normal installs, but cheap to refuse.

    Warn (not silent return) is deliberate: a clean-only workflow that
    never runs ``imgen generate`` would otherwise see "0 logs removed"
    forever and never learn why the dir is misconfigured.

    Trust-boundary scope: catches a symlink AT LOGS_DIR only. A
    symlinked STATE_DIR parent is NOT detected — see
    ``paths.ensure_state_dir`` docstring for why that's deliberate.
    """
    if LOGS_DIR.is_symlink():
        from .colors import warn
        warn(f"LOGS_DIR is a symlink ({LOGS_DIR}); refusing to operate. "
             "Remove the symlink or relocate ~/.imgen/logs/ to a real "
             "directory.")
        return False
    return True


def ensure_logs_dir() -> None:
    """Create LOGS_DIR (0o700) under STATE_DIR.

    Used by BatchLogger and direct callers (cmd_generate when it opens
    a per-batch log for multi-style runs). STATE_DIR is created first
    so a fresh user never hits ENOENT.

    Refuses to operate (warn + return) if LOGS_DIR is a symlink — see
    ``_logs_dir_is_safe`` for the rationale.
    """
    ensure_state_dir()
    if not _logs_dir_is_safe():
        return
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(mode=0o700)
    elif (LOGS_DIR.stat().st_mode & 0o777) != 0o700:
        try:
            LOGS_DIR.chmod(0o700)
        except OSError:
            pass


# ── BatchLogger + retention ─────────────────────────────────────────────


class BatchLogger:
    """Owner of one multi-style invocation's log file.

    v0.2.3 had the per-batch log lifecycle split across three modules
    (header in commands/generate.py, iteration markers in the same
    file's loop, stderr-tee in subprocess_helpers.py, retention in
    commands/clean.py). v0.2.4 (architect item I3) collapsed the
    header + marker concerns into this class. v0.2.5 (architect item
    IMP-4) folds in fd ownership too — the file is opened lazily on
    first write and held for the batch lifetime, instead of
    open/flush/close per marker. At N×M=50 in v0.3.0 that saves ~200
    open/close syscalls per batch.

    Single-style runs do NOT create a BatchLogger; caller gates on
    is_batch. The log *directory* is created eagerly in ``__init__``
    (LOGS_DIR mkdir + chmod 0o700); the log *file* is created lazily
    on first write via open_log_file_append, so a BatchLogger whose
    batch never writes (e.g. an error between construction and
    write_header) doesn't leave an empty file behind — but it may
    leave an empty LOGS_DIR if it didn't exist already, which is
    benign (0o700, no content).

    Lifecycle:
      * ``BatchLogger(batch_id)`` — mkdir's LOGS_DIR, sets self.path
      * first call to write_header / iteration_* / borrow_fd — opens
        the underlying file via open_log_file_append (0o600 from
        creation)
      * ``close()`` — flushes + closes the fd if it was opened
      * also usable as a context manager (``with BatchLogger(...) as
        logger: ...``) — __exit__ calls close().

    The subprocess stderr-tee borrows the open fd via
    ``borrow_fd()`` and writes redacted bytes through the same file
    object. BatchLogger remains the owner — subprocess_helpers does
    NOT close it. (architect FWD-6 from v0.2.4 review)
    """
    __slots__ = ("_batch_id", "_closed", "_fd", "path")

    def __init__(self, batch_id: str) -> None:
        ensure_logs_dir()
        self._batch_id = batch_id
        self._fd: BinaryIO | None = None
        self._closed = False
        self.path: Path = LOGS_DIR / f"{batch_id}.log"

    def __enter__(self) -> "BatchLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _ensure_open(self) -> BinaryIO:
        """Open the log file lazily on first write.

        Raises ValueError if the logger has been closed. v0.2.5 review
        IMP-2: a stale BatchLogger reference revived itself silently
        under the previous design — a forgotten reference in v0.3.0
        batch.py's nested loop would have opened a fresh fd against
        an already-finalised batch log.
        """
        if self._closed:
            raise ValueError(
                "BatchLogger is closed — cannot write to a finalised batch"
            )
        if self._fd is None:
            self._fd = open_log_file_append(self.path)
        return self._fd

    def close(self) -> None:
        """Flush and close the underlying fd if it was opened.

        Safe to call multiple times (second call is a no-op). Safe to
        call when no writes happened (fd was never opened — also no-op).
        After close, any further write_* / borrow_fd call raises
        ValueError instead of silently re-opening (v0.2.5 review IMP-2).
        """
        if self._fd is not None:
            try:
                self._fd.flush()
            finally:
                self._fd.close()
                self._fd = None
        self._closed = True

    def borrow_fd(self) -> BinaryIO:
        """Return the open (lazy-opened if needed) file object so the
        subprocess stderr-tee can write redacted bytes through it.

        Caller writes + flushes but MUST NOT close — BatchLogger owns
        the lifecycle via its own close() / __exit__.
        """
        return self._ensure_open()

    def write_header(
        self,
        *,
        input_paths: list[Path],
        styles: list[str],
        run_dir: Path | None,
        backend: str,
        quant: int,
        preview: bool,
        scope: str | None,
        seed: int,
        now: _dt.datetime | None = None,
    ) -> None:
        """Write the ``# imgen batch <id>`` header block.

        ``input_paths`` is a list — v0.3.0 unified shape covering both
        ``imgen generate`` (one input × M styles) and ``imgen batch``
        (N inputs × M styles). Renders ``# inputs (N):  name1, name2,
        ...`` with basenames only (full paths would balloon the line
        when the input dir is deep).

        Empty ``input_paths`` is a caller bug — every caller (cmd_generate
        / cmd_batch) validates upstream, but a defensive guard surfaces
        a regression here instead of writing a confusing ``inputs (0):``
        line. (v0.3.0)

        ``now`` defaults to wall-clock time but is injectable for tests.
        """
        if not input_paths:
            raise ValueError(
                "write_header requires non-empty input_paths "
                "(callers validate upstream)"
            )
        started = (now if now is not None else _dt.datetime.now()).isoformat(
            timespec="seconds"
        )
        input_names = ", ".join(p.name for p in input_paths)
        styles_names = ", ".join(styles)
        header = (
            f"# imgen batch {self._batch_id}\n"
            f"# started:  {started}\n"
            f"# inputs ({len(input_paths)}):  {input_names}\n"
            f"# styles ({len(styles)}):  {styles_names}\n"
            f"# output:   {run_dir}\n"
            f"# backend:  {backend} q{quant}  "
            f"preview={preview}  scope={scope}  seed={seed}\n"
        )
        fd = self._ensure_open()
        fd.write(header.encode())
        fd.flush()

    def input_section_start(
        self, idx_input: int, total_inputs: int, name: str
    ) -> None:
        """Open a per-input section block in the log (v0.3.0 batch).

        Marker shape: ``=== INPUT <name> (k/N) ===``. Bookends the M
        iteration blocks for one input, so ``tail -f`` users can see
        progress at the input level without counting iteration markers.

        Single-input runs (``imgen generate``) do NOT call this — only
        ``cmd_batch`` opens sections. Keeps the v0.2.x single-input
        log shape unchanged for that path.
        """
        marker = f"\n=== INPUT {name} ({idx_input}/{total_inputs}) ===\n"
        fd = self._ensure_open()
        fd.write(marker.encode())
        fd.flush()

    def input_section_end(
        self,
        *,
        idx_input: int,
        total_inputs: int,
        name: str,
        ok_count: int,
        fail_count: int,
        duration: int,
    ) -> None:
        """Close a per-input section with the per-input outcome summary.

        Shape: ``=== INPUT <name> → <ok>/<total> ok in <dur>s ===`` for
        all-success; appends ``fail=<n>`` when any iteration failed so
        ``grep 'INPUT.*fail='`` finds every partially-failed input in
        one pass.

        ``ok_count + fail_count`` equals the number of styles attempted
        for this input — the caller (cmd_batch) is responsible for the
        running tallies (mirrors how ``_run_one_iteration`` mutates
        ``succeeded`` / ``failed`` lists for the whole batch).
        """
        total = ok_count + fail_count
        if fail_count == 0:
            marker = (f"\n=== INPUT {name} → {ok_count}/{total} ok "
                      f"in {duration}s ===\n")
        else:
            marker = (f"\n=== INPUT {name} → {ok_count}/{total} ok "
                      f"fail={fail_count} in {duration}s ===\n")
        fd = self._ensure_open()
        fd.write(marker.encode())
        fd.flush()

    def iteration_start(
        self, idx: int, total: int, style: str, ts: _dt.datetime
    ) -> None:
        """Write the `=== [idx/total] style → <ts> ===` start marker."""
        marker = (f"\n=== [{idx}/{total}] {style} → "
                  f"{ts.isoformat(timespec='seconds')} ===\n")
        fd = self._ensure_open()
        fd.write(marker.encode())
        fd.flush()

    def iteration_end(
        self, idx: int, total: int, style: str, returncode: int, duration: int
    ) -> None:
        """Write the end marker: ` ok ` on success, `FAILED exit=N` on
        non-zero returncode."""
        status = "ok" if returncode == 0 else f"FAILED exit={returncode}"
        marker = (f"\n=== [{idx}/{total}] {style} → {status} "
                  f"in {duration}s ===\n")
        fd = self._ensure_open()
        fd.write(marker.encode())
        fd.flush()

    def iteration_cancelled(
        self, idx: int, total: int, style: str, duration: int
    ) -> None:
        """Write the CANCELLED marker (KeyboardInterrupt mid-mflux)."""
        marker = (f"\n=== [{idx}/{total}] {style} → "
                  f"CANCELLED in {duration}s ===\n")
        fd = self._ensure_open()
        fd.write(marker.encode())
        fd.flush()


def prune_old_batch_logs(
    days: int, dry_run: bool = False
) -> tuple[int, int]:
    """Delete .log files in LOGS_DIR with mtime older than `days`.

    Returns ``(count_removed, bytes_removed)`` so the UX layer (clean.py)
    can phrase the user message. ``dry_run=True`` counts without
    deleting. Non-existent LOGS_DIR (fresh user who never ran a batch)
    is silent: returns (0, 0). Only `*.log` files are touched —
    a user-dropped `notes.txt` survives intentionally.

    OSError on individual files is swallowed (file may have been
    removed mid-glob; the next pass picks up the rest).
    """
    if not LOGS_DIR.exists():
        return 0, 0
    if not _logs_dir_is_safe():
        # See ``_logs_dir_is_safe`` for the full rationale (centralized
        # in v0.2.6 review NIT-2 closure). Warn is symmetric with the
        # ensure_logs_dir path so a clean-only workflow surfaces the
        # misconfiguration instead of silently returning "0 removed".
        return 0, 0
    cutoff = _dt.datetime.now().timestamp() - days * 86400
    removed = 0
    removed_size = 0
    for log in LOGS_DIR.glob("*.log"):
        try:
            # lstat (NOT stat) so a user-dropped symlink under LOGS_DIR
            # (e.g. evil.log → /etc/passwd) doesn't get its TARGET's size
            # counted toward removed_size and its symlink entry doesn't
            # get unlinked silently. S_ISREG filters out symlinks, dirs,
            # special files — only regular files inside LOGS_DIR are
            # batch logs we own. (security N2 from v0.2.4 review)
            #
            # st_nlink == 1 additionally skips hardlinks — our own batch
            # logs always have nlink=1; a hardlink into LOGS_DIR is a
            # user-intent signal pointing at something they explicitly
            # want preserved. Unlinking wouldn't escalate (just removes
            # the dir entry, inode survives via other links), but
            # respecting the signal matches "only files we created".
            # (security NIT-3 from v0.2.5 review)
            #
            # Snapshot once — separate stat calls would let st_size
            # raise OSError after st_mtime succeeded, leaving the two
            # counters out of sync. (python C2 from v0.2.3 review)
            st = log.lstat()
            if not _stat.S_ISREG(st.st_mode):
                continue
            if st.st_nlink != 1:
                continue
            if st.st_mtime < cutoff:
                removed_size += st.st_size
                if not dry_run:
                    log.unlink()
                removed += 1
        except OSError:
            pass
    return removed, removed_size

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
    __dict__.

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
    """
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "ab")


def ensure_logs_dir() -> None:
    """Create LOGS_DIR (0o700) under STATE_DIR.

    Used by BatchLogger and direct callers (cmd_generate when it opens
    a per-batch log for multi-style runs). STATE_DIR is created first
    so a fresh user never hits ENOENT.
    """
    ensure_state_dir()
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
    commands/clean.py). v0.2.4 (architect item I3) collapses the
    header + marker concerns into this class so future log-shape
    changes — per-image rotation, structured JSON markers,
    per-(input, style) row in v0.3.0 — happen in one place.

    Single-style runs do NOT create a BatchLogger; caller gates on
    is_batch. The log *directory* is created eagerly in ``__init__``
    (LOGS_DIR mkdir + chmod 0o700); the log *file* is created lazily
    on first write via open_log_file_append, so a BatchLogger whose
    batch never writes (e.g. an error between construction and
    write_header) doesn't leave an empty file behind — but it may
    leave an empty LOGS_DIR if it didn't exist already, which is
    benign (0o700, no content).

    The subprocess stderr-tee still owns its own fd inside
    run_with_stderr_redaction (it needs raw bytes, not formatted
    markers). The BatchLogger's .path is what that helper opens.
    """
    __slots__ = ("_batch_id", "path")

    def __init__(self, batch_id: str) -> None:
        ensure_logs_dir()
        self._batch_id = batch_id
        self.path: Path = LOGS_DIR / f"{batch_id}.log"

    def write_header(
        self,
        *,
        input_path: Path,
        styles: list[str],
        run_dir: Path | None,
        backend: str,
        quant: int,
        preview: bool,
        scope: str | None,
        seed: int,
        now: _dt.datetime | None = None,
    ) -> None:
        """Write the # imgen batch <id> header block.

        `now` defaults to wall-clock time but is injectable for tests.
        """
        started = (now if now is not None else _dt.datetime.now()).isoformat(
            timespec="seconds"
        )
        header = (
            f"# imgen batch {self._batch_id}\n"
            f"# started:  {started}\n"
            f"# input:    {input_path}\n"
            f"# styles:   {', '.join(styles)}\n"
            f"# output:   {run_dir}\n"
            f"# backend:  {backend} q{quant}  "
            f"preview={preview}  scope={scope}  seed={seed}\n"
        )
        with open_log_file_append(self.path) as f:
            f.write(header.encode())

    def iteration_start(
        self, idx: int, total: int, style: str, ts: _dt.datetime
    ) -> None:
        """Write the `=== [idx/total] style → <ts> ===` start marker."""
        marker = (f"\n=== [{idx}/{total}] {style} → "
                  f"{ts.isoformat(timespec='seconds')} ===\n")
        with open_log_file_append(self.path) as f:
            f.write(marker.encode())

    def iteration_end(
        self, idx: int, total: int, style: str, returncode: int, duration: int
    ) -> None:
        """Write the end marker: ` ok ` on success, `FAILED exit=N` on
        non-zero returncode."""
        status = "ok" if returncode == 0 else f"FAILED exit={returncode}"
        marker = (f"\n=== [{idx}/{total}] {style} → {status} "
                  f"in {duration}s ===\n")
        with open_log_file_append(self.path) as f:
            f.write(marker.encode())

    def iteration_cancelled(
        self, idx: int, total: int, style: str, duration: int
    ) -> None:
        """Write the CANCELLED marker (KeyboardInterrupt mid-mflux)."""
        marker = (f"\n=== [{idx}/{total}] {style} → "
                  f"CANCELLED in {duration}s ===\n")
        with open_log_file_append(self.path) as f:
            f.write(marker.encode())


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
            # Snapshot once — separate stat calls would let st_size
            # raise OSError after st_mtime succeeded, leaving the two
            # counters out of sync. (python C2 from v0.2.3 review)
            st = log.lstat()
            if not _stat.S_ISREG(st.st_mode):
                continue
            if st.st_mtime < cutoff:
                removed_size += st.st_size
                if not dry_run:
                    log.unlink()
                removed += 1
        except OSError:
            pass
    return removed, removed_size

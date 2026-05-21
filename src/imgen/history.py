"""History persistence: load + atomic-append under flock.

History entries live in ~/.imgen/history.jsonl (mode 0600). Each entry gets
a monotonic `id` assigned under an exclusive flock (so parallel `--force`
runs don't collide on the counter) and a schema version `v` so future field
changes can refuse-to-replay rather than misinterpret old entries.
"""
from __future__ import annotations

import fcntl
import json
import os

from .colors import err, warn
from .defaults import HISTORY_SCHEMA_VERSION
from .paths import HISTORY_FILE, ensure_state_dir


def load_history() -> list[dict]:
    """Read history line-by-line so a massive file doesn't blow up RAM.

    Malformed lines (e.g. from a partial write / disk-full crash) are
    skipped with a warn rather than silently dropped — the user knows
    a record was lost.
    """
    if not HISTORY_FILE.exists():
        return []
    entries = []
    try:
        with HISTORY_FILE.open("r") as f:
            for lineno, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError:
                    warn(f"history.jsonl:{lineno}: skipping malformed line")
    except OSError:
        return []
    return entries


def append_history(entry: dict) -> int:
    """Assign id and append entry atomically under an exclusive flock.

    Returns the assigned id. Locking ensures parallel runs don't collide
    on the id counter, and POSIX append-mode + flock prevents interleaving
    of long lines that exceed PIPE_BUF.

    Does NOT mutate the caller's dict — the stored record is a local copy
    augmented with `id` + `v`. Earlier behaviour wrote those keys into the
    caller's dict, which is a hidden side-effect.

    Re-applies mode 0o600 under the lock even if the file already existed
    (e.g. user upgraded from v0.1.0 where the file was 0o644 before the
    v0.1.1 chmod fix). os.open's mode= argument is honoured only on
    creation, so we need an explicit fchmod for upgrade safety.
    """
    ensure_state_dir()

    # Hand off raw fd to a file object inside its own try so a failure of
    # os.fdopen doesn't leak the fd.
    try:
        fd = os.open(str(HISTORY_FILE),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError as e:
        err(f"Failed to open history file: {e}")
        return entry.get("id", 0)
    try:
        f = os.fdopen(fd, "a")
    except OSError as e:
        os.close(fd)
        err(f"Failed to wrap history fd: {e}")
        return entry.get("id", 0)
    # fd is owned by f from here on; closes on `with` exit.

    try:
        with f:
            os.fchmod(f.fileno(), 0o600)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                existing = load_history()
                assigned_id = max(
                    (e.get("id", 0) for e in existing), default=0
                ) + 1
                stored = {
                    **entry,
                    "id": assigned_id,
                    "v": HISTORY_SCHEMA_VERSION,
                }
                f.write(json.dumps(stored, ensure_ascii=False) + "\n")
                return assigned_id
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        err(f"Failed to write history: {e}")
        return entry.get("id", 0)

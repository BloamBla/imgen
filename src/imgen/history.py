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

from .colors import err
from .defaults import HISTORY_SCHEMA_VERSION
from .paths import HISTORY_FILE, ensure_state_dir


def load_history() -> list[dict]:
    """Read history line-by-line so a massive file doesn't blow up RAM."""
    if not HISTORY_FILE.exists():
        return []
    entries = []
    try:
        with HISTORY_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate corrupted lines
    except OSError:
        return []
    return entries


def append_history(entry: dict) -> int:
    """Assign id and append entry atomically under an exclusive flock.

    Returns the assigned id. Locking ensures parallel runs don't collide
    on the id counter, and POSIX append-mode + flock prevents interleaving
    of long lines that exceed PIPE_BUF.
    """
    ensure_state_dir()
    fd = os.open(str(HISTORY_FILE),
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                existing = load_history()
                entry["id"] = max(
                    (e.get("id", 0) for e in existing), default=0
                ) + 1
                entry["v"] = HISTORY_SCHEMA_VERSION
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return entry["id"]
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        err(f"Failed to write history: {e}")
        return entry.get("id", 0)

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

from ._safe import has_control_bytes as _has_control_bytes
from .colors import err, warn
from .defaults import HISTORY_SCHEMA_VERSION
from .paths import HISTORY_FILE, ensure_state_dir

__all__ = ["append_history", "entry_model_name", "load_history"]


def entry_model_name(entry: dict) -> str | None:
    """Resolve the v0.8 canonical model name from a history entry,
    regardless of schema version.

    Dispatch order (v=4 wins, v=3 fallback):
      * v=4 entries carry ``model`` (commit 9 schema rename).
      * v=3 entries carry ``backend`` (v0.7.x and earlier shape).
      * Value runs through ``_V07_TO_V08_MODEL_RENAMES`` so a v=3
        ``"flux"`` entry renders / replays / ETA-matches as
        ``"flux-kontext"`` (the v0.8 canonical name).

    Returns ``None`` when:
      * neither key is present (very old entries, hand-edited rows);
      * the stored value is not a string (defensive — hand-edited
        JSONL with a typo'd shape);
      * the value contains C0/DEL/C1 control bytes (§A.5 security:
        replay must not feed a dirty string into argv; list/ETA
        callers downgrade to "no match" rather than rendering
        garbage in the user's terminal).

    Pure: no I/O, no registry lookup. Deliberately does NOT call
    ``get_backend()`` — architect 4b pre-vet M-3: a registry lookup
    at history-read time would crash ``imgen history --last`` if a
    referenced user TOML was deleted. Display/listing must not
    require a live registry.

    Replaces the v0.8.0 commit 4b-era ``_normalize_backend_value``
    helper in ``commands/history.py``: that one only handled the
    rename map; this one folds in the dual-shape read AND the §A.5
    filter. HIGH-1 fix (cmd_helpers ETA matcher, §R.3) lives here.
    """
    from .models import _V07_TO_V08_MODEL_RENAMES
    raw = entry.get("model")
    if raw is None:
        raw = entry.get("backend")
    if raw is None or not isinstance(raw, str):
        return None
    if _has_control_bytes(raw):
        return None
    return _V07_TO_V08_MODEL_RENAMES.get(raw, raw)


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
                # load_history() opens a SEPARATE read fd without flock —
                # that's intentional and safe. Our LOCK_EX serializes
                # ALL writers (any second imgen process trying to
                # append blocks on flock above), so the file is stable
                # for the duration of our critical section. Unlocked
                # readers (this load_history call) see a consistent
                # state because no concurrent writer can interleave.
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

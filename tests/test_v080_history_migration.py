"""v0.8.0 commit 9 — history schema v=3 → v=4 lock-in suite.

Per [[project-v080-design]] §K + §Q commit 9:

* history schema bumps v=3 → v=4 with KEY RENAME ``backend`` →
  ``model``;
* ``history.entry_model_name(entry)`` provides dual-shape read
  dispatch (v=4 ``model`` wins, v=3 ``backend`` fallback) plus rename-
  map translation (v=3 "flux" → v0.8 "flux-kontext") plus §A.5
  control-byte filter (security — replay must not feed a dirty
  string into argv);
* the §R.3 mid-impl review HIGH-1 fix lives here: cmd_helpers ETA
  matcher now goes through ``entry_model_name`` so a user with v=3
  "flux" history rows gets ETA on a v0.8 "flux-kontext" run.

All tests are pure-Python — no mflux subprocess, no GPU, no network.
"""
from __future__ import annotations

import json

import pytest

from imgen.cmd_helpers import estimate_one_seconds
from imgen.commands.history import cmd_history
from imgen.defaults import HISTORY_SCHEMA_VERSION
from imgen.history import (
    append_history,
    entry_model_name,
    load_history,
)


# ── entry_model_name dispatch ────────────────────────────────────────


def test_history_v4_entry_writes_model_not_backend(tmp_state_dir):
    """Lock-in: new entries written via append_history land with the
    v=4 "model" key, not the v=3 "backend" key. Prevents accidental
    schema-downgrade in a future commit that re-introduces a
    ``"backend": ...`` literal at the writer side."""
    append_history({"model": "flux-kontext", "status": "success"})
    [entry] = load_history()
    assert entry["v"] == 4
    assert entry["model"] == "flux-kontext"
    assert "backend" not in entry


def test_history_v3_entry_replays_under_v4_dispatch(tmp_state_dir):
    """A v=3 row on disk (``"backend": "flux"`` + ``"v": 3``) resolves
    through entry_model_name() to its v0.8 canonical name. The rename
    map fires regardless of which key carried the value."""
    v3_entry = {"v": 3, "backend": "flux", "status": "success"}
    assert entry_model_name(v3_entry) == "flux-kontext"

    # Same value under the v=4 key resolves identically (no rename
    # needed; "flux-kontext" is already canonical).
    v4_entry = {"v": 4, "model": "flux-kontext", "status": "success"}
    assert entry_model_name(v4_entry) == "flux-kontext"


def test_history_v4_key_wins_over_v3_key_on_mixed_entry(tmp_state_dir):
    """Defence-in-depth: a hand-edited row with BOTH keys present
    resolves via "model" (the v=4 winner). Prevents a future writer
    accidentally leaving the legacy "backend" key alongside the new
    "model" key from silently flipping the resolution."""
    mixed = {"model": "flux-kontext", "backend": "qwen", "status": "success"}
    assert entry_model_name(mixed) == "flux-kontext"


def test_history_entry_model_name_returns_none_when_both_keys_absent():
    """Entries with neither key (pre-v0.6 / hand-edited) return None
    so the caller can decide on a subcommand-appropriate default
    (rather than the helper guessing)."""
    assert entry_model_name({"status": "success"}) is None


def test_history_control_byte_filter_rejects_dirty_model_field():
    """§A.5 security lock-in: a history entry whose model value
    contains C0/DEL/C1 control bytes (e.g. hand-edited JSONL with
    embedded ANSI escape) returns None from entry_model_name. The
    replay call sites translate None → subcommand default, so the
    dirty string never reaches argv."""
    # \x1b (ESC) is a C0 byte (< 0x20) — common in injected ANSI
    # escape sequences. \x07 (BEL) is another C0 representative.
    dirty_v4 = {"model": "flux-\x1bkontext", "v": 4}
    dirty_v3 = {"backend": "flux\x07", "v": 3}
    assert entry_model_name(dirty_v4) is None
    assert entry_model_name(dirty_v3) is None

    # Sanity: a clean v=4 entry still resolves correctly.
    clean = {"model": "flux-kontext", "v": 4}
    assert entry_model_name(clean) == "flux-kontext"


# ── HIGH-1 regression: ETA matcher matches v=3 entries after rename ──


def test_estimate_one_seconds_matches_v3_entries_after_rename(tmp_state_dir):
    """v0.8.0 §R.3 HIGH-1 fix: a user upgrading from v0.7 has a
    history.jsonl full of v=3 ``"backend": "flux"`` rows. Their first
    v0.8 ``imgen generate`` run resolves args.model → ``"flux-kontext"``
    (the v0.8 canonical name); the ETA matcher must find the v=3
    rows as matches via the rename map, not skip them and show
    "no data available".

    Pre-fix the matcher compared ``e.get("backend") == backend``
    directly — v=3 "flux" never equalled v0.8 "flux-kontext" so ETA
    went cold until 5 new post-upgrade entries accumulated. This
    test locks in the fix so a future "optimization" can't regress
    silently.
    """
    # 3 v=3 success rows with the legacy "backend":"flux" shape
    v3_rows = [
        {"status": "success", "v": 3, "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 300},
        {"status": "success", "v": 3, "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 360},
        {"status": "success", "v": 3, "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 420},
    ]
    # avg(300, 360, 420) = 360
    assert estimate_one_seconds(
        v3_rows, "flux-kontext", 4, False,
    ) == 360


def test_estimate_one_seconds_mixed_v3_v4_entries_average_together(
    tmp_state_dir,
):
    """Defence-in-depth on the §R.3 HIGH-1 fix: a transition history
    with BOTH v=3 and v=4 rows for the same model resolution averages
    them together (rather than one shape silently dominating)."""
    mixed = [
        # v=3 row: legacy shape
        {"status": "success", "v": 3, "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 300},
        # v=4 row: new shape with v0.8 canonical name
        {"status": "success", "v": 4, "model": "flux-kontext", "quantize": 4,
         "preview": False, "duration_sec": 360},
    ]
    # avg(300, 360) = 330
    assert estimate_one_seconds(
        mixed, "flux-kontext", 4, False,
    ) == 330


# ── list-render unified column (architect IMPORTANT lock-in) ─────────


def test_history_list_unified_column_across_v3_and_v4_entries(
    tmp_state_dir, capsys,
):
    """Architect IMPORTANT (§Q commit 9): ``imgen history`` list output
    renders the v0.8 canonical model name consistently for both v=3
    rows (``"backend":"flux"``) and v=4 rows (``"model":"flux-kontext"``).
    Without unification, a user's list would show alternating "flux"
    and "flux-kontext" depending on when each row was written —
    confusing and inconsistent with the v0.8 ``--list-models`` /
    ``--model`` flag surface.
    """
    # Pre-populate history.jsonl directly to control the schema
    # version of each row (append_history always writes v=4 now).
    # Late import — conftest's monkeypatch rebinds HISTORY_FILE
    # per test, so module-top imports see the pre-patch path.
    from imgen.history import HISTORY_FILE
    HISTORY_FILE.write_text(
        json.dumps({
            "id": 1, "v": 3, "ts": "2026-05-25T10:00:00",
            "backend": "flux", "style": "anime",
            "input": "/a.jpg", "output": "/a.png",
            "status": "success",
        }) + "\n"
        + json.dumps({
            "id": 2, "v": 4, "ts": "2026-05-26T10:00:00",
            "model": "flux-kontext", "style": "pixar",
            "input": "/b.jpg", "output": "/b.png",
            "status": "success",
        }) + "\n",
    )

    # Strip ANSI for assertion stability.
    import argparse
    cmd_history(argparse.Namespace(last=10))
    out = capsys.readouterr().out

    # Both rows show "flux-kontext" in the model column — v=3 went
    # through the rename map, v=4 was already canonical.
    assert out.count("flux-kontext") == 2
    # The legacy raw "flux" name (sans -kontext) must NOT appear —
    # if it does, the v=3 row leaked its pre-rename value into the
    # display. Negative-match anchored to a word boundary so
    # "flux-kontext" itself doesn't accidentally satisfy the check.
    import re
    assert not re.search(r"\bflux\b(?!-kontext)", out), (
        "v=3 row's legacy 'flux' name leaked into list display "
        "without rename-map translation"
    )

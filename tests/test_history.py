"""History persistence + schema-version refuse-on-future.

Uses the tmp_state_dir fixture so HISTORY_FILE is per-test, no real
~/.imgen/history.jsonl is touched.
"""
from __future__ import annotations

import json

import pytest

from imgen.defaults import HISTORY_SCHEMA_VERSION
from imgen.history import append_history, load_history


def test_load_history_empty_returns_empty_list(tmp_state_dir):
    assert load_history() == []


def test_append_history_assigns_v_and_monotonic_id(tmp_state_dir):
    id1 = append_history({"input": "/a.jpg", "output": "/a.png"})
    id2 = append_history({"input": "/b.jpg", "output": "/b.png"})
    id3 = append_history({"input": "/c.jpg", "output": "/c.png"})
    assert (id1, id2, id3) == (1, 2, 3)


def test_append_history_sets_schema_version(tmp_state_dir):
    """The STORED record (not the caller's dict) gets `v` stamped. The
    earlier version of this test asserted mutation on the caller dict —
    that was a side-effect bug; the test now reads back via load_history
    and verifies the persisted record has the schema version."""
    append_history({"input": "/a.jpg", "output": "/a.png"})
    [entry] = load_history()
    assert entry["v"] == HISTORY_SCHEMA_VERSION


def test_load_history_roundtrips_appended_entries(tmp_state_dir):
    append_history({"input": "/a.jpg", "output": "/a.png", "style": "anime"})
    append_history({"input": "/b.jpg", "output": "/b.png", "style": "pixar"})
    entries = load_history()
    assert len(entries) == 2
    assert entries[0]["style"] == "anime"
    assert entries[1]["style"] == "pixar"
    # ids reassigned in order, v stamped
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2
    assert entries[0]["v"] == HISTORY_SCHEMA_VERSION


def test_load_history_tolerates_corrupted_lines(tmp_state_dir):
    # Simulate a partial write / disk corruption mid-line
    from imgen.history import HISTORY_FILE
    HISTORY_FILE.write_text(
        json.dumps({"id": 1, "input": "/a.jpg"}) + "\n"
        + "{not json at all\n"
        + json.dumps({"id": 2, "input": "/b.jpg"}) + "\n",
    )
    entries = load_history()
    assert len(entries) == 2  # corrupted line dropped, others survive
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2


def test_load_history_warns_on_corrupted_line(tmp_state_dir, capsys):
    """Silent `pass` on JSONDecodeError loses user data with no feedback.
    A warn surfaces the loss so the user knows. (python-reviewer I3)"""
    from imgen.history import HISTORY_FILE
    HISTORY_FILE.write_text(
        json.dumps({"id": 1, "input": "/a.jpg"}) + "\n"
        + "{broken\n"
    )
    load_history()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "skip" in combined.lower() or "malformed" in combined.lower()


def test_append_history_does_not_mutate_caller_dict(tmp_state_dir):
    """`entry["id"] = ...` was being written into the caller's dict
    as a hidden side-effect. (python-reviewer C1)"""
    original = {"input": "/a.jpg", "output": "/a.png"}
    snapshot = dict(original)
    append_history(original)
    # Caller's dict must be untouched
    assert original == snapshot


def test_append_history_resets_mode_on_existing_world_readable_file(tmp_state_dir):
    """If history.jsonl pre-existed at 0o644 (e.g. v0.1.0 install before
    the v0.1.1 chmod fix), os.open(O_CREAT, 0o600) ignores mode on existing
    files. Must explicitly fchmod under the lock. (security I3)"""
    import os as _os
    from imgen.history import HISTORY_FILE
    # Pre-create with permissive mode
    HISTORY_FILE.write_text("")
    HISTORY_FILE.chmod(0o644)
    assert (HISTORY_FILE.stat().st_mode & 0o777) == 0o644

    append_history({"input": "/a.jpg", "output": "/a.png"})

    mode = HISTORY_FILE.stat().st_mode & 0o777
    assert mode == 0o600, f"history.jsonl mode {oct(mode)} after append, want 0o600"


def test_replay_entry_refuses_future_schema(tmp_state_dir, capsys):
    """Entry written by a newer install must refuse replay rather than
    silently producing weird output. Pins architect #3."""
    from imgen.commands.history import replay_entry
    entry = {
        "id": 99,
        "v": HISTORY_SCHEMA_VERSION + 1,
        "input": "/some.jpg",
    }
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "newer schema" in stderr  # the gate's specific message
    assert "imgen upgrade" in stderr  # the user-facing hint


def test_replay_entry_missing_input_fails_cleanly(tmp_state_dir):
    from imgen.commands.history import replay_entry
    entry = {"id": 99, "v": HISTORY_SCHEMA_VERSION}  # no "input"
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 1


def test_replay_entry_legacy_v0_entries_pass_schema_gate(tmp_state_dir, capsys):
    """Pre-v0.2 history.jsonl entries lack the 'v' field. Default-to-0
    treatment means they pass the schema gate (don't refuse-on-future).
    Pins architect #3 'best-effort .get' contract."""
    from imgen.commands.history import replay_entry
    legacy_entry = {"id": 1, "input": "/nonexistent_xyz.jpg"}  # no 'v' key
    with pytest.raises(SystemExit):
        replay_entry(legacy_entry)
    stderr = capsys.readouterr().err
    # The future-schema gate would say "newer schema"; legacy entries
    # must NOT trigger it (they pass through to cmd_generate which then
    # dies on the missing input file — that's an unrelated failure).
    assert "newer schema" not in stderr

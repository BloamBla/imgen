"""Shared pytest fixtures.

Keep fixtures lightweight — entire test suite targets <2s. No real
subprocess to mflux, no GPU, no network.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_state_dir(tmp_path, monkeypatch):
    """Redirect imgen state (STATE_DIR, HISTORY_FILE) to a fresh tmp dir.

    Tests touching history.append/load use this to avoid clobbering the
    user's real ~/.imgen/history.jsonl and to get a clean state per test.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    history_file = state_dir / "history.jsonl"

    import imgen.history as history_mod
    import imgen.paths as paths_mod
    import imgen.styles as styles_mod

    monkeypatch.setattr(paths_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(paths_mod, "HISTORY_FILE", history_file)
    # history.py imported HISTORY_FILE at module load — rebind locally too
    monkeypatch.setattr(history_mod, "HISTORY_FILE", history_file)
    # styles.py caches the merged built-in + user-styles dict on first
    # access; reset so a test using this fixture sees the patched
    # STATE_DIR/styles.d (which is empty in the tmp dir) rather than
    # the real ~/.imgen/styles.d state from a previous test.
    monkeypatch.setattr(styles_mod, "_cached_merged", None)

    return state_dir

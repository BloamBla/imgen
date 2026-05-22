"""Shared pytest fixtures.

Keep fixtures lightweight — entire test suite targets <2s. No real
subprocess to mflux, no GPU, no network.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def tmp_state_dir(tmp_path, monkeypatch):
    """Redirect imgen state (STATE_DIR, HISTORY_FILE, LOGS_DIR) to a
    fresh tmp dir.

    Tests touching history.append/load or BatchLogger use this to avoid
    clobbering the user's real ~/.imgen/ and to get a clean state per
    test. LOGS_DIR is captured into `runs.py` at module import time,
    so a bare STATE_DIR monkeypatch doesn't propagate — patch both
    explicitly here.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    history_file = state_dir / "history.jsonl"
    logs_dir = state_dir / "logs"

    import imgen.history as history_mod
    import imgen.paths as paths_mod
    import imgen.runs as runs_mod
    import imgen.styles as styles_mod

    monkeypatch.setattr(paths_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(paths_mod, "HISTORY_FILE", history_file)
    # v0.4 architect IMP-4 made STYLES_D / BACKENDS_D module constants
    # in paths.py — captured at import time, so STATE_DIR monkeypatch
    # alone doesn't update them. Rebind explicitly here so any test
    # using this fixture sees the tmp state_dir for both subdirs.
    monkeypatch.setattr(paths_mod, "STYLES_D", state_dir / "styles.d")
    monkeypatch.setattr(paths_mod, "BACKENDS_D", state_dir / "backends.d")
    # history.py imported HISTORY_FILE at module load — rebind locally too
    monkeypatch.setattr(history_mod, "HISTORY_FILE", history_file)
    # runs.py captured LOGS_DIR at module load (= STATE_DIR / "logs"),
    # so a fresh STATE_DIR monkeypatch alone leaves LOGS_DIR pointing
    # at the real ~/.imgen/logs/. BatchLogger reads LOGS_DIR; rebind.
    monkeypatch.setattr(runs_mod, "LOGS_DIR", logs_dir)
    # styles.py caches the merged built-in + user-styles dict on first
    # access; reset so a test using this fixture sees the patched
    # STYLES_D (empty in the tmp dir) rather than the real
    # ~/.imgen/styles.d state from a previous test.
    monkeypatch.setattr(styles_mod, "_cached_merged", None)

    return state_dir

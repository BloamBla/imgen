"""v0.7.15 → v0.8.0 commit 5: doctor's deprecation surface for
config keys.

History: v0.7.15 (architect advisory) added a doctor warn when
config.toml carried the soft-deprecated `[defaults] style` key
(dead-code fallback since v0.7.13 BREAKING). v0.8.0 commit 5
hard-removed the key — config.py raises ConfigError at load time
on `[defaults] style = ...`, so the deprecation warn never runs.
The helper was generalised to `warn_deprecated_keys(cfg)` per the
v0.7.15 architect advisory, returning a list of (warn, hint) pairs
that cmd_doctor renders.

The legacy `warn_deprecated_defaults_style(defaults_section)` is
kept as a NO-OP shim — calling it with a `style` key triggers
nothing because the schema layer hard-errored on it upstream. The
shim exists only so any v0.7.x test importing the symbol keeps
working through the v0.8.x deprecation window.
"""
from __future__ import annotations

from imgen.commands.doctor import (
    warn_deprecated_defaults_style,
    warn_deprecated_keys,
)


def test_warn_deprecated_keys_returns_empty_list_for_clean_config():
    """v0.8.0 commit 5: at this point no doctor-visible deprecation
    is fired in code (the only v0.8.0 deprecation — `[defaults]
    backend` → `model` — fires at config load time, not doctor time).
    The helper exists with its v0.8 shape; commits 9+ add concrete
    deprecations into the returned list.
    """
    notices = warn_deprecated_keys({"defaults": {"model": "flux-kontext"}})
    assert notices == []


def test_warn_deprecated_keys_handles_empty_cfg():
    """Edge case: empty config dict. No deprecation notices, no
    crashes on `cfg["defaults"]` lookup."""
    assert warn_deprecated_keys({}) == []
    assert warn_deprecated_keys({"defaults": {}}) == []


def test_legacy_warn_deprecated_defaults_style_is_noop(capsys):
    """v0.8.0 commit 5: the v0.7.15 helper became a no-op shim.
    Calling it with the now-removed `style` key fires nothing —
    config.py raises ConfigError at load time before this helper
    would ever see the key. The shim is kept only to avoid breaking
    v0.7.15 test imports during the v0.8.x window."""
    warn_deprecated_defaults_style({"style": "anime", "quantize": 4})
    captured = capsys.readouterr()
    assert (captured.out + captured.err) == ""


def test_legacy_warn_deprecated_defaults_style_handles_empty(capsys):
    """Edge case mirror: shim is no-op regardless of input shape."""
    warn_deprecated_defaults_style({})
    warn_deprecated_defaults_style({"quantize": 4, "steps": 20})
    captured = capsys.readouterr()
    assert (captured.out + captured.err) == ""


# ── Gap 3 lock-in: autouse=True conftest fixture ───────────────────────


def test_autouse_fixture_redirects_history_file_to_tmp():
    """v0.7.15 (gap 3 closure): `tmp_state_dir` in conftest.py is
    `autouse=True` so EVERY test gets an isolated HISTORY_FILE by
    default — no test ever writes into the user's real
    ~/.imgen/history.jsonl. Pre-v0.7.15 tests that didn't explicitly
    request the fixture (e.g. test_gated_repo_hint_surfaces) polluted
    user state with test-generated entries.

    This test doesn't request the fixture explicitly — proving the
    autouse path. If autouse were ever removed, this assertion would
    fail because HISTORY_FILE would still point at the real
    ~/.imgen/history.jsonl."""
    from imgen import paths
    # Real user state lives at ~/.imgen/history.jsonl. Under the
    # autouse fixture, HISTORY_FILE is rebound to <tmp_path>/state/
    # history.jsonl — a per-test pytest tmp directory.
    assert "pytest" in str(paths.HISTORY_FILE).lower() or \
           "tmp" in str(paths.HISTORY_FILE).lower(), (
        f"HISTORY_FILE {paths.HISTORY_FILE} not isolated to tmp — "
        "autouse fixture may have regressed"
    )
    assert not str(paths.HISTORY_FILE).startswith(str(
        __import__("pathlib").Path.home() / ".imgen"
    )), (
        f"HISTORY_FILE {paths.HISTORY_FILE} points at the REAL "
        "~/.imgen/ — test pollution surface is open"
    )

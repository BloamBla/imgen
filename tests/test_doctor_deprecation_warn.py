"""v0.7.15: doctor warns when config.toml has the deprecated
`[defaults] style` key.

v0.7.13 (gap 8 BREAKING) made `--style` an explicit opt-in;
`merged_defaults["style"]` is no longer consulted as a fallback.
The config-schema key was kept for back-compat — but pre-v0.7.15
colleagues with `[defaults] style = "anime"` in their config.toml
had no visible signal the key was dead. doctor now surfaces the
deprecation so users can clean up before v0.8 removes the field
from the schema entirely.

Tested via `warn_deprecated_defaults_style` helper (extracted from
cmd_doctor for testability — full cmd_doctor invocation runs slow
network + system probes the deprecation logic doesn't need).
"""
from __future__ import annotations

from imgen.commands.doctor import warn_deprecated_defaults_style


def test_warns_when_defaults_style_present(capsys):
    """v0.7.15 (architect advisory): warn surfaces with the
    DEPRECATED phrase + v0.7.13 reference + v0.8 removal timeline
    + the offending value (so colleagues can grep their config to
    confirm WHICH style was set)."""
    warn_deprecated_defaults_style({"style": "anime", "quantize": 4})
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "[defaults] style" in output
    assert "DEPRECATED" in output
    assert "v0.7.13" in output  # references the breaking-change release
    assert "v0.8" in output     # tells the user when the key dies
    assert "'anime'" in output  # surfaces the offending value


def test_no_warn_when_defaults_style_absent(capsys):
    """Symmetric lock-in: defaults section without `style` triggers
    no deprecation noise. Prevents a future drift where the warn
    fires unconditionally."""
    warn_deprecated_defaults_style({"quantize": 4, "steps": 20})
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "DEPRECATED" not in output


def test_no_warn_on_empty_defaults_section(capsys):
    """Edge case: empty `defaults` dict (valid TOML with empty
    `[defaults]` table). No warn fires."""
    warn_deprecated_defaults_style({})
    captured = capsys.readouterr()
    assert "DEPRECATED" not in (captured.out + captured.err)


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

"""v0.8.0 commit 11 — end-to-end ship-state smoke test.

Per [[project-v080-design]] §Q commit 11. Single integration test
that exercises the v0.8.0 surface end-to-end so the next colleague
who breaks something accidentally hits the failure at suite-run
time, not in production:

* All 4 designed-for-v0.8 built-in Models registered with v0.8
  canonical names.
* Each Model declares a valid engine + supported_quants.
* History schema constant at v=4.
* Migrate-toml subcommand dispatchable through the CLI parser.
* Doctor's shadowing-warn helper imports clean.
* List-models / list-loras CLI surfaces don't crash.

Counts as the "+1 test" smoke per §Q commit 11. Whatever future
v0.8.x cleanup commits widen, they must keep this test green —
breaking any single assertion here means a release-blocking
regression on the locked v0.8.0 surface.
"""
from __future__ import annotations

import pytest

from imgen.defaults import HISTORY_SCHEMA_VERSION
from imgen.models import BUILTIN_MODELS


def test_v080_builtin_models_full_set_registered():
    """v0.8.0 ships exactly these 4 built-in Models with v0.8
    canonical names. New rows are additive; this lock-in catches an
    accidental delete or rename during v0.8.x cleanup commits."""
    expected = {
        "flux-kontext",
        "flux-dev",
        "flux2-klein-edit-9b",
        "qwen-image-edit-v1",
    }
    actual = set(BUILTIN_MODELS.keys())
    assert expected <= actual, (
        f"v0.8.0 ship-state requires these built-in Models: "
        f"{sorted(expected)}. Missing: {sorted(expected - actual)}"
    )


def test_v080_every_builtin_model_declares_engine_and_quants():
    """Each built-in Model carries a non-None engine field and a
    non-empty supported_quants tuple. Empty supported_quants would
    silently omit the model from the doctor RAM table (§R.3 M-3).

    v0.9 commit 7: video Models (model.video is not None) are
    carved out of the supported_quants assertion — LTX-Video is
    bf16-only at v0.9.0 (supported_quants=()) and the doctor video-
    Model RAM forecast lands in commit 9 via a separate code path
    that doesn't iterate supported_quants.
    """
    for name, model in BUILTIN_MODELS.items():
        assert model.engine in {"mflux", "diffusers_mps"}, (
            f"Model {name!r}: engine={model.engine!r} is not a known "
            "engine value"
        )
        if model.video is None:
            assert model.supported_quants, (
                f"Model {name!r}: supported_quants is empty — would be "
                "silently omitted from doctor RAM table per §R.3 M-3"
            )


def test_v080_history_schema_at_v4():
    """v=4 is the v0.8.0 schema version (commit 9). v0.9.0 may bump
    again; until then this is locked."""
    assert HISTORY_SCHEMA_VERSION == 4


def test_v080_migrate_toml_subcommand_dispatchable():
    """The ``imgen migrate-toml`` subcommand is wired into the parser
    (parser.py stanza) AND the CLI dispatch table (cli._HANDLERS). A
    regression that drops one but not the other would only surface
    on an actual ``imgen migrate-toml`` invocation, which no other
    test triggers."""
    from imgen.cli import _HANDLERS, _KNOWN_SUBCOMMANDS

    assert "migrate-toml" in _KNOWN_SUBCOMMANDS
    assert "migrate-toml" in _HANDLERS
    from imgen.commands.migrate_toml import cmd_migrate_toml
    assert _HANDLERS["migrate-toml"] is cmd_migrate_toml


def test_v080_doctor_shadowing_warn_helper_imports():
    """Doctor's ``_warn_shadowing_user_tomls`` is the §G.3 architect
    IMPORTANT surface. Its presence is locked-in here so a future
    refactor doesn't silently remove it. (Behaviour is covered by
    tests/test_v080_cli_migration.py.)"""
    from imgen.commands.doctor import _warn_shadowing_user_tomls
    assert callable(_warn_shadowing_user_tomls)


def test_v080_list_models_cli_runs_clean(capsys):
    """``imgen --list-models`` is the canonical model lister at v0.8
    (replacing ``--list-backends``). Smoke-runs the printer."""
    from imgen.parser import print_models
    rc = print_models()
    assert rc == 0
    out = capsys.readouterr().out
    # Must surface every built-in by its v0.8 canonical name.
    assert "flux-kontext" in out
    assert "flux-dev" in out
    assert "flux2-klein-edit-9b" in out
    assert "qwen-image-edit-v1" in out


def test_v080_v07_aliases_resolve_through_rename_map():
    """v0.7 names ``flux`` / ``qwen`` translate to v0.8 canonical via
    the rename map (commit 2 + 4a). Locks the back-compat surface so
    a v0.7 user's history-replay / scripted-CLI gets the right model
    after upgrade."""
    from imgen.models import _V07_TO_V08_MODEL_RENAMES
    assert _V07_TO_V08_MODEL_RENAMES["flux"] == "flux-kontext"
    assert _V07_TO_V08_MODEL_RENAMES["qwen"] == "qwen-image-edit-v1"

"""v0.8.0 commit 5 — config.toml schema migration lock-ins.

Per [[project-v080-design]] §J + §Q commit 5:

* ``[defaults] backend = ...`` → ``[defaults] model = ...``
  warn-and-bridge through v0.8.x; v0.9.0 drops the legacy key.
* ``[defaults] style = ...`` REMOVED with hard-error + STATIC
  migration hint (no value echo — security MEDIUM round-2 lock-in).
* doctor's ``warn_deprecated_keys(cfg)`` returns the v0.8.0
  deprecation list (currently empty — `style` hard-errors at load
  time so it can't reach doctor; `backend` warn fires at load time
  too; both are stricter than doctor-time surfacing).
"""
from __future__ import annotations

import pytest

from imgen.commands.doctor import warn_deprecated_keys
from imgen.config import (
    ConfigError,
    _apply_v08_defaults_aliases,
    _reject_removed_defaults_keys,
    load_validated_config,
)


# ── §J lock-in 1: [defaults] backend warn-and-bridge ───────────────────


def test_config_defaults_backend_warns_and_maps_to_model(
    tmp_state_dir, tmp_path, capsys,
):
    """``[defaults] backend = "flux"`` emits DEPRECATED warn at load
    time AND auto-maps to ``model = "flux-kontext"`` through the
    v0.7→v0.8 rename map. Result is a validated dict containing
    ``model`` (NOT ``backend``)."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[defaults]\nbackend = "flux"\n')
    loaded = load_validated_config(cfg)
    assert loaded["defaults"] == {"model": "flux-kontext"}

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "DEPRECATED" in combined
    assert "[defaults] backend" in combined
    assert "[defaults] model" in combined


def test_config_defaults_backend_unchanged_name_passes_through(
    tmp_state_dir, tmp_path, capsys,
):
    """Unchanged names (``flux-dev``, user TOML stems) pass through
    the rename map by identity — but still warn that the LEGACY KEY
    is deprecated."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[defaults]\nbackend = "flux-dev"\n')
    loaded = load_validated_config(cfg)
    assert loaded["defaults"] == {"model": "flux-dev"}

    captured = capsys.readouterr()
    assert "DEPRECATED" in (captured.out + captured.err)


def test_config_defaults_backend_and_model_both_set_model_wins(
    tmp_state_dir, tmp_path, capsys,
):
    """Per §J: when both ``backend`` and ``model`` are set, ``model``
    wins (the legacy key is silently DROPPED, the DEPRECATED warn
    still fires)."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[defaults]\nbackend = "flux"\nmodel = "qwen-image-edit-v1"\n'
    )
    loaded = load_validated_config(cfg)
    assert loaded["defaults"] == {"model": "qwen-image-edit-v1"}

    captured = capsys.readouterr()
    assert "DEPRECATED" in (captured.out + captured.err)


# ── §J lock-in 2: [defaults] style HARD-ERROR ──────────────────────────


def test_config_defaults_style_hard_error(tmp_state_dir, tmp_path):
    """``[defaults] style = ...`` was REMOVED in v0.8.0 (soft-
    deprecated since v0.7.13, doctor-warned since v0.7.15). The
    schema raises ConfigError at load time with a static migration
    hint — no silent drop, no warn-and-continue."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('[defaults]\nstyle = "pixar"\n')
    with pytest.raises(ConfigError) as exc_info:
        load_validated_config(cfg)
    msg = str(exc_info.value)
    assert "[defaults] style was removed in v0.8.0" in msg
    assert "--style NAME" in msg  # migration hint surfaces the new path


# ── §J lock-in 3: removed-key error does NOT echo the rejected value ──


def test_config_defaults_style_error_does_not_echo_value(
    tmp_state_dir, tmp_path,
):
    """Security MEDIUM round-2 lock-in: a ConfigError raised for
    ``[defaults] style = "<EVIL>"`` does NOT contain the EVIL value
    anywhere — neither in ``.args`` nor in the rendered string. The
    error is static migration text only.

    Threat model: a colleague-shared config.toml with a malicious
    style value could leak terminal-escape sequences through the
    config-load error path (e.g. into a cron job's stderr log).

    TOML itself rejects raw control bytes in unescaped strings (the
    parser flags ``Illegal character`` long before our schema runs).
    Exercise the no-echo contract with TOML-valid-but-suspicious
    values: shell-command-substitution syntax + a unicode-escaped
    control byte (``\\u001b``) inside the string literal.
    """
    evil_value = "$(touch /tmp/x)\\u001b[2Jsentinel-evil-marker"
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[defaults]\nstyle = "{evil_value}"\n')

    with pytest.raises(ConfigError) as exc_info:
        load_validated_config(cfg)

    # Render in every shape the caller might surface it:
    rendered = str(exc_info.value)
    args_str = " ".join(str(a) for a in exc_info.value.args)

    # The TOML-parsed VALUE never appears in the error. TOML decodes
    # `` to the actual escape byte; assert neither form leaks.
    parsed_form = "$(touch /tmp/x)\x1b[2Jsentinel-evil-marker"
    assert parsed_form not in rendered
    assert parsed_form not in args_str
    # Spot-checks: the escape byte specifically is absent
    assert "\x1b" not in rendered
    # The shell substitution syntax is absent (would be leaked if
    # the value were repr'd into the message)
    assert "$(touch" not in rendered
    # And the sentinel marker we'd have grep'd to find a leak
    assert "sentinel-evil-marker" not in rendered


def test_reject_removed_defaults_keys_static_message():
    """Direct unit lock-in on ``_reject_removed_defaults_keys``: any
    value supplied via the ``style`` key produces the same static
    migration text — no per-value formatting."""
    msg_a = None
    msg_b = None
    try:
        _reject_removed_defaults_keys({"style": "anime"}, "[defaults]")
    except ConfigError as e:
        msg_a = str(e)
    try:
        _reject_removed_defaults_keys({"style": "pixar"}, "[defaults]")
    except ConfigError as e:
        msg_b = str(e)
    assert msg_a is not None and msg_b is not None
    # Identical error texts → static template; differing texts would
    # mean a per-value formatter snuck into the helper.
    assert msg_a == msg_b


# ── §J lock-in 4: doctor surfaces v0.8 deprecation list ────────────────


def test_doctor_warn_deprecated_keys_empty_for_v08_clean_config():
    """At v0.8.0 commit 5 the doctor-time deprecation list is empty
    by design — the two deprecations active at this commit
    (``style``: hard-error; ``backend``: warn-and-bridge) both fire
    at config LOAD time, not doctor time. The helper exists in its
    v0.8 shape so commits 9+ deprecations land into a stable
    contract.
    """
    cfg = {
        "defaults": {"model": "flux-kontext", "steps": 20},
        "ui": {},
        "enhance": {},
    }
    assert warn_deprecated_keys(cfg) == []


def test_apply_v08_defaults_aliases_returns_new_dict():
    """Migration helper is pure: returns a NEW dict, does not mutate
    the input. Locks the discipline against accidental in-place
    edits during future schema-migration evolutions."""
    original = {"backend": "flux"}
    out = _apply_v08_defaults_aliases(original, "[defaults]")
    assert original == {"backend": "flux"}  # unmutated
    assert out == {"model": "flux-kontext"}  # migrated copy


def test_apply_v08_defaults_aliases_no_legacy_key_is_noop():
    """No ``backend`` key → return the input unchanged. No warn
    fired (the warn only fires when the legacy key is actually
    present)."""
    cfg = {"model": "flux-kontext", "steps": 20}
    out = _apply_v08_defaults_aliases(cfg, "[defaults]")
    assert out == cfg

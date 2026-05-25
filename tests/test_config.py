"""config.toml parser + validator + precedence merge.

Pure slice — load, validate, merge. cmd_generate integration is smoke-tested
in CI via --dry-run, not here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from imgen.config import (
    ConfigError,
    effective_defaults,
    effective_output_dir,
    load_config,
    load_validated_config,
    validate_section,
    DEFAULTS_SCHEMA,
    UI_SCHEMA,
)


# ── load_config — raw TOML read ──────────────────────────────────────────

def test_load_config_missing_file_returns_empty(tmp_path):
    assert load_config(tmp_path / "nonexistent.toml") == {}


def test_load_config_parses_known_sections(tmp_path):
    """v0.8.0 commit 5: ``style`` removed; pivot to ``model`` for a
    non-empty [defaults] fixture. ``load_config`` is the RAW TOML
    reader (no v0.8 migration / schema run yet) — at this layer the
    value is read as-is."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[defaults]\nmodel = \"flux-kontext\"\n"
        "[ui]\nopen_in_preview = false\n"
    )
    raw = load_config(cfg)
    assert raw["defaults"]["model"] == "flux-kontext"
    assert raw["ui"]["open_in_preview"] is False


def test_load_config_malformed_toml_returns_empty_with_warning(tmp_path, capsys):
    cfg = tmp_path / "broken.toml"
    cfg.write_text("[defaults\nstyle = anime")  # missing closing bracket + quotes
    result = load_config(cfg)
    assert result == {}
    captured = capsys.readouterr()
    assert "broken.toml" in (captured.out + captured.err)


def test_load_config_oversized_file_returns_empty_with_warning(tmp_path, capsys):
    """A 500 MB config file shouldn't get fully loaded into RAM by tomllib.
    Cap at 1 MB — anything larger gets warn + empty. (security I2)"""
    from imgen.config import CONFIG_MAX_BYTES
    cfg = tmp_path / "huge.toml"
    cfg.write_bytes(b"x = 1\n" * (CONFIG_MAX_BYTES // 6 + 100))
    result = load_config(cfg)
    assert result == {}
    captured = capsys.readouterr()
    assert "too large" in (captured.out + captured.err).lower()


# ── validate_section — type + range gate for known keys ─────────────────

def test_validate_section_accepts_all_known_keys():
    """v0.8.0 commit 5: ``style`` removed from schema (hard-errored
    upstream by ``_reject_removed_defaults_keys``); ``backend``
    removed from schema (migrated to ``model`` upstream by
    ``_apply_v08_defaults_aliases``); ``model`` is the canonical key."""
    raw = {
        "model": "qwen-image-edit-v1",
        "quantize": 4,
        "steps": 30,
        "guidance": 4.5,
        "strength": 0.6,
        "output_dir": "~/Pictures/imgen",
        "mlx_cache_gb": 24,
        "battery_stop": 15,
    }
    out = validate_section("defaults", raw, DEFAULTS_SCHEMA)
    assert out == raw


@pytest.mark.parametrize("bad", [0, -1, 1024])
def test_validate_section_rejects_mlx_cache_gb_out_of_range(bad):
    """architect C2: mlx_cache_gb missing from schema would warn-as-unknown
    instead of validating. Lock the range."""
    with pytest.raises(ConfigError):
        validate_section("defaults", {"mlx_cache_gb": bad}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad", [-1, 101, 999])
def test_validate_section_rejects_battery_stop_out_of_range(bad):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"battery_stop": bad}, DEFAULTS_SCHEMA)


def test_validate_section_drops_unknown_keys_with_warning(capsys):
    """v0.8.0 commit 5: ``style`` is unknown to DEFAULTS_SCHEMA now
    (removed at 4b-5). At the LOW-LEVEL ``validate_section`` API
    (which doesn't run pre-validate hooks), unknown keys still
    warn-and-drop — but the user-facing load path raises ConfigError
    on style via ``_reject_removed_defaults_keys`` upstream.
    """
    raw = {"model": "flux-kontext", "made_up_key": "whatever"}
    out = validate_section("defaults", raw, DEFAULTS_SCHEMA)
    assert "made_up_key" not in out
    assert out["model"] == "flux-kontext"
    captured = capsys.readouterr()
    assert "made_up_key" in (captured.out + captured.err)


def test_validate_section_rejects_unknown_model():
    """v0.8.0 commit 5: schema key is ``model`` (was ``backend`` pre-4b).
    Unknown model name → ConfigError at schema-validation time."""
    with pytest.raises(ConfigError):
        validate_section(
            "defaults", {"model": "stable-diffusion"}, DEFAULTS_SCHEMA,
        )


@pytest.mark.parametrize("bad_q", [2, 7, 9, 16])
def test_validate_section_rejects_quantize_outside_allowed_set(bad_q):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"quantize": bad_q}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad_steps", [0, -1, 201, 10_000])
def test_validate_section_rejects_steps_out_of_range(bad_steps):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"steps": bad_steps}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad_g", [-0.1, 15.1, -1.0, 100.0])
def test_validate_section_rejects_guidance_out_of_range(bad_g):
    """v0.7.11 (gap 2): lower bound is now 0.0 (was 0.5 pre-v0.7.11).
    Distilled models like Z-Image-Turbo / FLUX-schnell train with CFG
    disabled and require ``guidance=0.0``."""
    with pytest.raises(ConfigError):
        validate_section("defaults", {"guidance": bad_g}, DEFAULTS_SCHEMA)


def test_validate_section_accepts_guidance_zero():
    """v0.7.11 (gap 2): guidance=0.0 must validate. The 0.5 floor
    pre-v0.7.11 silently blocked distilled-model configs."""
    out = validate_section("defaults", {"guidance": 0.0}, DEFAULTS_SCHEMA)
    assert out["guidance"] == 0.0


@pytest.mark.parametrize("bad_s", [-0.1, 1.1, 2.0])
def test_validate_section_rejects_strength_out_of_range(bad_s):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"strength": bad_s}, DEFAULTS_SCHEMA)


def test_validate_section_rejects_bool_as_numeric():
    """TOML bool would silently pass `isinstance(v, int)` since bool subclasses
    int. Explicit guard required: `guidance = true` shouldn't be accepted."""
    with pytest.raises(ConfigError):
        validate_section("defaults", {"guidance": True}, DEFAULTS_SCHEMA)
    with pytest.raises(ConfigError):
        validate_section("defaults", {"steps": False}, DEFAULTS_SCHEMA)


def test_validate_section_accepts_int_as_float():
    """TOML `guidance = 4` (int literal) should be accepted where float is
    expected — Python int/float coercion is fine."""
    out = validate_section("defaults", {"guidance": 4}, DEFAULTS_SCHEMA)
    assert out["guidance"] == 4


# ── UI section ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_validate_ui_color_accepts_known_modes(mode):
    out = validate_section("ui", {"color": mode}, UI_SCHEMA)
    assert out["color"] == mode


def test_validate_ui_rejects_bad_color_mode():
    with pytest.raises(ConfigError):
        validate_section("ui", {"color": "rainbow"}, UI_SCHEMA)


def test_validate_ui_open_in_preview_must_be_bool():
    out = validate_section("ui", {"open_in_preview": False}, UI_SCHEMA)
    assert out["open_in_preview"] is False
    with pytest.raises(ConfigError):
        validate_section("ui", {"open_in_preview": "yes"}, UI_SCHEMA)


# ── load_validated_config — composite ────────────────────────────────────

def test_load_validated_config_empty_file(tmp_path):
    result = load_validated_config(tmp_path / "missing.toml")
    # v0.5 added the `enhance` section. Missing file = all sections empty.
    assert result == {"defaults": {}, "ui": {}, "enhance": {}}


def test_load_validated_config_round_trip(tmp_path):
    """v0.8.0 commit 5: ``[defaults] style`` REMOVED — use ``model``
    for the round-trip lock-in. ``style`` would now hard-error via
    ``_reject_removed_defaults_keys``; that path is tested separately
    in ``tests/test_v080_config_migration.py``."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[defaults]\nmodel = \"flux-kontext\"\nsteps = 12\n"
        "[ui]\nopen_in_preview = false\n"
    )
    result = load_validated_config(cfg)
    assert result["defaults"] == {"model": "flux-kontext", "steps": 12}
    assert result["ui"] == {"open_in_preview": False}


def test_load_validated_config_raises_on_bad_value(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[defaults]\nsteps = 999\n")
    with pytest.raises(ConfigError):
        load_validated_config(cfg)


# ── effective_defaults — merge config over module DEFAULTS ──────────────

def test_effective_defaults_empty_config_returns_module_defaults():
    module = {"style": "pixar", "steps": 20}
    assert effective_defaults({}, module) == module


def test_effective_defaults_config_overrides_module():
    module = {"style": "pixar", "steps": 20, "guidance": 3.5}
    config = {"style": "anime", "steps": 12}
    merged = effective_defaults(config, module)
    assert merged == {"style": "anime", "steps": 12, "guidance": 3.5}


def test_effective_defaults_does_not_mutate_inputs():
    module = {"style": "pixar"}
    config = {"style": "anime"}
    effective_defaults(config, module)
    assert module == {"style": "pixar"}
    assert config == {"style": "anime"}


# ── effective_output_dir — env > config > module-default ────────────────

def test_effective_output_dir_env_wins(monkeypatch):
    monkeypatch.setenv("IMGEN_OUTPUT_DIR", "/from-env")
    result = effective_output_dir(
        config_value="/from-config", module_default=Path("/from-default")
    )
    assert result == Path("/from-env")


def test_effective_output_dir_config_when_no_env(monkeypatch):
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    result = effective_output_dir(
        config_value="~/Pictures/imgen", module_default=Path("/from-default")
    )
    # `~` should expand
    assert result == Path("~/Pictures/imgen").expanduser()


def test_effective_output_dir_module_default_when_neither(monkeypatch):
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    default = Path("/some/default")
    assert effective_output_dir(config_value=None, module_default=default) == default


def test_effective_output_dir_empty_config_treated_as_none(monkeypatch):
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    default = Path("/some/default")
    assert effective_output_dir(config_value="", module_default=default) == default


def test_effective_output_dir_env_set_after_import_is_picked_up(monkeypatch):
    """paths.DEFAULT_OUTPUT_DIR baked env at module import — but
    effective_output_dir reads env at CALL time, so an env change after
    import (e.g. in tests via monkeypatch.setenv) must be visible.
    (python-reviewer I5)"""
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    default = Path("/module/default")
    # Before setenv: returns module_default
    assert effective_output_dir(config_value=None, module_default=default) == default
    # After setenv: env wins
    monkeypatch.setenv("IMGEN_OUTPUT_DIR", "/from-env-late")
    assert effective_output_dir(config_value=None, module_default=default) == Path("/from-env-late")


# ── effective_output_dir cli_value (--output-dir flag, v0.2.3) ──────────

def test_effective_output_dir_cli_value_beats_env(monkeypatch):
    """`--output-dir` is the highest-priority channel — beats env even if
    env was the "one-off override" in v0.1.x. CLI > env > config > default.
    (architect I1 from v0.2.2 audit)"""
    monkeypatch.setenv("IMGEN_OUTPUT_DIR", "/from-env")
    result = effective_output_dir(
        cli_value="/from-cli",
        config_value="/from-config",
        module_default=Path("/from-default"),
    )
    assert result == Path("/from-cli")


def test_effective_output_dir_cli_value_beats_config(monkeypatch):
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    result = effective_output_dir(
        cli_value="/from-cli",
        config_value="/from-config",
        module_default=Path("/from-default"),
    )
    assert result == Path("/from-cli")


def test_effective_output_dir_cli_value_expands_tilde(monkeypatch):
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    result = effective_output_dir(
        cli_value="~/runs",
        config_value=None,
        module_default=Path("/from-default"),
    )
    assert result == Path("~/runs").expanduser()


def test_effective_output_dir_cli_value_none_falls_through_to_env(monkeypatch):
    """`--output-dir` not passed → behaviour matches pre-v0.2.3 (env > config > default)."""
    monkeypatch.setenv("IMGEN_OUTPUT_DIR", "/from-env")
    result = effective_output_dir(
        cli_value=None,
        config_value="/from-config",
        module_default=Path("/from-default"),
    )
    assert result == Path("/from-env")


def test_effective_output_dir_cli_value_empty_string_treated_as_unset(monkeypatch):
    """`--output-dir ''` (someone scripting badly) → treat as no override,
    not as "write to cwd". Same forgiveness as empty config_value."""
    monkeypatch.delenv("IMGEN_OUTPUT_DIR", raising=False)
    default = Path("/from-default")
    assert effective_output_dir(
        cli_value="",
        config_value=None,
        module_default=default,
    ) == default

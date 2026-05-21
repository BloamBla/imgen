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
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[defaults]\nstyle = \"anime\"\n"
        "[ui]\nopen_in_preview = false\n"
    )
    raw = load_config(cfg)
    assert raw["defaults"]["style"] == "anime"
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
    raw = {
        "style": "anime",
        "backend": "qwen",
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
    raw = {"style": "anime", "made_up_key": "whatever"}
    out = validate_section("defaults", raw, DEFAULTS_SCHEMA)
    assert "made_up_key" not in out
    assert out["style"] == "anime"
    captured = capsys.readouterr()
    assert "made_up_key" in (captured.out + captured.err)


def test_validate_section_rejects_unknown_style():
    with pytest.raises(ConfigError) as exc_info:
        validate_section("defaults", {"style": "stalin_neopop"}, DEFAULTS_SCHEMA)
    assert "style" in str(exc_info.value)


def test_validate_section_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        validate_section("defaults", {"backend": "stable-diffusion"}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad_q", [2, 7, 9, 16])
def test_validate_section_rejects_quantize_outside_allowed_set(bad_q):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"quantize": bad_q}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad_steps", [0, -1, 201, 10_000])
def test_validate_section_rejects_steps_out_of_range(bad_steps):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"steps": bad_steps}, DEFAULTS_SCHEMA)


@pytest.mark.parametrize("bad_g", [0.4, 15.1, -1.0, 100.0])
def test_validate_section_rejects_guidance_out_of_range(bad_g):
    with pytest.raises(ConfigError):
        validate_section("defaults", {"guidance": bad_g}, DEFAULTS_SCHEMA)


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
    assert result == {"defaults": {}, "ui": {}}


def test_load_validated_config_round_trip(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[defaults]\nstyle = \"anime\"\nsteps = 12\n"
        "[ui]\nopen_in_preview = false\n"
    )
    result = load_validated_config(cfg)
    assert result["defaults"] == {"style": "anime", "steps": 12}
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

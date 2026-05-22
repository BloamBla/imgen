"""CLI flags + config schema for the v0.5 LLM prompt enhancer.

Covers:

* ``[enhance]`` section parsing in config.toml — every key validated,
  bad values raise ConfigError, missing section yields empty dict.
* ``effective_enhance()`` precedence: CLI > config > module defaults.
* ``--enhance-prompt`` / ``--no-enhance`` mutex + dest=enhance shape.
* ``--enhance-model`` / ``--enhance-temperature`` as string/float
  overrides with proper range validation on temperature.

No subprocess / no mlx_lm — these are pure config + parser tests.
"""
from __future__ import annotations

import pytest

from imgen.config import (
    ENHANCE_SCHEMA,
    ConfigError,
    effective_enhance,
    load_validated_config,
    validate_section,
)
from imgen.parser import build_parser


# ── ENHANCE_SCHEMA validators ───────────────────────────────────────────


class TestEnhanceSchema:
    def test_all_fields_present(self):
        # Lock-in: future commits must not silently drop a field.
        assert set(ENHANCE_SCHEMA.keys()) == {
            "default", "model", "temperature", "max_tokens", "timeout_s",
        }

    def test_default_must_be_bool(self):
        with pytest.raises(ConfigError):
            validate_section("enhance", {"default": "yes"}, ENHANCE_SCHEMA)
        # Bools pass cleanly.
        assert validate_section(
            "enhance", {"default": True}, ENHANCE_SCHEMA
        ) == {"default": True}

    def test_model_must_be_non_empty_string(self):
        with pytest.raises(ConfigError):
            validate_section("enhance", {"model": ""}, ENHANCE_SCHEMA)
        with pytest.raises(ConfigError):
            validate_section("enhance", {"model": "   "}, ENHANCE_SCHEMA)
        with pytest.raises(ConfigError):
            validate_section("enhance", {"model": 7}, ENHANCE_SCHEMA)
        assert validate_section(
            "enhance", {"model": "Qwen/Qwen2.5-7B-Instruct"}, ENHANCE_SCHEMA,
        ) == {"model": "Qwen/Qwen2.5-7B-Instruct"}

    def test_temperature_range(self):
        with pytest.raises(ConfigError):
            validate_section(
                "enhance", {"temperature": -0.1}, ENHANCE_SCHEMA
            )
        with pytest.raises(ConfigError):
            validate_section(
                "enhance", {"temperature": 2.5}, ENHANCE_SCHEMA
            )
        # Bool must NOT silently pass as int=0/1.
        with pytest.raises(ConfigError):
            validate_section("enhance", {"temperature": True}, ENHANCE_SCHEMA)
        # Boundary values pass.
        for v in (0.0, 0.5, 1.0, 2.0):
            assert validate_section(
                "enhance", {"temperature": v}, ENHANCE_SCHEMA
            ) == {"temperature": v}

    def test_max_tokens_range(self):
        with pytest.raises(ConfigError):
            validate_section(
                "enhance", {"max_tokens": 0}, ENHANCE_SCHEMA
            )
        with pytest.raises(ConfigError):
            validate_section(
                "enhance", {"max_tokens": 4097}, ENHANCE_SCHEMA
            )
        with pytest.raises(ConfigError):
            validate_section("enhance", {"max_tokens": True}, ENHANCE_SCHEMA)
        assert validate_section(
            "enhance", {"max_tokens": 200}, ENHANCE_SCHEMA
        ) == {"max_tokens": 200}

    def test_timeout_range(self):
        with pytest.raises(ConfigError):
            validate_section("enhance", {"timeout_s": 0}, ENHANCE_SCHEMA)
        with pytest.raises(ConfigError):
            validate_section("enhance", {"timeout_s": 3601}, ENHANCE_SCHEMA)
        assert validate_section(
            "enhance", {"timeout_s": 60}, ENHANCE_SCHEMA
        ) == {"timeout_s": 60}

    def test_unknown_field_dropped_with_warn(self, capsys):
        out = validate_section(
            "enhance", {"unknown_key": 42}, ENHANCE_SCHEMA
        )
        assert out == {}
        # colors.warn() writes to stdout, not stderr — matches the
        # project's `warn` convention (only `err` goes to stderr).
        captured = capsys.readouterr()
        assert "unknown_key" in captured.out


# ── load_validated_config carries the enhance section ──────────────────


def test_load_validated_config_includes_enhance_section(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[enhance]\n"
        "default = true\n"
        "model = \"Qwen/Qwen2.5-7B-Instruct\"\n"
        "temperature = 0.0\n"
        "max_tokens = 150\n"
        "timeout_s = 90\n"
    )
    loaded = load_validated_config(cfg)
    assert "enhance" in loaded
    assert loaded["enhance"]["default"] is True
    assert loaded["enhance"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert loaded["enhance"]["temperature"] == 0.0
    assert loaded["enhance"]["max_tokens"] == 150
    assert loaded["enhance"]["timeout_s"] == 90


def test_missing_enhance_section_yields_empty_dict(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[defaults]\nstyle = \"anime\"\n")
    loaded = load_validated_config(cfg)
    assert loaded["enhance"] == {}


# ── effective_enhance precedence ───────────────────────────────────────


class TestEffectiveEnhance:
    def test_module_defaults_when_config_empty_cli_none(self):
        out = effective_enhance(cli_enable=None, config_enhance={})
        assert out["enabled"] is False  # opt-in
        assert out["model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert out["temperature"] == 0.0
        assert out["max_tokens"] == 200
        assert out["timeout_s"] == 120

    def test_config_overrides_module(self):
        out = effective_enhance(
            cli_enable=None,
            config_enhance={
                "default": True,
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "temperature": 0.2,
            },
        )
        assert out["enabled"] is True       # config opted in
        assert out["model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert out["temperature"] == 0.2
        # Untouched fields still inherit module defaults.
        assert out["max_tokens"] == 200
        assert out["timeout_s"] == 120

    def test_cli_enable_true_beats_config_default_false(self):
        out = effective_enhance(
            cli_enable=True,
            config_enhance={"default": False},
        )
        assert out["enabled"] is True

    def test_cli_no_enhance_beats_config_default_true(self):
        out = effective_enhance(
            cli_enable=False,
            config_enhance={"default": True},
        )
        assert out["enabled"] is False

    def test_cli_model_overrides_config_model(self):
        out = effective_enhance(
            cli_enable=None,
            config_enhance={"model": "Qwen/Qwen2.5-3B-Instruct"},
            cli_model="my-custom-model",
        )
        assert out["model"] == "my-custom-model"

    def test_cli_temperature_overrides_config_temperature(self):
        out = effective_enhance(
            cli_enable=None,
            config_enhance={"temperature": 0.5},
            cli_temperature=0.0,
        )
        assert out["temperature"] == 0.0

    def test_does_not_mutate_config_input(self):
        config_in = {"default": True, "model": "X"}
        snapshot = dict(config_in)
        effective_enhance(
            cli_enable=False, config_enhance=config_in,
            cli_model="other", cli_temperature=1.0,
        )
        assert config_in == snapshot


# ── CLI flag parsing on `generate` ─────────────────────────────────────


_DUMMY_DEFAULTS = {
    "style": "pixar", "backend": "flux", "quantize": 8, "steps": 20,
    "guidance": 3.5, "strength": 0.55, "mlx_cache_gb": 12, "battery_stop": 20,
}


def _parse(*argv: str):
    p = build_parser(_DUMMY_DEFAULTS)
    return p.parse_args(argv)


class TestParserEnhanceFlags:
    def test_no_flag_means_enhance_is_none(self):
        args = _parse("generate", "photo.jpg", "-s", "anime")
        assert args.enhance is None
        assert args.enhance_model is None
        assert args.enhance_temperature is None

    def test_enhance_prompt_sets_true(self):
        args = _parse("generate", "photo.jpg", "-s", "anime", "--enhance-prompt")
        assert args.enhance is True

    def test_no_enhance_sets_false(self):
        args = _parse("generate", "photo.jpg", "-s", "anime", "--no-enhance")
        assert args.enhance is False

    def test_enhance_and_no_enhance_are_mutex(self, capsys):
        with pytest.raises(SystemExit):
            _parse("generate", "photo.jpg", "-s", "anime",
                   "--enhance-prompt", "--no-enhance")

    def test_enhance_model_captured(self):
        args = _parse(
            "generate", "photo.jpg", "-s", "anime",
            "--enhance-model", "Qwen/Qwen2.5-3B-Instruct",
        )
        assert args.enhance_model == "Qwen/Qwen2.5-3B-Instruct"

    def test_enhance_temperature_captured(self):
        args = _parse(
            "generate", "photo.jpg", "-s", "anime",
            "--enhance-temperature", "0.5",
        )
        assert args.enhance_temperature == 0.5

    def test_enhance_temperature_range_validated(self):
        with pytest.raises(SystemExit):
            _parse("generate", "photo.jpg", "-s", "anime",
                   "--enhance-temperature", "-0.1")
        with pytest.raises(SystemExit):
            _parse("generate", "photo.jpg", "-s", "anime",
                   "--enhance-temperature", "2.1")


# ── CLI flag parsing on `batch` — same flags must work ─────────────────


class TestParserEnhanceFlagsOnBatch:
    def test_batch_accepts_enhance_prompt(self):
        args = _parse("batch", "/some/dir", "-s", "anime",
                      "--enhance-prompt")
        assert args.enhance is True

    def test_batch_accepts_no_enhance(self):
        args = _parse("batch", "/some/dir", "-s", "anime",
                      "--no-enhance")
        assert args.enhance is False

    def test_batch_accepts_enhance_model(self):
        args = _parse("batch", "/some/dir", "-s", "anime",
                      "--enhance-model", "x/y")
        assert args.enhance_model == "x/y"

    def test_batch_mutex_enforced(self):
        with pytest.raises(SystemExit):
            _parse("batch", "/some/dir", "-s", "anime",
                   "--enhance-prompt", "--no-enhance")

"""Optional user config at ~/.imgen/config.toml.

Two sections:

  [defaults]   overrides the module DEFAULTS (style, backend, quantize,
               steps, guidance, strength, output_dir). Precedence at use
               time: CLI flag > config > module DEFAULTS.

  [ui]         open_in_preview (bool), color (auto/always/never).

Validation rules:
  - Known keys: type + range checked. Bad value → ConfigError.
  - Unknown keys: warned, dropped — forward-compat with future imgen.
  - Missing file or malformed TOML: warn, treat as empty.

For output_dir specifically the precedence is env > config > module
default, because $IMGEN_OUTPUT_DIR is the "one-off override" channel
and matches the v0.1.x behavior already documented in the README.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Callable

from ._schema import validate_against_schema
from .backends import BACKENDS
from .colors import warn
from .styles import list_styles


__all__ = [
    "CONFIG_MAX_BYTES",
    "ConfigError",
    "DEFAULTS_SCHEMA",
    "ENHANCE_SCHEMA",
    "UI_SCHEMA",
    "effective_defaults",
    "effective_enhance",
    "effective_output_dir",
    "load_config",
    "load_validated_config",
    "validate_section",
]

# Cap config.toml size so a rogue/oversized file can't OOM tomllib.
# Real configs are well under 1 KB; the cap is several orders of magnitude
# above realistic use.
CONFIG_MAX_BYTES = 1 * 1024 * 1024  # 1 MB


class ConfigError(Exception):
    """Bad value in a known key of ~/.imgen/config.toml."""


# ── Schema ───────────────────────────────────────────────────────────────

# Each entry: key → (human-readable expected-type description, predicate).
# Predicate returns True if the value is acceptable. bool is explicitly
# rejected for numeric fields since bool subclasses int in Python (so
# `guidance = true` would otherwise silently pass isinstance(v, int)).

def _is_int_not_bool(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number_not_bool(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_SchemaEntry = tuple[str, Callable[[Any], bool]]

DEFAULTS_SCHEMA: dict[str, _SchemaEntry] = {
    "style": (
        "string from imgen.styles.list_styles()",
        lambda v: isinstance(v, str) and v in list_styles(),
    ),
    "backend": (
        f"one of {sorted(BACKENDS.keys())!r}",
        lambda v: isinstance(v, str) and v in BACKENDS,
    ),
    "quantize": (
        "int in {3, 4, 5, 6, 8}",
        lambda v: _is_int_not_bool(v) and v in (3, 4, 5, 6, 8),
    ),
    "steps": (
        "int 1..200",
        lambda v: _is_int_not_bool(v) and 1 <= v <= 200,
    ),
    "guidance": (
        # v0.7.11 (gap 2): lower bound dropped 0.5 → 0.0 so distilled
        # models (Z-Image-Turbo, FLUX-schnell) can run with CFG fully
        # disabled, which is the regime they were trained for.
        "number 0.0..15.0",
        lambda v: _is_number_not_bool(v) and 0.0 <= v <= 15.0,
    ),
    "strength": (
        "number 0.0..1.0",
        lambda v: _is_number_not_bool(v) and 0.0 <= v <= 1.0,
    ),
    "output_dir": (
        "path string",
        lambda v: isinstance(v, str),
    ),
    "mlx_cache_gb": (
        "int 1..256",
        lambda v: _is_int_not_bool(v) and 1 <= v <= 256,
    ),
    "battery_stop": (
        "int 0..100",
        lambda v: _is_int_not_bool(v) and 0 <= v <= 100,
    ),
}

UI_SCHEMA: dict[str, _SchemaEntry] = {
    "open_in_preview": (
        "bool (true / false)",
        lambda v: isinstance(v, bool),
    ),
    "color": (
        "one of 'auto' / 'always' / 'never'",
        lambda v: isinstance(v, str) and v in ("auto", "always", "never"),
    ),
}

# v0.5: [enhance] section governs the LLM prompt enhancer.
#
# default = false  → enhancer is opt-in (--enhance-prompt enables it
#                    on the CLI). Setting true here makes every run
#                    enhance unless --no-enhance is passed.
# model    = HF repo name passed to mlx_lm.load. Empty string rejected
#            (would fail later at load anyway, fail-fast at config time).
# temperature = sampler temp; 0.0 = greedy (deterministic, replay-friendly).
# max_tokens  = LLM output cap. 200 is generous for ~60-80 token expansions.
# timeout_s   = wall-clock cap on the runner subprocess; kills it if
#               mlx_lm hangs.
def _is_model_ref(v: Any) -> bool:
    """``[enhance] model`` validator: non-empty, no C0/DEL/C1 control bytes.

    The field flows into mlx_lm.load (HF repo id) AND into our terminal
    display (doctor's "Enhance" section). A control byte here could leak
    escape sequences into the user's terminal on `imgen doctor` output.
    (v0.5 security-reviewer IMP-4.)
    """
    if not (isinstance(v, str) and v.strip() != ""):
        return False
    return not any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in v
    )


ENHANCE_SCHEMA: dict[str, _SchemaEntry] = {
    "default": (
        "bool (true / false)",
        lambda v: isinstance(v, bool),
    ),
    "model": (
        "non-empty string (HF repo or absolute path, no control bytes)",
        _is_model_ref,
    ),
    "temperature": (
        "number 0.0..2.0",
        lambda v: _is_number_not_bool(v) and 0.0 <= v <= 2.0,
    ),
    "max_tokens": (
        "int 1..4096",
        lambda v: _is_int_not_bool(v) and 1 <= v <= 4096,
    ),
    "timeout_s": (
        "int 1..3600",
        lambda v: _is_int_not_bool(v) and 1 <= v <= 3600,
    ),
}


# ── Loaders + validator ──────────────────────────────────────────────────

def load_config(path: Path) -> dict[str, Any]:
    """Read TOML file. Missing → empty dict. Malformed/oversized → empty + warn.

    Pure on the file contents; no side effects beyond the warn print.

    Cap at CONFIG_MAX_BYTES (1 MB) so a rogue file can't OOM tomllib,
    which slurps the whole file before parsing.
    """
    if not path.exists():
        return {}
    try:
        size = path.stat().st_size
    except OSError as e:
        warn(f"Couldn't stat {path}: {e} — using built-in defaults")
        return {}
    if size > CONFIG_MAX_BYTES:
        warn(f"{path} too large ({size} bytes; cap {CONFIG_MAX_BYTES}) "
             "— using built-in defaults")
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        warn(f"Couldn't parse {path}: {e} — using built-in defaults")
        return {}


def validate_section(
    section_name: str,
    raw: dict[str, Any],
    schema: dict[str, _SchemaEntry],
) -> dict[str, Any]:
    """Type+value-check known keys; drop unknown keys (with warning).

    Raises ConfigError on a known key with a bad value — caller decides
    whether to die() or fall back to module DEFAULTS.

    Thin wrapper around the shared :func:`_schema.validate_against_schema`
    helper (architect IMP-2 from v0.4 review unified the three inline
    copies into one). Source label is ``"[section_name]"`` to keep the
    historical config.toml error prefix shape.
    """
    return validate_against_schema(
        raw, schema, ConfigError, source=f"[{section_name}]",
    )


def load_validated_config(path: Path) -> dict[str, dict[str, Any]]:
    """Load + validate config.toml. Returns {'defaults': {...}, 'ui': {...}}.

    Missing file → empty sections (callers use module DEFAULTS).
    Bad value in known key → ConfigError propagates so cli can die clean.
    """
    raw = load_config(path)
    return {
        "defaults": validate_section(
            "defaults", raw.get("defaults", {}), DEFAULTS_SCHEMA
        ),
        "ui": validate_section("ui", raw.get("ui", {}), UI_SCHEMA),
        "enhance": validate_section(
            "enhance", raw.get("enhance", {}), ENHANCE_SCHEMA
        ),
    }


# ── Precedence merges ────────────────────────────────────────────────────

def effective_defaults(
    config_defaults: dict[str, Any],
    module_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Merge `[defaults]` from config.toml over the module DEFAULTS.

    Does NOT mutate either input. config_defaults wins on overlap;
    keys present only in module_defaults are kept; keys present only
    in config_defaults are added.
    """
    return {**module_defaults, **config_defaults}


# Built-in defaults for the [enhance] section — used when config.toml is
# absent / a key is missing. Picked to match the v0.5 design memo:
# enhancer is opt-in (default=False), Qwen2.5-7B-4bit, deterministic
# (temp=0.0), 200-token output cap, 120-second runner timeout.
_ENHANCE_MODULE_DEFAULTS: dict[str, Any] = {
    "default": False,
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "temperature": 0.0,
    "max_tokens": 200,
    "timeout_s": 120,
}


def effective_enhance(
    cli_enable: bool | None,
    config_enhance: dict[str, Any],
    cli_model: str | None = None,
    cli_temperature: float | None = None,
) -> dict[str, Any]:
    """Resolve the effective [enhance] settings for one CLI invocation.

    Returns a dict with the same keys as ``ENHANCE_SCHEMA`` plus an
    ``"enabled"`` boolean (the resolved on/off for this invocation).

    Precedence (highest first):
        * ``cli_enable`` — explicit ``--enhance-prompt`` (True) or
          ``--no-enhance`` (False) on the CLI. None means "no CLI
          override, use config".
        * ``cli_model`` / ``cli_temperature`` — explicit CLI overrides
          for specific fields. None means "use config".
        * ``config_enhance`` — the validated ``[enhance]`` section
          from config.toml; missing keys fall to module defaults.
        * Module defaults — :data:`_ENHANCE_MODULE_DEFAULTS`.

    Does not mutate ``config_enhance``.
    """
    merged: dict[str, Any] = {**_ENHANCE_MODULE_DEFAULTS, **config_enhance}
    if cli_model is not None:
        merged["model"] = cli_model
    if cli_temperature is not None:
        merged["temperature"] = cli_temperature
    enabled = merged["default"] if cli_enable is None else cli_enable
    merged["enabled"] = enabled
    return merged


def effective_output_dir(
    cli_value: str | None = None,
    config_value: str | None = None,
    module_default: Path = Path("/"),
) -> Path:
    """Resolve the output directory.

    Precedence (highest first):
        1. cli_value from `--output-dir` (v0.2.3+) — explicit user
           intent on this invocation, beats env.
        2. $IMGEN_OUTPUT_DIR env var (v0.1.x one-off override channel).
        3. config_value from [defaults] output_dir in config.toml.
        4. module_default (DEFAULT_OUTPUT_DIR).

    `~` is expanded for cli_value / config_value. Empty strings at
    cli/config are treated as unset (`--output-dir ""` should not
    silently mean "write to cwd").
    """
    if cli_value:
        return Path(cli_value).expanduser()
    env = os.environ.get("IMGEN_OUTPUT_DIR")
    if env:
        return Path(env)
    if config_value:
        return Path(config_value).expanduser()
    return module_default

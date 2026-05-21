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

from .backends import BACKENDS
from .colors import warn
from .styles import list_styles


__all__ = [
    "ConfigError",
    "DEFAULTS_SCHEMA",
    "UI_SCHEMA",
    "load_config",
    "validate_section",
    "load_validated_config",
    "effective_defaults",
    "effective_output_dir",
]


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
        "number 0.5..15.0",
        lambda v: _is_number_not_bool(v) and 0.5 <= v <= 15.0,
    ),
    "strength": (
        "number 0.0..1.0",
        lambda v: _is_number_not_bool(v) and 0.0 <= v <= 1.0,
    ),
    "output_dir": (
        "path string",
        lambda v: isinstance(v, str),
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


# ── Loaders + validator ──────────────────────────────────────────────────

def load_config(path: Path) -> dict[str, Any]:
    """Read TOML file. Missing → empty dict. Malformed → empty + warn.

    Pure on the file contents; no side effects beyond the warn print.
    """
    if not path.exists():
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
    """
    validated: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in schema:
            warn(f"[{section_name}] unknown key '{key}' in config.toml — ignored")
            continue
        expected_desc, predicate = schema[key]
        if not predicate(value):
            raise ConfigError(
                f"[{section_name}] {key}: expected {expected_desc}, "
                f"got {value!r} ({type(value).__name__})"
            )
        validated[key] = value
    return validated


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


def effective_output_dir(
    config_value: str | None,
    module_default: Path,
) -> Path:
    """Resolve the output directory.

    Precedence:
        1. $IMGEN_OUTPUT_DIR if set (matches v0.1.x behavior)
        2. config_value from [defaults] output_dir (if non-empty)
        3. module_default

    `~` is expanded for config_value via Path.expanduser.
    """
    env = os.environ.get("IMGEN_OUTPUT_DIR")
    if env:
        return Path(env)
    if config_value:
        return Path(config_value).expanduser()
    return module_default

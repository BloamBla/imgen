"""Terminal colors + leveled print helpers.

v0.2.2 moved the enable/disable resolution from import-time
(`_USE_COLOR = sys.stdout.isatty()` frozen for the process) to lazy on
first access. Order:

  1. `NO_COLOR` env var (https://no-color.org/) — any non-empty value
     disables. Beats config + tty.
  2. `[ui] color` from ~/.imgen/config.toml — "auto" / "always" / "never".
     - "always" → enabled regardless of tty.
     - "never"  → disabled regardless of tty.
     - "auto"   → fall through to tty check.
  3. Fallback: `sys.stdout.isatty()`.

The result is cached for the process. Tests call `reset_color_cache()`
between cases. The `C` namespace (`C.OK`, `C.WARN`, ...) returns the ANSI
escape or `""` per attribute access, so all existing `f"{C.OK}..."` call
sites keep working without code change.
"""
from __future__ import annotations

import os
import sys
from typing import NoReturn

__all__ = [
    "C",
    "color_enabled",
    "reset_color_cache",
    "ok",
    "warn",
    "err",
    "info",
    "step",
    "dim",
    "die",
]


# ── Enable/disable resolution ───────────────────────────────────────────

_color_enabled_cache: bool | None = None


def color_enabled() -> bool:
    """Return whether ANSI color should be emitted. Cached after first call."""
    global _color_enabled_cache
    if _color_enabled_cache is not None:
        return _color_enabled_cache
    # Prime with the tty/NO_COLOR default so any color access during the
    # config load (warnings printed by malformed config.toml etc.) gets
    # a sensible value instead of re-entering this function and blowing
    # the stack.
    _color_enabled_cache = (
        sys.stdout.isatty() and not _no_color_env_set()
    )
    _color_enabled_cache = _compute_color_enabled()
    return _color_enabled_cache


def reset_color_cache() -> None:
    """Clear the cached resolution. For tests + manual config reloads."""
    global _color_enabled_cache
    _color_enabled_cache = None


def _no_color_env_set() -> bool:
    """True if NO_COLOR is set to any non-empty value (no-color.org spec)."""
    return bool(os.environ.get("NO_COLOR"))


def _compute_color_enabled() -> bool:
    if _no_color_env_set():
        return False
    mode = _resolve_ui_color()
    if mode == "never":
        return False
    if mode == "always":
        return True
    return sys.stdout.isatty()


def _resolve_ui_color() -> str:
    """Read `[ui] color` from config.toml. Returns "auto" if unavailable.

    Swallows all exceptions: a broken config must not break terminal
    colors. The config validator surfaces real errors elsewhere (cli +
    doctor).
    """
    try:
        from . import paths as paths_mod
        from .config import load_validated_config
        cfg = load_validated_config(paths_mod.CONFIG_FILE)
        return cfg.get("ui", {}).get("color", "auto")
    except Exception:
        return "auto"


# ── Color namespace ─────────────────────────────────────────────────────

class _ColorNamespace:
    """Per-attribute ANSI lookup. `C.OK` → "\\033[92m" or "" at call time."""

    _CODES = {
        "OK": "\033[92m",
        "WARN": "\033[93m",
        "ERR": "\033[91m",
        "INFO": "\033[94m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "END": "\033[0m",
    }

    def __getattr__(self, name: str) -> str:
        if name in self._CODES:
            return self._CODES[name] if color_enabled() else ""
        raise AttributeError(f"colors.C has no attribute {name!r}")


C = _ColorNamespace()


# ── Leveled print helpers ───────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"{C.OK}✅{C.END} {msg}")


def warn(msg: str) -> None:
    print(f"{C.WARN}⚠️ {C.END} {msg}")


def err(msg: str) -> None:
    print(f"{C.ERR}❌{C.END} {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{C.INFO}🔍{C.END} {msg}")


def step(msg: str) -> None:
    print(f"{C.BOLD}{C.INFO}🚀 {msg}{C.END}")


def dim(msg: str) -> None:
    print(f"{C.DIM}{msg}{C.END}")


def die(msg: str, code: int = 1, hint: str | None = None) -> NoReturn:
    """Print ``msg`` (red), optional hint (dim), then ``sys.exit(code)``.

    v0.7.0 (python pre-tag review IMPORTANT): return annotation
    ``NoReturn`` makes mypy / pyright understand that callers don't
    need to handle the fall-through path. ``_resolve_draw_prompt`` and
    similar helpers can have a trailing ``die(...)`` instead of an
    ``if/else`` ladder without the type-checker complaining about a
    missing return.
    """
    err(msg)
    if hint:
        print(f"   {C.DIM}{hint}{C.END}", file=sys.stderr)
    sys.exit(code)

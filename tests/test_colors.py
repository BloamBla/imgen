"""Dynamic color resolution: NO_COLOR env, [ui] color config, tty fallback.

Before v0.3 `_USE_COLOR = sys.stdout.isatty()` was evaluated at import
and frozen for the process. v0.3 makes it lazy so:
  - `NO_COLOR` env (https://no-color.org/) takes precedence.
  - `[ui] color` from ~/.imgen/config.toml: auto / always / never.
  - Fallback: stdout.isatty().

All ~50 `C.OK` / `C.WARN` / ... call sites stay unchanged via a small
descriptor namespace whose attributes return the ANSI code or '' at
access time.
"""
from __future__ import annotations

import pytest

import imgen.colors as colors_mod
from imgen.colors import C, color_enabled, reset_color_cache


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Fresh color cache + no NO_COLOR env at the start and end of every test."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    reset_color_cache()
    yield
    reset_color_cache()


@pytest.fixture
def fake_isatty(monkeypatch):
    """Force sys.stdout.isatty() to return whatever we hand in."""
    def _set(value: bool) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: value)
    return _set


@pytest.fixture
def fake_ui_color(monkeypatch):
    """Stub _resolve_ui_color() so tests don't depend on a real config.toml."""
    def _set(mode: str) -> None:
        monkeypatch.setattr(colors_mod, "_resolve_ui_color", lambda: mode)
    return _set


# ── color_enabled() decision matrix ─────────────────────────────────────

def test_auto_mode_enables_when_stdout_is_tty(fake_isatty, fake_ui_color):
    fake_ui_color("auto")
    fake_isatty(True)
    assert color_enabled() is True


def test_auto_mode_disables_when_stdout_is_pipe(fake_isatty, fake_ui_color):
    fake_ui_color("auto")
    fake_isatty(False)
    assert color_enabled() is False


def test_always_mode_enables_even_without_tty(fake_isatty, fake_ui_color):
    fake_ui_color("always")
    fake_isatty(False)
    assert color_enabled() is True


def test_never_mode_disables_even_on_tty(fake_isatty, fake_ui_color):
    fake_ui_color("never")
    fake_isatty(True)
    assert color_enabled() is False


def test_no_color_env_overrides_always_mode(fake_isatty, fake_ui_color, monkeypatch):
    """NO_COLOR env wins over [ui] color = 'always'."""
    fake_ui_color("always")
    fake_isatty(True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled() is False


def test_no_color_env_any_non_empty_value_disables(fake_isatty, fake_ui_color, monkeypatch):
    """https://no-color.org/: 'when present, regardless of value' — except
    empty string which is treated as unset per the spec's wording."""
    fake_ui_color("always")
    fake_isatty(True)
    monkeypatch.setenv("NO_COLOR", "yes-please-no-color")
    assert color_enabled() is False


def test_no_color_env_empty_string_does_not_disable(fake_isatty, fake_ui_color, monkeypatch):
    """NO_COLOR='' should be treated as unset (spec ambiguity, but empty
    string in shells often means 'not really set' — match GNU coreutils)."""
    fake_ui_color("always")
    fake_isatty(False)
    monkeypatch.setenv("NO_COLOR", "")
    assert color_enabled() is True


# ── C.* attribute access ────────────────────────────────────────────────

def test_C_returns_ansi_codes_when_color_enabled(fake_isatty, fake_ui_color):
    fake_ui_color("always")
    fake_isatty(True)
    assert C.OK == "\033[92m"
    assert C.WARN == "\033[93m"
    assert C.ERR == "\033[91m"
    assert C.INFO == "\033[94m"
    assert C.BOLD == "\033[1m"
    assert C.DIM == "\033[2m"
    assert C.END == "\033[0m"


def test_C_returns_empty_strings_when_color_disabled(fake_isatty, fake_ui_color):
    fake_ui_color("never")
    fake_isatty(True)
    assert C.OK == ""
    assert C.WARN == ""
    assert C.END == ""


def test_C_unknown_attribute_raises_attribute_error(fake_isatty, fake_ui_color):
    fake_ui_color("always")
    fake_isatty(True)
    with pytest.raises(AttributeError):
        _ = C.NOT_A_REAL_COLOR


# ── Caching ─────────────────────────────────────────────────────────────

def test_color_enabled_result_is_cached(fake_isatty, fake_ui_color, monkeypatch):
    """Repeated calls don't re-invoke the resolver (~50 color sites per run)."""
    calls = {"n": 0}

    def counting_resolver() -> str:
        calls["n"] += 1
        return "always"

    monkeypatch.setattr(colors_mod, "_resolve_ui_color", counting_resolver)
    fake_isatty(True)

    color_enabled()
    color_enabled()
    color_enabled()

    assert calls["n"] == 1


def test_reset_color_cache_forces_re_evaluation(fake_isatty, fake_ui_color):
    """Tests rely on this; manual flips between modes also need it."""
    fake_ui_color("always")
    fake_isatty(False)
    assert color_enabled() is True

    fake_ui_color("never")
    assert color_enabled() is True  # still cached

    reset_color_cache()
    assert color_enabled() is False


# ── _resolve_ui_color() reads config.toml ───────────────────────────────

def test_resolve_ui_color_reads_color_value_from_config_toml(tmp_path, monkeypatch):
    import imgen.paths as paths_mod
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ui]\ncolor = "never"\n')
    monkeypatch.setattr(paths_mod, "CONFIG_FILE", cfg)

    assert colors_mod._resolve_ui_color() == "never"


def test_resolve_ui_color_defaults_to_auto_when_config_missing(tmp_path, monkeypatch):
    import imgen.paths as paths_mod
    monkeypatch.setattr(paths_mod, "CONFIG_FILE", tmp_path / "missing.toml")

    assert colors_mod._resolve_ui_color() == "auto"


def test_resolve_ui_color_defaults_to_auto_when_config_malformed(tmp_path, monkeypatch, capsys):
    """A broken config.toml must not break terminal colors."""
    import imgen.paths as paths_mod
    cfg = tmp_path / "config.toml"
    cfg.write_text("this is [[[[ not toml")
    monkeypatch.setattr(paths_mod, "CONFIG_FILE", cfg)

    assert colors_mod._resolve_ui_color() == "auto"


def test_resolve_ui_color_defaults_to_auto_when_bad_value(tmp_path, monkeypatch):
    """Bad [ui] color value raises ConfigError; resolver swallows + returns auto."""
    import imgen.paths as paths_mod
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ui]\ncolor = "rainbow"\n')
    monkeypatch.setattr(paths_mod, "CONFIG_FILE", cfg)

    assert colors_mod._resolve_ui_color() == "auto"


# ── Re-entrancy: config parse warnings must not cause infinite recursion ─

def test_color_access_during_config_load_does_not_recurse(tmp_path, monkeypatch):
    """`warn()` printed by config parsing uses C.WARN → would re-enter
    color_enabled() if not guarded. Prove a broken config doesn't blow up."""
    import imgen.paths as paths_mod
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ui]\nopen_in_preview = "not a bool"\n')  # raises ConfigError
    monkeypatch.setattr(paths_mod, "CONFIG_FILE", cfg)

    # Just calling color_enabled triggers _resolve → config parse → warn → C.WARN.
    # If recursion is unguarded this RecursionErrors.
    _ = C.WARN
    assert color_enabled() in (True, False)  # any answer fine; we want no crash

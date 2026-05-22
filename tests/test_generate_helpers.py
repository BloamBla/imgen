"""Tests for private helpers extracted from cmd_generate during v0.2.4.

cmd_generate grew to ~590 lines / 18 phases by v0.2.3. v0.2.4 extracts
its pre-build phases into named helpers (architect item I1). These
tests lock in the behaviour of each helper so the extraction is a
pure mechanical move with no semantic shift.

Patterns:
- `die()` exits via sys.exit; helpers calling it are caught with
  `pytest.raises(SystemExit)` + `exc_info.value.code`.
- `args` mimicked via `types.SimpleNamespace` so tests don't need to
  build a full argparse Namespace.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from types import SimpleNamespace

from imgen.commands.generate import (
    _check_prompt_style_compat,
    _resolve_output_layout,
    _resolve_styles_list,
    _validate_input_path,
)


# ── _validate_input_path ────────────────────────────────────────────────

def test_validate_input_path_existing_file_returns_resolved(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake jpeg bytes")

    result = _validate_input_path(str(img))

    assert result == img.resolve()
    assert result.is_absolute()


def test_validate_input_path_expands_tilde(tmp_path, monkeypatch):
    """`~/photo.jpg` must expand — colleagues drop pictures into ~/Desktop
    and reference them with ~ in shell."""
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")

    result = _validate_input_path("~/photo.jpg")

    assert result == img.resolve()


def test_validate_input_path_resolves_relative(tmp_path, monkeypatch):
    """Relative paths must become absolute so downstream `mflux` invoke
    doesn't depend on cwd."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    result = _validate_input_path("photo.jpg")

    assert result == img.resolve()
    assert result.is_absolute()


def test_validate_input_path_missing_file_exits_code_2(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jpg"

    with pytest.raises(SystemExit) as exc_info:
        _validate_input_path(str(missing))

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Image not found" in err
    assert str(missing) in err


def test_validate_input_path_directory_exits_code_2(tmp_path, capsys):
    """If the user points at a folder we reject — only file inputs make
    sense for image→style transfer."""
    folder = tmp_path / "not-a-file"
    folder.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        _validate_input_path(str(folder))

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Not a file" in err


# ── _resolve_styles_list ────────────────────────────────────────────────

def _args(style=None, output=None) -> SimpleNamespace:
    """Minimal args for _resolve_styles_list."""
    return SimpleNamespace(style=style, output=output)


def test_resolve_styles_list_uses_explicit_list_when_passed():
    """Parser already validated names + de-duped; helper passes through."""
    result = _resolve_styles_list(
        _args(style=["anime", "ghibli"]),
        merged_defaults={"style": "anime"},
    )
    assert result == ["anime", "ghibli"]


def test_resolve_styles_list_single_explicit_style_preserved():
    result = _resolve_styles_list(
        _args(style=["pixar"]),
        merged_defaults={"style": "anime"},
    )
    assert result == ["pixar"]


def test_resolve_styles_list_falls_back_to_default_when_unspecified():
    result = _resolve_styles_list(
        _args(style=None),
        merged_defaults={"style": "anime"},
    )
    assert result == ["anime"]


def test_resolve_styles_list_unknown_default_exits_code_2(capsys):
    """If [defaults] style in config.toml points at a missing preset,
    fail fast with a clear hint mentioning the config path."""
    with pytest.raises(SystemExit) as exc_info:
        _resolve_styles_list(
            _args(style=None),
            merged_defaults={"style": "nonexistent-preset"},
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Default style 'nonexistent-preset' not found" in err


def test_resolve_styles_list_output_file_with_multi_style_rejected(capsys):
    """--output FILE writes to one path — M styles would clobber the
    same destination M times. Caller must use --output-dir for batches."""
    with pytest.raises(SystemExit) as exc_info:
        _resolve_styles_list(
            _args(style=["anime", "ghibli"], output="/tmp/forced.png"),
            merged_defaults={"style": "anime"},
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "writes to one path" in err
    assert "anime" in err and "ghibli" in err


def test_resolve_styles_list_output_file_with_single_style_ok():
    """Single-style + --output FILE is the legitimate v0.1.x use case
    — must keep working unchanged."""
    result = _resolve_styles_list(
        _args(style=["anime"], output="/tmp/forced.png"),
        merged_defaults={"style": "anime"},
    )
    assert result == ["anime"]


def test_resolve_styles_list_output_file_with_default_single_ok():
    """No --style, --output set → default style applies, no rejection."""
    result = _resolve_styles_list(
        _args(style=None, output="/tmp/forced.png"),
        merged_defaults={"style": "anime"},
    )
    assert result == ["anime"]


# ── _check_prompt_style_compat ──────────────────────────────────────────


@pytest.fixture
def fake_styles(monkeypatch):
    """Stub get_style with a controlled registry per test.

    Lets us mix prompt-bearing and param-only styles without touching
    the real built-in registry or styles.d/. The helper imports
    get_style locally; patch at the call site
    (imgen.commands.generate.get_style)."""
    registry: dict = {}

    def fake_get_style(name: str) -> dict:
        if name not in registry:
            raise KeyError(f"Unknown style '{name}'")
        return registry[name]

    monkeypatch.setattr(
        "imgen.commands.generate.get_style", fake_get_style
    )
    return registry


def test_check_prompt_style_compat_custom_prompt_with_param_only_ok(fake_styles):
    """Param-only style (no `prompt` key) + custom-prompt is the
    legitimate combo: style contributes guidance/strength/etc., CLI
    contributes prompt text."""
    fake_styles["paramonly"] = {"strength": 0.6}  # no `prompt` key

    # Must not raise — returns None on success.
    _check_prompt_style_compat(
        styles_list=["paramonly"],
        effective_custom_prompt="my custom prompt",
    )


def test_check_prompt_style_compat_custom_prompt_with_prompt_bearing_rejected(
    fake_styles, capsys
):
    """Style that ships its own prompt can't combine with --custom-prompt
    — would be two prompts fighting for the slot."""
    fake_styles["anime"] = {"prompt": "anime portrait", "strength": 0.6}

    with pytest.raises(SystemExit) as exc_info:
        _check_prompt_style_compat(
            styles_list=["anime"],
            effective_custom_prompt="my custom prompt",
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "can't combine with --custom-prompt" in err
    assert "anime" in err


def test_check_prompt_style_compat_lists_all_offenders_in_multi_style(
    fake_styles, capsys
):
    """User should see every clashing style at once, not have to fix
    one and re-run to discover the next."""
    fake_styles["anime"] = {"prompt": "anime portrait", "strength": 0.6}
    fake_styles["ghibli"] = {"prompt": "ghibli scene", "strength": 0.5}
    fake_styles["paramonly"] = {"strength": 0.7}

    with pytest.raises(SystemExit):
        _check_prompt_style_compat(
            styles_list=["anime", "paramonly", "ghibli"],
            effective_custom_prompt="custom",
        )
    err = capsys.readouterr().err
    assert "anime" in err
    assert "ghibli" in err
    # paramonly is fine — should NOT be listed as an offender
    assert "paramonly" not in err.split("can't combine")[1].split(".")[0]


def test_check_prompt_style_compat_no_prompt_with_prompt_bearing_style_ok(
    fake_styles,
):
    """No custom prompt + style with its own prompt = v0.1.x default
    path. Must remain unchanged."""
    fake_styles["anime"] = {"prompt": "anime portrait", "strength": 0.6}

    _check_prompt_style_compat(
        styles_list=["anime"],
        effective_custom_prompt=None,
    )


def test_check_prompt_style_compat_no_prompt_with_param_only_rejected(
    fake_styles, capsys
):
    """Param-only style (e.g. user-added in styles.d/) needs a CLI
    prompt — no prompt + no style prompt = nothing to feed mflux."""
    fake_styles["paramonly"] = {"strength": 0.6}

    with pytest.raises(SystemExit) as exc_info:
        _check_prompt_style_compat(
            styles_list=["paramonly"],
            effective_custom_prompt=None,
        )
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "without a prompt" in err
    assert "paramonly" in err


def test_check_prompt_style_compat_empty_string_prompt_treated_as_missing(
    fake_styles,
):
    """A style with `prompt: ""` is effectively param-only — falsy
    string in `if get_style(s).get("prompt")` matches the predicate."""
    fake_styles["empty"] = {"prompt": "", "strength": 0.5}

    # No CLI prompt + falsy style prompt → reject (matches mutex semantics).
    with pytest.raises(SystemExit):
        _check_prompt_style_compat(
            styles_list=["empty"],
            effective_custom_prompt=None,
        )


# ── _resolve_output_layout ──────────────────────────────────────────────


@pytest.fixture
def fixed_run_dirname(monkeypatch):
    """Pin auto_run_dirname → '2026-05-22-10-00-00' for deterministic tests."""
    monkeypatch.setattr(
        "imgen.commands.generate.auto_run_dirname",
        lambda now=None: "2026-05-22-10-00-00",
    )


def test_resolve_output_layout_explicit_file_returns_path_no_run_dir(
    tmp_path,
):
    """--output FILE bypasses the run-folder layout entirely. Returns
    (resolved_path, None)."""
    target = tmp_path / "forced.png"
    args = SimpleNamespace(output=str(target))

    explicit_output, run_dir = _resolve_output_layout(
        args, config_output_dir=None
    )

    assert explicit_output == target.resolve()
    assert run_dir is None


def test_resolve_output_layout_explicit_file_expands_tilde(
    tmp_path, monkeypatch
):
    """`--output ~/Pictures/out.png` must expand."""
    monkeypatch.setenv("HOME", str(tmp_path))
    args = SimpleNamespace(output="~/out.png")

    explicit_output, run_dir = _resolve_output_layout(
        args, config_output_dir=None
    )

    assert explicit_output == (tmp_path / "out.png").resolve()
    assert run_dir is None


def test_resolve_output_layout_default_uses_module_default(
    tmp_path, monkeypatch, fixed_run_dirname
):
    """No --output, no --output-dir, no config → fall back to module
    DEFAULT_OUTPUT_DIR. Verify via monkeypatch since real default is
    ~/Desktop/imgen (don't want to write there from tests)."""
    fake_default = tmp_path / "fake_default"
    monkeypatch.setattr(
        "imgen.commands.generate.DEFAULT_OUTPUT_DIR", fake_default
    )
    args = SimpleNamespace(output=None, output_dir=None)

    explicit_output, run_dir = _resolve_output_layout(
        args, config_output_dir=None
    )

    assert explicit_output is None
    assert run_dir == fake_default / "2026-05-22-10-00-00"


def test_resolve_output_layout_cli_output_dir_beats_config(
    tmp_path, fixed_run_dirname
):
    """CLI > config > module default. --output-dir wins even if config
    sets a different one."""
    cli_dir = tmp_path / "cli"
    config_dir = tmp_path / "config"
    args = SimpleNamespace(output=None, output_dir=str(cli_dir))

    explicit_output, run_dir = _resolve_output_layout(
        args, config_output_dir=str(config_dir)
    )

    assert explicit_output is None
    assert run_dir == cli_dir / "2026-05-22-10-00-00"


def test_resolve_output_layout_config_beats_module_default(
    tmp_path, monkeypatch, fixed_run_dirname
):
    """No CLI --output-dir, config set → config used."""
    monkeypatch.setattr(
        "imgen.commands.generate.DEFAULT_OUTPUT_DIR",
        tmp_path / "module_default",
    )
    config_dir = tmp_path / "config"
    args = SimpleNamespace(output=None, output_dir=None)

    explicit_output, run_dir = _resolve_output_layout(
        args, config_output_dir=str(config_dir)
    )

    assert explicit_output is None
    assert run_dir == config_dir / "2026-05-22-10-00-00"


def test_resolve_output_layout_does_not_create_run_dir(
    tmp_path, fixed_run_dirname
):
    """Pure: returns the path that *would* be used, caller mkdir's after
    confirm gate (so cancel doesn't orphan an empty dir)."""
    args = SimpleNamespace(output=None, output_dir=str(tmp_path))

    _, run_dir = _resolve_output_layout(args, config_output_dir=None)

    assert not run_dir.exists()


def test_resolve_output_layout_suffixes_run_dir_on_collision(
    tmp_path, fixed_run_dirname
):
    """If the auto-named folder already exists (rare — scripted double
    invoke), next_available_run_dir adds `_2`."""
    (tmp_path / "2026-05-22-10-00-00").mkdir()
    args = SimpleNamespace(output=None, output_dir=str(tmp_path))

    _, run_dir = _resolve_output_layout(args, config_output_dir=None)

    assert run_dir == tmp_path / "2026-05-22-10-00-00_2"

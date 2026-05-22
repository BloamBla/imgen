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

from imgen.backends import BACKENDS
from imgen.commands.generate import (
    _build_iterations,
    _check_prompt_style_compat,
    _exit_code,
    _load_backend_and_token,
    _open_results,
    _preflight_resources,
    _print_batch_summary,
    _resolve_output_layout,
    _resolve_styles_list,
    _run_one_iteration,
    _validate_input_path,
)
from imgen.runs import BatchContext, BatchLogger, Iteration


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


# ── _load_backend_and_token ─────────────────────────────────────────────


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    """Stub VENV_BIN to a tmp dir and pre-create both backend binaries.

    Lets tests assert binary path resolution without depending on a real
    mflux install. Individual tests may delete a binary to exercise the
    missing-binary branch."""
    venv = tmp_path / "venv-bin"
    venv.mkdir()
    (venv / "mflux-generate-kontext").write_bytes(b"#!/bin/sh\n")
    (venv / "mflux-generate-qwen-edit").write_bytes(b"#!/bin/sh\n")
    monkeypatch.setattr("imgen.commands.generate.VENV_BIN", venv)
    return venv


@pytest.fixture
def passing_checks(monkeypatch):
    """Stub check_venv + check_mflux to True. Individual tests override
    to False to exercise the not-installed branch."""
    monkeypatch.setattr("imgen.commands.generate.check_venv", lambda: True)
    monkeypatch.setattr("imgen.commands.generate.check_mflux", lambda: True)


def test_load_backend_and_token_flux_with_token(
    fake_venv, passing_checks, monkeypatch
):
    monkeypatch.setattr(
        "imgen.commands.generate.load_token", lambda: "hf_TOKEN_VALUE"
    )

    backend, be, token, binary = _load_backend_and_token(
        SimpleNamespace(backend="flux")
    )

    assert backend == "flux"
    assert be.needs_token is True
    assert token == "hf_TOKEN_VALUE"
    assert binary == fake_venv / "mflux-generate-kontext"


def test_load_backend_and_token_flux_without_token_exits_3(
    fake_venv, passing_checks, monkeypatch, capsys
):
    """FLUX is gated — missing HF token must fail fast with the setup
    hint, not silently call mflux and hit a 401 mid-run."""
    monkeypatch.setattr("imgen.commands.generate.load_token", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        _load_backend_and_token(SimpleNamespace(backend="flux"))

    assert exc_info.value.code == 3
    err = capsys.readouterr().err
    assert "FLUX backend requires HuggingFace token" in err


def test_load_backend_and_token_qwen_no_token_needed(
    fake_venv, passing_checks, monkeypatch
):
    """Qwen is open-weights — no token. load_token must not even be
    called (defensive: avoids touching the keyring/disk for nothing)."""
    called = []
    monkeypatch.setattr(
        "imgen.commands.generate.load_token",
        lambda: called.append("load_token") or None,
    )

    backend, be, token, binary = _load_backend_and_token(
        SimpleNamespace(backend="qwen")
    )

    assert backend == "qwen"
    assert be.needs_token is False
    assert token is None
    assert binary == fake_venv / "mflux-generate-qwen-edit"
    assert called == [], "load_token() should NOT be invoked for open backends"


def test_load_backend_and_token_venv_check_failure_exits_3(
    fake_venv, monkeypatch, capsys
):
    monkeypatch.setattr("imgen.commands.generate.check_venv", lambda: False)
    monkeypatch.setattr("imgen.commands.generate.check_mflux", lambda: True)
    monkeypatch.setattr("imgen.commands.generate.load_token", lambda: "x")

    with pytest.raises(SystemExit) as exc_info:
        _load_backend_and_token(SimpleNamespace(backend="qwen"))

    assert exc_info.value.code == 3
    assert "mflux not installed" in capsys.readouterr().err


def test_load_backend_and_token_mflux_check_failure_exits_3(
    fake_venv, monkeypatch, capsys
):
    monkeypatch.setattr("imgen.commands.generate.check_venv", lambda: True)
    monkeypatch.setattr("imgen.commands.generate.check_mflux", lambda: False)
    monkeypatch.setattr("imgen.commands.generate.load_token", lambda: "x")

    with pytest.raises(SystemExit) as exc_info:
        _load_backend_and_token(SimpleNamespace(backend="qwen"))

    assert exc_info.value.code == 3
    assert "mflux not installed" in capsys.readouterr().err


def test_load_backend_and_token_missing_binary_exits_3(
    fake_venv, passing_checks, monkeypatch, capsys
):
    """venv reports installed but the per-backend binary is missing —
    half-broken state, point at `imgen upgrade`."""
    (fake_venv / "mflux-generate-qwen-edit").unlink()
    monkeypatch.setattr("imgen.commands.generate.load_token", lambda: "x")

    with pytest.raises(SystemExit) as exc_info:
        _load_backend_and_token(SimpleNamespace(backend="qwen"))

    assert exc_info.value.code == 3
    err = capsys.readouterr().err
    assert "Backend binary not found" in err
    assert "imgen upgrade" in err


# ── _build_iterations ───────────────────────────────────────────────────


def _build_args(**overrides) -> SimpleNamespace:
    """argparse Namespace shape that _build_iterations reads from.

    Default values mirror "no CLI overrides" so each test only sets the
    one field it cares about."""
    defaults = dict(
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        scope=None,
        preview=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


_FULL_DEFAULTS = {
    "steps": 14,
    "quantize": 8,
    "guidance": 2.5,
    "strength": 0.6,
    "mlx_cache_gb": 8,
    "battery_stop": 20,
    "style": "anime",
}


def _run_dir_at(tmp_path) -> Path:
    """Tests build iteration list against a non-existent run_dir — pure,
    no FS side effects."""
    return tmp_path / "run-2026-05-22-10-00-00"


def _build(*, fake_styles, tmp_path, **overrides) -> list[Iteration]:
    """Call _build_iterations with a sensible default kwarg set,
    overriding only what the test cares about."""
    kwargs = dict(
        styles_list=["anime"],
        args=_build_args(),
        effective_custom_prompt=None,
        merged_defaults=_FULL_DEFAULTS,
        be=BACKENDS["flux"],
        binary=Path("/fake/bin/mflux-generate-kontext"),
        input_path=tmp_path / "photo.jpg",
        width=1024,
        height=1024,
        explicit_output=None,
        run_dir=_run_dir_at(tmp_path),
        seed=42,
    )
    kwargs.update(overrides)
    return _build_iterations(**kwargs)


def test_build_iterations_single_style_preset_prompt(fake_styles, tmp_path):
    """No CLI overrides, no scope → prompt comes verbatim from preset."""
    fake_styles["anime"] = {
        "prompt": "cinematic anime portrait of this person",
        "negative": "bad anatomy",
    }

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert len(its) == 1
    it = its[0]
    assert isinstance(it, Iteration)
    assert it.style_name == "anime"
    assert it.prompt == "cinematic anime portrait of this person"
    assert it.negative == "bad anatomy"


def test_build_iterations_negative_defaults_to_empty(fake_styles, tmp_path):
    """Style without `negative` key → it.negative == "" (not None,
    not missing — mflux gets an empty string)."""
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].negative == ""


def test_build_iterations_uses_custom_prompt_over_preset(fake_styles, tmp_path):
    fake_styles["anime"] = {"prompt": "PRESET PROMPT"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        effective_custom_prompt="MY OVERRIDE",
    )

    assert its[0].prompt == "MY OVERRIDE"


def test_build_iterations_custom_prompt_ignores_scope(fake_styles, tmp_path):
    """Scope-warn fires elsewhere (cmd_generate); the helper just must
    NOT apply scope to a custom prompt — scope mutates 'this person'
    inside preset prompts only."""
    fake_styles["anime"] = {"prompt": "this person ignored"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        effective_custom_prompt="this person",  # has the scope-substring
        args=_build_args(scope="man"),
    )

    # Custom prompt passed through verbatim — no scope substitution.
    assert its[0].prompt == "this person"


def test_build_iterations_scope_applied_to_preset_prompt(fake_styles, tmp_path):
    """No custom prompt + scope=scene → apply_scope rewrites 'this person'
    → 'this entire scene' in the preset prompt."""
    fake_styles["anime"] = {"prompt": "portrait of this person, anime style"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(scope="scene"),
    )

    # scope=scene swaps "this person" → "this entire scene".
    assert "this person" not in its[0].prompt
    assert "this entire scene" in its[0].prompt


def test_build_iterations_scope_person_adds_suffix(fake_styles, tmp_path):
    """scope=person appends a background-preservation suffix."""
    fake_styles["anime"] = {"prompt": "portrait of this person"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(scope="person"),
    )

    assert its[0].prompt.startswith("portrait of this person")
    assert "background" in its[0].prompt


# ── _build_iterations precedence: CLI > preset > preview > defaults ─────

def test_build_iterations_steps_cli_beats_preset_and_preview(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x", "steps": 18}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(steps=25, preview=True),
    )

    assert its[0].final_steps == 25


def test_build_iterations_steps_preview_beats_defaults(fake_styles, tmp_path):
    """Note: preview overrides preset+default for steps (PREVIEW_OVERRIDES
    is checked before merged_defaults, but AFTER CLI). Preset.steps is
    NOT in the precedence chain for steps — only CLI > preview > merged
    defaults applies. This is intentional from v0.1.x — preset shouldn't
    pin steps because the user picks --preview for speed."""
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(preview=True),
    )

    # PREVIEW_OVERRIDES["steps"] == 8 in current defaults; assert via import.
    from imgen.defaults import PREVIEW_OVERRIDES
    assert its[0].final_steps == PREVIEW_OVERRIDES["steps"]


def test_build_iterations_steps_falls_back_to_merged_defaults(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].final_steps == _FULL_DEFAULTS["steps"]


def test_build_iterations_quantize_cli_wins(fake_styles, tmp_path):
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(quantize=4, preview=True),
    )

    assert its[0].final_quantize == 4


def test_build_iterations_quantize_preview_beats_defaults(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(preview=True),
    )

    from imgen.defaults import PREVIEW_OVERRIDES
    assert its[0].final_quantize == PREVIEW_OVERRIDES["quantize"]


def test_build_iterations_guidance_cli_beats_preset(fake_styles, tmp_path):
    fake_styles["anime"] = {"prompt": "x", "guidance": 3.5}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(guidance=5.0),
    )

    assert its[0].final_guidance == pytest.approx(5.0)


def test_build_iterations_guidance_preset_beats_defaults(
    fake_styles, tmp_path
):
    """Unlike steps, guidance precedence is CLI > preset > defaults —
    preview has no opinion on guidance."""
    fake_styles["anime"] = {"prompt": "x", "guidance": 3.5}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].final_guidance == pytest.approx(3.5)


def test_build_iterations_guidance_falls_back_to_defaults(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x"}  # no guidance key

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].final_guidance == pytest.approx(
        _FULL_DEFAULTS["guidance"]
    )


def test_build_iterations_strength_cli_beats_preset(fake_styles, tmp_path):
    fake_styles["anime"] = {"prompt": "x", "strength": 0.55}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        args=_build_args(strength=0.9),
    )

    assert its[0].final_strength == pytest.approx(0.9)


def test_build_iterations_strength_preset_beats_defaults(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x", "strength": 0.55}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].final_strength == pytest.approx(0.55)


def test_build_iterations_strength_falls_back_to_defaults(
    fake_styles, tmp_path
):
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert its[0].final_strength == pytest.approx(
        _FULL_DEFAULTS["strength"]
    )


# ── _build_iterations multi-style + output_path resolution ──────────────

def test_build_iterations_multi_style_one_per_name(fake_styles, tmp_path):
    """List length matches styles_list len; order preserved."""
    fake_styles["anime"] = {"prompt": "a"}
    fake_styles["ghibli"] = {"prompt": "g"}
    fake_styles["pixar"] = {"prompt": "p"}

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        styles_list=["anime", "ghibli", "pixar"],
    )

    assert [it.style_name for it in its] == ["anime", "ghibli", "pixar"]
    assert [it.prompt for it in its] == ["a", "g", "p"]


def test_build_iterations_output_path_uses_run_dir(fake_styles, tmp_path):
    """run_dir mode: each iteration → <run_dir>/<input.stem>-<style>.png."""
    fake_styles["anime"] = {"prompt": "x"}
    fake_styles["ghibli"] = {"prompt": "y"}
    input_path = tmp_path / "vacation.jpg"
    run_dir = tmp_path / "run-1"

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        styles_list=["anime", "ghibli"],
        input_path=input_path,
        explicit_output=None,
        run_dir=run_dir,
    )

    assert its[0].output_path == run_dir / "vacation-anime.png"
    assert its[1].output_path == run_dir / "vacation-ghibli.png"


def test_build_iterations_explicit_output_overrides_run_dir(
    fake_styles, tmp_path
):
    """--output FILE mode: every iteration uses the explicit path. (In
    practice multi-style + --output is rejected upstream, but the
    helper itself doesn't enforce — defensive that it still produces
    something coherent.)"""
    fake_styles["anime"] = {"prompt": "x"}
    explicit = tmp_path / "forced.png"

    its = _build(
        fake_styles=fake_styles,
        tmp_path=tmp_path,
        explicit_output=explicit,
        run_dir=None,
    )

    assert its[0].output_path == explicit


# ── _build_iterations contract guarantees ───────────────────────────────

def test_build_iterations_returns_iteration_dataclass_not_dict(
    fake_styles, tmp_path
):
    """Item 2's Iteration is the new typed contract — verify the helper
    returns it, not a dict (v0.2.3 shape)."""
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert all(isinstance(it, Iteration) for it in its)


def test_build_iterations_iteration_is_frozen(fake_styles, tmp_path):
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    with pytest.raises((AttributeError, TypeError)):
        its[0].style_name = "ghibli"  # type: ignore[misc]


def test_build_iterations_cmd_is_list_of_str(fake_styles, tmp_path):
    """Smoke check: build_mflux_cmd was called, result stored in
    it.cmd as a list of strings ready for subprocess.Popen."""
    fake_styles["anime"] = {"prompt": "x"}

    its = _build(fake_styles=fake_styles, tmp_path=tmp_path)

    assert isinstance(its[0].cmd, list)
    assert all(isinstance(arg, str) for arg in its[0].cmd)
    assert "/fake/bin/mflux-generate-kontext" in its[0].cmd


# ── _exit_code ──────────────────────────────────────────────────────────


def _ok(name: str) -> tuple[str, Path, int]:
    """succeeded-list entry shape: (style_name, output_path, duration_s)."""
    return (name, Path(f"/tmp/{name}.png"), 1)


def _fail(name: str, rc: int) -> tuple[str, int, Path]:
    """failed-list entry shape: (style_name, returncode, output_path)."""
    return (name, rc, Path(f"/tmp/{name}.png"))


def test_exit_code_single_style_success_returns_0():
    """v0.1.x contract: single-style + ok → exit 0."""
    assert _exit_code(is_batch=False, succeeded=[_ok("anime")], failed=[]) == 0


def test_exit_code_single_style_failure_passes_through_returncode():
    """v0.1.x contract: single-style + failure → caller gets mflux's
    own returncode (so scripts can grep by exit code)."""
    assert _exit_code(
        is_batch=False,
        succeeded=[],
        failed=[_fail("anime", 42)],
    ) == 42


def test_exit_code_multi_all_ok_returns_0():
    assert _exit_code(
        is_batch=True,
        succeeded=[_ok("anime"), _ok("ghibli")],
        failed=[],
    ) == 0


def test_exit_code_multi_all_failed_returns_1():
    """All M iterations failed → exit 1 (generic batch failure)."""
    assert _exit_code(
        is_batch=True,
        succeeded=[],
        failed=[_fail("anime", 1), _fail("ghibli", 1)],
    ) == 1


def test_exit_code_multi_partial_returns_5():
    """Mixed batch (some ok, some failed) → exit 5 — distinct from
    user-input 2, missing-tool 3, resource 4 — keeps grep-by-code
    scripting clean for callers."""
    assert _exit_code(
        is_batch=True,
        succeeded=[_ok("anime")],
        failed=[_fail("ghibli", 1)],
    ) == 5


# ── _print_batch_summary ────────────────────────────────────────────────


def test_print_batch_summary_all_ok(capsys):
    _print_batch_summary(
        succeeded=[_ok("anime"), _ok("ghibli")],
        failed=[],
        total=2,
    )
    out = capsys.readouterr().out
    assert "Batch summary" in out
    assert "2 generations" in out
    assert "2 ok" in out
    assert "failed" not in out.lower()


def test_print_batch_summary_all_failed_lists_each(capsys):
    """Every failed style needs to surface (else user can't tell which
    succeeded and which need retry). No 'N ok' line when succeeded is
    empty — the `if succeeded:` guard skips it."""
    _print_batch_summary(
        succeeded=[],
        failed=[_fail("anime", 1), _fail("ghibli", 7)],
        total=2,
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "2 failed" in combined
    assert "anime" in combined and "exit 1" in combined
    assert "ghibli" in combined and "exit 7" in combined
    assert " ok" not in combined


def test_print_batch_summary_mixed(capsys):
    _print_batch_summary(
        succeeded=[_ok("anime")],
        failed=[_fail("ghibli", 1)],
        total=2,
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "1 ok" in combined
    assert "1 failed" in combined
    assert "ghibli" in combined


def test_print_batch_summary_singular_when_total_is_1(capsys):
    """Pedantic but visible in output — 1 generation, not 1 generations."""
    _print_batch_summary(succeeded=[_ok("anime")], failed=[], total=1)
    out = capsys.readouterr().out
    assert "1 generation" in out
    assert "1 generations" not in out


def test_print_batch_summary_plural_when_total_is_3(capsys):
    _print_batch_summary(
        succeeded=[_ok("a"), _ok("b"), _ok("c")],
        failed=[],
        total=3,
    )
    out = capsys.readouterr().out
    assert "3 generations" in out


# ── _preflight_resources ────────────────────────────────────────────────


def _clean_res() -> dict:
    """check_resources() shape with everything green."""
    return {
        "other_mflux_pid": None,
        "ram_ok": True,
        "ram_required_gb": 12,
        "ram_available_gb": 24.0,
        "ram_total_gb": 64.0,
        "disk_ok": True,
        "disk_free_gb": 500.0,
        "battery_ok": True,
        "battery_pct": 100,
    }


@pytest.fixture
def stub_check_resources(monkeypatch):
    """Return a dict the caller can mutate before _preflight_resources
    invokes check_resources. Lets each test stage a specific failure."""
    state = {"res": _clean_res()}

    def fake_check(backend, quant):
        state["last_call"] = (backend, quant)
        return state["res"]

    monkeypatch.setattr(
        "imgen.commands.generate.check_resources", fake_check
    )
    return state


def test_preflight_resources_force_skips_check(monkeypatch):
    """--force bypasses preflight entirely. check_resources MUST NOT be
    called — even reading it would be misleading in a session where
    psutil reports a stale state."""
    called = []
    monkeypatch.setattr(
        "imgen.commands.generate.check_resources",
        lambda b, q: called.append((b, q)) or _clean_res(),
    )

    _preflight_resources(backend="flux", heaviest_quant=8, force=True)

    assert called == [], "force=True must short-circuit before check_resources"


def test_preflight_resources_clean_passes(stub_check_resources):
    """All green → returns None (no SystemExit, no warning crash)."""
    _preflight_resources(backend="flux", heaviest_quant=8, force=False)
    # Confirms helper passed backend + quant through.
    assert stub_check_resources["last_call"] == ("flux", 8)


def test_preflight_resources_other_mflux_running_exits_4(
    stub_check_resources, capsys
):
    stub_check_resources["res"]["other_mflux_pid"] = 12345

    with pytest.raises(SystemExit) as exc_info:
        _preflight_resources(backend="flux", heaviest_quant=8, force=False)

    assert exc_info.value.code == 4
    err = capsys.readouterr().err
    assert "Another mflux process" in err
    assert "12345" in err


def test_preflight_resources_low_ram_exits_4(stub_check_resources, capsys):
    stub_check_resources["res"]["ram_ok"] = False
    stub_check_resources["res"]["ram_available_gb"] = 4.0
    stub_check_resources["res"]["ram_required_gb"] = 24

    with pytest.raises(SystemExit) as exc_info:
        _preflight_resources(backend="flux", heaviest_quant=8, force=False)

    assert exc_info.value.code == 4
    err = capsys.readouterr().err
    assert "Not enough RAM" in err
    # Hint shows the fix menu.
    assert "--preview" in err or "--quantize" in err


def test_preflight_resources_low_disk_warns_not_dies(
    stub_check_resources, capsys
):
    """Disk space is advisory — model download might still fit. Warn,
    don't bail. One call covers both: SystemExit would propagate up out
    of the test function (failing it); warning text shows in capsys."""
    stub_check_resources["res"]["disk_ok"] = False
    stub_check_resources["res"]["disk_free_gb"] = 2.5

    _preflight_resources(backend="flux", heaviest_quant=8, force=False)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "GB disk free" in combined
    assert "imgen clean" in combined


def test_preflight_resources_low_battery_warns_not_dies(
    stub_check_resources, capsys
):
    """Battery low → advisory; user may have a charger nearby."""
    stub_check_resources["res"]["battery_ok"] = False
    stub_check_resources["res"]["battery_pct"] = 12

    _preflight_resources(backend="flux", heaviest_quant=8, force=False)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Battery" in combined and "12%" in combined


# ── _open_results ───────────────────────────────────────────────────────


@pytest.fixture
def stub_subprocess_run(monkeypatch):
    """Record subprocess.run invocations from _open_results without
    actually spawning anything. Tests assert against the recorded list."""
    calls: list[list[str]] = []

    def fake_run(argv, check=False, **kwargs):
        calls.append(list(argv))

        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(
        "imgen.commands.generate.subprocess.run", fake_run
    )
    return calls


def test_open_results_no_open_flag_skips(stub_subprocess_run, tmp_path):
    """--no-open opt-out wins over everything else."""
    img = tmp_path / "out.png"
    img.touch()
    _open_results(
        succeeded=[("anime", img, 1)],
        run_dir=None,
        is_batch=False,
        no_open=True,
    )
    assert stub_subprocess_run == []


def test_open_results_empty_succeeded_skips(stub_subprocess_run):
    """Nothing succeeded → nothing to open."""
    _open_results(
        succeeded=[], run_dir=None, is_batch=False, no_open=False
    )
    assert stub_subprocess_run == []


def test_open_results_multi_opens_run_dir(stub_subprocess_run, tmp_path):
    """Multi-style → user sees the whole batch in Finder at once."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    img = run_dir / "out-anime.png"
    img.touch()

    _open_results(
        succeeded=[("anime", img, 1), ("ghibli", img, 1)],
        run_dir=run_dir,
        is_batch=True,
        no_open=False,
    )

    assert stub_subprocess_run == [["open", str(run_dir)]]


def test_open_results_multi_skips_if_run_dir_missing(
    stub_subprocess_run, tmp_path
):
    """Defence-in-depth: if the dir somehow doesn't exist by now, don't
    let `open` autolaunch some other registered handler for the path."""
    run_dir = tmp_path / "phantom"
    # Note: deliberately NOT created.
    _open_results(
        succeeded=[("anime", tmp_path / "x", 1)],
        run_dir=run_dir,
        is_batch=True,
        no_open=False,
    )
    assert stub_subprocess_run == []


def test_open_results_single_opens_last_file(stub_subprocess_run, tmp_path):
    """v0.2.x behaviour: single-style → open the file in Preview."""
    img = tmp_path / "out.png"
    img.touch()
    _open_results(
        succeeded=[("anime", img, 1)],
        run_dir=None,
        is_batch=False,
        no_open=False,
    )
    assert stub_subprocess_run == [["open", str(img)]]


def test_open_results_unsafe_extension_warns_no_open(
    stub_subprocess_run, tmp_path, capsys
):
    """macOS `open` delegates to the registered app for the suffix; a
    .terminal / .command would auto-execute. The SAFE_OUTPUT_EXTS
    guard rejects anything not in the image whitelist."""
    img = tmp_path / "out.sh"
    img.touch()

    _open_results(
        succeeded=[("anime", img, 1)],
        run_dir=None,
        is_batch=False,
        no_open=False,
    )

    assert stub_subprocess_run == []
    out = capsys.readouterr().out
    assert "unsafe extension" in out
    assert ".sh" in out


def test_open_results_swallows_filenotfound_single(monkeypatch, tmp_path):
    """If `open` binary somehow isn't there (very unusual on macOS),
    don't crash the whole CLI — generation already succeeded.
    Single-style branch."""
    img = tmp_path / "out.png"
    img.touch()

    def raising_run(argv, check=False, **kwargs):
        raise FileNotFoundError("no such binary: open")

    monkeypatch.setattr(
        "imgen.commands.generate.subprocess.run", raising_run
    )

    # Should NOT raise.
    _open_results(
        succeeded=[("anime", img, 1)],
        run_dir=None,
        is_batch=False,
        no_open=False,
    )


def test_open_results_swallows_filenotfound_multi(monkeypatch, tmp_path):
    """Same defence-in-depth for the multi-style (Finder) branch —
    architect NIT-5: the v0.2.3 → v0.2.4 extraction kept the try/except
    in both branches; both deserve coverage."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def raising_run(argv, check=False, **kwargs):
        raise FileNotFoundError("no such binary: open")

    monkeypatch.setattr(
        "imgen.commands.generate.subprocess.run", raising_run
    )

    # Should NOT raise.
    _open_results(
        succeeded=[("anime", run_dir / "x.png", 1)],
        run_dir=run_dir,
        is_batch=True,
        no_open=False,
    )


# ── _run_one_iteration ──────────────────────────────────────────────────


def _full_iter(tmp_path, style="anime") -> Iteration:
    """Iteration with real-ish values for tests against the run loop."""
    return Iteration(
        style_name=style,
        prompt="prompt text",
        negative="",
        final_steps=14,
        final_quantize=8,
        final_guidance=2.5,
        final_strength=0.6,
        output_path=tmp_path / f"out-{style}.png",
        cmd=["/fake/mflux", "--prompt", "x"],
    )


def _full_args(**overrides) -> SimpleNamespace:
    defaults = dict(scope=None, preview=False)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def stub_mflux(monkeypatch):
    """Replace run_with_stderr_redaction with a controllable stub.

    Default returns 0 (success); tests override `state["returncode"]`
    or `state["raise"]` to drive failure / cancel scenarios.

    v0.2.5: parameter renamed log_path → log_file (BinaryIO) to match
    the post-FWD-6 signature. The stub records whether a borrowed fd
    was passed without exercising the actual subprocess stream."""
    state: dict = {"returncode": 0, "raise": None, "calls": []}

    def fake_run(cmd, env, log_file=None):
        state["calls"].append({"cmd": cmd, "env": env, "log_file": log_file})
        if state["raise"] is not None:
            raise state["raise"]
        return state["returncode"]

    monkeypatch.setattr(
        "imgen.commands.generate.run_with_stderr_redaction", fake_run
    )
    return state


_CTX_FIELDS = (
    "backend", "seed", "width", "height", "input_path",
    "effective_custom_prompt", "args", "batch_id", "env",
)


def _run(*, it=None, tmp_path, succeeded, failed, logger=None, **kwargs):
    """Wrapper threading sensible defaults for _run_one_iteration.

    Tests can override any of:
      - top-level helper args (idx, total, is_batch, logger)
      - BatchContext fields (backend, seed, width, height, input_path,
        effective_custom_prompt, args, batch_id, env) — pulled out of
        kwargs and bundled into a BatchContext before the call

    Lets each test stay terse — only set what's relevant."""
    if it is None:
        it = _full_iter(tmp_path)
    ctx_kwargs = dict(
        backend="flux",
        seed=42,
        width=1024,
        height=1024,
        input_path=tmp_path / "in.jpg",
        effective_custom_prompt=None,
        args=_full_args(),
        batch_id=None,
        env={"PATH": "/usr/bin"},
    )
    for field in _CTX_FIELDS:
        if field in kwargs:
            ctx_kwargs[field] = kwargs.pop(field)
    ctx = BatchContext(**ctx_kwargs)

    defaults = dict(
        it=it,
        idx=1,
        total=1,
        is_batch=False,
        ctx=ctx,
        logger=logger,
        succeeded=succeeded,
        failed=failed,
    )
    defaults.update(kwargs)
    return _run_one_iteration(**defaults)


def test_run_one_iteration_success_appends_to_succeeded(
    tmp_state_dir, tmp_path, stub_mflux
):
    succeeded: list = []
    failed: list = []

    cont = _run(
        tmp_path=tmp_path, succeeded=succeeded, failed=failed
    )

    assert cont is True, "successful iteration must return True (continue)"
    assert len(succeeded) == 1
    assert len(failed) == 0
    style_name, output_path, duration = succeeded[0]
    assert style_name == "anime"
    assert output_path == tmp_path / "out-anime.png"
    assert isinstance(duration, int) and duration >= 0


def test_run_one_iteration_failure_appends_to_failed(
    tmp_state_dir, tmp_path, stub_mflux, capsys
):
    """Non-zero returncode → recorded in failed, batch continues (True)."""
    stub_mflux["returncode"] = 7
    succeeded: list = []
    failed: list = []

    cont = _run(
        tmp_path=tmp_path, succeeded=succeeded, failed=failed
    )

    assert cont is True, "failed iteration still returns True (multi continues)"
    assert succeeded == []
    assert len(failed) == 1
    name, rc, path = failed[0]
    assert name == "anime"
    assert rc == 7
    assert path == tmp_path / "out-anime.png"


def test_run_one_iteration_keyboard_interrupt_returns_false(
    tmp_state_dir, tmp_path, stub_mflux, capsys
):
    """Ctrl-C mid-mflux → helper catches KeyboardInterrupt, writes a
    cancel history entry, returns False so cmd_generate early-exits 130."""
    stub_mflux["raise"] = KeyboardInterrupt()
    succeeded: list = []
    failed: list = []

    cont = _run(
        tmp_path=tmp_path, succeeded=succeeded, failed=failed
    )

    assert cont is False
    # Neither list mutated — we exit immediately, not "failed".
    assert succeeded == []
    assert failed == []


def test_run_one_iteration_writes_history_on_success(
    tmp_state_dir, tmp_path, stub_mflux
):
    """append_history fires regardless of outcome — replay needs the
    record to exist so a re-run can pick up the same params."""
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
    )

    entries = load_history()
    assert len(entries) == 1
    e = entries[0]
    assert e["status"] == "success"
    assert e["style"] == "anime"
    assert e["backend"] == "flux"
    assert e["seed"] == 42


def test_run_one_iteration_writes_history_on_failure(
    tmp_state_dir, tmp_path, stub_mflux
):
    stub_mflux["returncode"] = 1
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
    )

    entries = load_history()
    assert entries[0]["status"] == "failed"


def test_run_one_iteration_writes_history_on_cancel(
    tmp_state_dir, tmp_path, stub_mflux
):
    stub_mflux["raise"] = KeyboardInterrupt()
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
    )

    entries = load_history()
    assert len(entries) == 1
    assert entries[0]["status"] == "cancelled"


def test_run_one_iteration_history_style_null_when_custom_prompt(
    tmp_state_dir, tmp_path, stub_mflux
):
    """Custom prompt means replay can't reconstruct from preset — the
    `style` field is None in that case so replay falls back to the
    stored prompt directly."""
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        effective_custom_prompt="my custom",
    )

    e = load_history()[0]
    assert e["style"] is None
    assert e["custom_prompt"] == "my custom"


def test_run_one_iteration_history_batch_fields(
    tmp_state_dir, tmp_path, stub_mflux
):
    """Multi-style runs stamp batch_id + batch_index per entry; single-
    style keeps them None (preserves v0.2.x shape)."""
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        is_batch=True, idx=2, total=3, batch_id="abc123def456",
    )

    e = load_history()[0]
    assert e["batch_id"] == "abc123def456"
    assert e["batch_index"] == "2/3"


def test_run_one_iteration_history_batch_fields_null_when_single(
    tmp_state_dir, tmp_path, stub_mflux
):
    from imgen.history import load_history

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        is_batch=False, batch_id=None,
    )

    e = load_history()[0]
    assert e["batch_id"] is None
    assert e["batch_index"] is None


def test_run_one_iteration_log_markers_written_when_logger_given(
    tmp_state_dir, tmp_path, stub_mflux
):
    """logger set → start + end markers appended via BatchLogger."""
    logger = BatchLogger("testbatch1")

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        logger=logger, is_batch=True, idx=1, total=2,
    )

    assert logger.path.exists()
    content = logger.path.read_bytes().decode()
    assert "[1/2] anime" in content
    # Tightened from " ok " — anchor against the actual marker shape so
    # a stray " ok " elsewhere can't satisfy the assertion.
    assert " → ok in " in content


def test_run_one_iteration_log_markers_record_cancel(
    tmp_state_dir, tmp_path, stub_mflux
):
    stub_mflux["raise"] = KeyboardInterrupt()
    logger = BatchLogger("testbatch2")

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        logger=logger, is_batch=True, idx=1, total=2,
    )

    content = logger.path.read_bytes().decode()
    # Both markers must be present — start was written BEFORE the
    # KeyboardInterrupt; cancel was written AFTER. Locks the ordering
    # so a refactor that moves iteration_start past the try doesn't
    # silently lose the start record. (architect NIT-7)
    assert "[1/2] anime" in content
    assert "CANCELLED" in content
    assert content.index("[1/2] anime") < content.rindex("[1/2] anime")


def test_run_one_iteration_log_skipped_when_logger_none(
    tmp_state_dir, tmp_path, stub_mflux
):
    """logger=None (single-style) → no log file created."""
    _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        logger=None,
    )

    # No log files anywhere (BatchLogger never instantiated → no LOGS_DIR
    # writes; tmp_state_dir LOGS_DIR stays empty).
    import imgen.runs as runs_mod
    assert not runs_mod.LOGS_DIR.exists() or not any(
        runs_mod.LOGS_DIR.glob("*.log")
    )


def test_run_one_iteration_passes_env_to_subprocess(
    tmp_state_dir, tmp_path, stub_mflux
):
    """The minimal env built by cmd_generate must reach mflux verbatim
    — token + COLUMNS/LINES are critical."""
    custom_env = {"PATH": "/usr/bin", "HF_TOKEN": "hf_xxx", "COLUMNS": "120"}

    _run(
        tmp_path=tmp_path, succeeded=[], failed=[], env=custom_env,
    )

    assert stub_mflux["calls"][0]["env"] == custom_env


# ── history-vs-log coherence (v0.2.5 — IMP-2 from v0.2.4 review) ───────


def test_run_one_iteration_log_marker_lands_even_if_history_raises(
    tmp_state_dir, tmp_path, stub_mflux, monkeypatch
):
    """If append_history raises an unexpected exception (today
    unreachable — history.py catches OSError — but defence for future
    cases), the iteration_end log marker MUST still land. Otherwise
    the log would show a start marker with no matching end and the
    next iteration's start marker would look like part of this one.

    The bug shape is: subprocess succeeded → history broken → next
    log entry orphaned. The test stages an exception class that
    history.py wouldn't normally catch (RuntimeError) so we exercise
    the new _safe_append_history wrapper."""
    monkeypatch.setattr(
        "imgen.commands.generate.append_history",
        lambda entry: (_ for _ in ()).throw(RuntimeError("json busted")),
    )
    logger = BatchLogger("coherence1")

    cont = _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        logger=logger, is_batch=True, idx=1, total=2,
    )

    # Iteration still completes — succeeded list gets the entry, batch
    # continues, history error gets a warn().
    assert cont is True
    assert logger.path.exists()
    content = logger.path.read_bytes().decode()
    # Both markers must be present — start before subprocess, end after
    # despite the history-write exception.
    assert "[1/2] anime" in content
    assert " → ok in " in content


def test_run_one_iteration_cancel_marker_lands_even_if_history_raises(
    tmp_state_dir, tmp_path, stub_mflux, monkeypatch
):
    """Same coherence guarantee on the KeyboardInterrupt path: cancel
    marker must land even if the cancel-history record write blew up."""
    stub_mflux["raise"] = KeyboardInterrupt()
    monkeypatch.setattr(
        "imgen.commands.generate.append_history",
        lambda entry: (_ for _ in ()).throw(RuntimeError("json busted")),
    )
    logger = BatchLogger("coherence2")

    cont = _run(
        tmp_path=tmp_path, succeeded=[], failed=[],
        logger=logger, is_batch=True, idx=1, total=2,
    )

    # Caller still gets the early-exit signal (False → cmd_generate
    # returns 130).
    assert cont is False
    content = logger.path.read_bytes().decode()
    assert "[1/2] anime" in content
    assert "CANCELLED" in content


def test_run_one_iteration_warns_on_history_failure(
    tmp_state_dir, tmp_path, stub_mflux, monkeypatch, capsys
):
    """User has to see *something* — log + history mismatch otherwise
    looks like data was never recorded but the user has no clue why."""
    monkeypatch.setattr(
        "imgen.commands.generate.append_history",
        lambda entry: (_ for _ in ()).throw(RuntimeError("json busted")),
    )

    _run(tmp_path=tmp_path, succeeded=[], failed=[])

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "history entry not recorded" in combined
    assert "RuntimeError" in combined
    assert "json busted" in combined


def test_safe_append_history_propagates_keyboard_interrupt(monkeypatch):
    """The broad-except in _safe_append_history catches `Exception` —
    KeyboardInterrupt inherits from BaseException, so it MUST still
    propagate. v0.2.5 review (security NIT-4) flagged this contract
    as untested; lock it.

    Without this, a Ctrl-C delivered exactly while append_history was
    on the stack would be swallowed and the user would see no batch
    cancellation."""
    from imgen.commands.generate import _safe_append_history

    monkeypatch.setattr(
        "imgen.commands.generate.append_history",
        lambda entry: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        _safe_append_history({"k": "v"})


def test_safe_append_history_propagates_system_exit(monkeypatch):
    """SystemExit is also BaseException, also must propagate.
    Same contract as KeyboardInterrupt."""
    from imgen.commands.generate import _safe_append_history

    monkeypatch.setattr(
        "imgen.commands.generate.append_history",
        lambda entry: (_ for _ in ()).throw(SystemExit(99)),
    )

    with pytest.raises(SystemExit) as exc_info:
        _safe_append_history({"k": "v"})
    assert exc_info.value.code == 99

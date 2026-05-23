"""argparse validators — bounded ranges + safe output paths.

These run at parse time (CLI level), so they're the first line of
defense against bad user input. Off-by-one in a range or a missing
extension in the allowlist could land bad values in cmd_generate.
"""
from __future__ import annotations

import argparse

import pytest

from imgen.parser import _float_range, _int_range, _safe_output_path


# ── _int_range ────────────────────────────────────────────────────────

def test_int_range_accepts_in_range():
    v = _int_range(1, 100)
    assert v("50") == 50


@pytest.mark.parametrize("boundary", ["1", "100"])
def test_int_range_accepts_inclusive_boundaries(boundary):
    v = _int_range(1, 100)
    assert v(boundary) == int(boundary)


@pytest.mark.parametrize("bad", ["0", "101", "-1", "1000"])
def test_int_range_rejects_out_of_range(bad):
    v = _int_range(1, 100)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


@pytest.mark.parametrize("bad", ["abc", "1.5", "", "1e2"])
def test_int_range_rejects_non_integer(bad):
    v = _int_range(1, 100)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


# ── _float_range ──────────────────────────────────────────────────────

def test_float_range_accepts_in_range():
    v = _float_range(0.0, 1.0)
    assert v("0.55") == 0.55


@pytest.mark.parametrize("boundary", ["0.0", "1.0"])
def test_float_range_accepts_inclusive_boundaries(boundary):
    v = _float_range(0.0, 1.0)
    assert v(boundary) == float(boundary)


@pytest.mark.parametrize("bad", ["-0.1", "1.1", "2.0", "-1"])
def test_float_range_rejects_out_of_range(bad):
    v = _float_range(0.0, 1.0)
    with pytest.raises(argparse.ArgumentTypeError):
        v(bad)


def test_float_range_rejects_non_float():
    v = _float_range(0.0, 1.0)
    with pytest.raises(argparse.ArgumentTypeError):
        v("not-a-number")


# ── _safe_output_path ─────────────────────────────────────────────────

@pytest.mark.parametrize("good", ["out.png", "out.jpg", "out.jpeg", "out.webp",
                                  "/abs/path/x.PNG", "x.JPEG"])
def test_safe_output_path_accepts_known_image_extensions(good):
    """Allowlist enforced case-insensitively."""
    assert _safe_output_path(good) == good


@pytest.mark.parametrize("bad", [
    "out.terminal",   # macOS would launch Terminal.app
    "out.command",    # macOS would execute as shell
    "out.sh",         # shell script
    "out.app",        # would launch the .app bundle
    "out",            # no extension
    "out.gif",        # not in allowlist
    "out.bmp",
])
def test_safe_output_path_rejects_non_image_extensions(bad):
    """The auto-`open` path would launch the registered app for the
    suffix; restricting to image-only suffixes is defence-in-depth.
    Pins security #8 v0.1.1 fix."""
    with pytest.raises(argparse.ArgumentTypeError):
        _safe_output_path(bad)


# ── --scope default (v0.3.2) ───────────────────────────────────────────


from imgen.parser import build_parser


def test_generate_scope_defaults_to_scene():
    """v0.3.2: ``--scope`` defaults to ``scene`` (was ``None`` in v0.3.1
    and earlier). Most photos colleagues batch are scenes / group shots;
    person-focus is the special case the user opts into explicitly."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg"])
    assert args.scope == "scene"


def test_batch_scope_defaults_to_scene():
    """Same default applies to ``imgen batch <dir>``."""
    parser = build_parser()
    args = parser.parse_args(["batch", "/tmp/dir"])
    assert args.scope == "scene"


def test_generate_scope_person_explicit():
    """Person-focus requires explicit opt-in."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg", "--scope", "person"])
    assert args.scope == "person"


def test_batch_scope_person_explicit():
    parser = build_parser()
    args = parser.parse_args(
        ["batch", "/tmp/dir", "--scope", "person"]
    )
    assert args.scope == "person"


def test_generate_scope_scene_explicit_still_works():
    """Passing --scope scene explicitly resolves to the same default
    (back-compat with users who already typed it before v0.3.2)."""
    parser = build_parser()
    args = parser.parse_args(["generate", "photo.jpg", "--scope", "scene"])
    assert args.scope == "scene"


# ── -v short flag for --version (v0.3.5) ───────────────────────────────


def test_short_v_prints_version_and_exits(capsys):
    """v0.3.5: `imgen -v` mirrors `--version`. node/npm/pip ergonomics
    — every user types `imgen -v` first; previously got "unrecognized
    arguments"."""
    from imgen import __version__

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["-v"])
    # argparse's version action exits 0
    assert exc.value.code == 0
    captured = capsys.readouterr()
    # argparse writes version output to stdout
    assert __version__ in captured.out


def test_short_v_and_long_version_both_print_same(capsys):
    """`-v` and `--version` must produce identical output."""
    from imgen import __version__

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-v"])
    short_out = capsys.readouterr().out

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])
    long_out = capsys.readouterr().out

    assert short_out == long_out
    assert __version__ in short_out


# ── v0.4: --backend choices include user backends from backends.d/ ──────


def test_parser_loads_user_backends_before_choices(tmp_path, monkeypatch):
    """v0.4 design decision 3: --backend choices are loaded at parse
    time via list_backends(), so a TOML in ~/.imgen/backends.d/ shows
    up as a valid --backend argument without code changes.

    Without this, `imgen --backend custom_thing` died with "invalid
    choice" even when the TOML was valid — defeating the registry."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    (backends_dir / "mythical.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", state / "backends.d")
    backends_mod.reset_backends_cache()
    try:
        parser = build_parser()
        args = parser.parse_args(
            ["generate", "photo.jpg", "--backend", "mythical"]
        )
        assert args.backend == "mythical"
    finally:
        backends_mod.reset_backends_cache()


def test_parser_rejects_unknown_backend(tmp_path, monkeypatch):
    """Sanity: a string that doesn't match any built-in or user
    backend still dies with argparse's "invalid choice"."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    (state / "backends.d").mkdir()  # empty
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", state / "backends.d")
    backends_mod.reset_backends_cache()
    try:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["generate", "photo.jpg", "--backend", "totally_unknown_xyz"]
            )
    finally:
        backends_mod.reset_backends_cache()


# ── --list-backends action (v0.4 sibling to --list-styles) ──────────────


def test_list_backends_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--list-backends"])
    assert args.list_backends is True


def test_print_backends_shows_builtins(tmp_path, monkeypatch, capsys):
    """`--list-backends` action lists every backend (built-in + custom)
    with binary, custom marker, and secret marker if declared."""
    from imgen.parser import print_backends
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    (state / "backends.d").mkdir()
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", state / "backends.d")
    backends_mod.reset_backends_cache()
    try:
        rc = print_backends()
        out = capsys.readouterr().out
        assert rc == 0
        assert "flux" in out
        assert "qwen" in out
        assert "mflux-generate-kontext" in out
    finally:
        backends_mod.reset_backends_cache()


def test_print_backends_marks_custom_and_secret(tmp_path, monkeypatch, capsys):
    """User backend with [secret] gets both the `(custom)` marker and
    the `[secret: $ENV_VAR (required)]` suffix."""
    from imgen.parser import print_backends
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    (backends_dir / "fancy.toml").write_text(
        'binary = "fancy-bin"\n'
        'image_flag = "--image-path"\n'
        '\n[secret]\n'
        'env_var = "FANCY_API_KEY"\n'
        'required = true\n'
    )
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", state / "backends.d")
    backends_mod.reset_backends_cache()
    try:
        print_backends()
        out = capsys.readouterr().out
        assert "fancy" in out
        assert "(custom)" in out
        assert "FANCY_API_KEY" in out
        assert "required" in out
    finally:
        backends_mod.reset_backends_cache()


# ── v0.6: --list-loras discovery flag ─────────────────────────────────


def test_list_loras_flag_parsed():
    parser = build_parser()
    args = parser.parse_args(["--list-loras"])
    assert args.list_loras is True


def test_print_loras_shows_built_in_mappings(tmp_path, capsys):
    """`--list-loras` surfaces every built-in style's LoRA mapping with
    repo ref, weight, trigger word, and cache state. Both caches
    pointed at empty tmp dirs so every entry shows 'not downloaded'.

    v0.6.3: six built-in styles ship LoRAs after Phase 1 + Phase 2
    research (anime / anime_alt / pixar / pixar_alt / ghibli / vangogh
    / pencil). simpsons stays text-only. Smoke-test the output's
    structural invariants (ghibli line present, weight precision,
    trigger format, compat-group label) without pinning each row —
    per-style row content is locked in tests/test_styles.py.
    """
    from imgen.parser import print_loras
    # Empty mflux cache too so the cache-state column is deterministic.
    rc = print_loras(hf_cache=tmp_path, mflux_loras_cache=tmp_path / "empty")
    out = capsys.readouterr().out
    assert rc == 0
    # ghibli line was the v0.6.0 surviving built-in and is still shipped.
    assert "openfree/flux-chatgpt-ghibli-lora" in out
    # Weight is shown with 2-decimal precision.
    assert "@0.80" in out
    # Trigger shown for activation discoverability.
    assert "Ghibli style" in out
    # flux-1 compat group shown.
    assert "flux-1" in out


def test_print_loras_lists_text_only_styles_separately(tmp_path, capsys):
    """Styles without `loras` are surfaced as a comma-list under a
    `Text-only styles` section so the user knows what's NOT going to
    download anything."""
    from imgen.parser import print_loras
    print_loras(hf_cache=tmp_path)
    out = capsys.readouterr().out
    assert "Text-only" in out
    for style in ("pencil", "simpsons", "vangogh"):
        assert style in out


def test_print_loras_marks_cached_when_hf_dir_present(tmp_path, capsys):
    """When the HF hub cache contains the LoRA's
    `models--<author>--<name>` directory, the line reads `(cached)`.

    Tests cache-resolution against the standard ``~/.cache/huggingface
    /hub/`` root. Pair test ``..._mflux_cache_dir_present`` covers the
    mflux loras cache root.
    """
    from imgen.parser import print_loras
    cached_dir = tmp_path / "models--openfree--flux-chatgpt-ghibli-lora"
    cached_dir.mkdir()
    # Empty mflux cache → only HF hub cache hit.
    print_loras(hf_cache=tmp_path, mflux_loras_cache=tmp_path / "empty")
    out = capsys.readouterr().out
    ghibli_lines = [
        line for line in out.splitlines()
        if "openfree/flux-chatgpt-ghibli-lora" in line
    ]
    assert ghibli_lines, "expected ghibli LoRA line in --list-loras output"
    assert "(cached)" in ghibli_lines[0]


def test_print_loras_marks_cached_when_mflux_dir_present(tmp_path, capsys):
    """v0.6.4 task #21 regression: when the LoRA weights are in mflux's
    private cache (~/Library/Caches/mflux/loras/) but NOT in the
    standard HF hub cache, --list-loras still reports `(cached)`.

    v0.6.3 only probed the HF hub cache, so every successful smoke-test
    LoRA appeared as "not downloaded" because mflux writes to its own
    cache root. Fix: probe both roots, OR together.
    """
    from imgen.parser import print_loras
    hf_root = tmp_path / "hub"
    hf_root.mkdir()
    mflux_root = tmp_path / "mflux-loras"
    mflux_root.mkdir()
    # LoRA dir present ONLY in the mflux cache, NOT in HF hub.
    (mflux_root / "models--openfree--flux-chatgpt-ghibli-lora").mkdir()
    print_loras(hf_cache=hf_root, mflux_loras_cache=mflux_root)
    out = capsys.readouterr().out
    ghibli_lines = [
        line for line in out.splitlines()
        if "openfree/flux-chatgpt-ghibli-lora" in line
    ]
    assert ghibli_lines, "expected ghibli LoRA line"
    assert "(cached)" in ghibli_lines[0], (
        f"LoRA present in mflux cache but reported not-cached: {ghibli_lines[0]!r}"
    )


def test_print_loras_marks_not_downloaded_when_both_caches_empty(
    tmp_path, capsys,
):
    """v0.6.4 task #21 regression — symmetric to the cached test.
    When NEITHER cache root has the LoRA dir, line reads
    `(not downloaded)`."""
    from imgen.parser import print_loras
    print_loras(
        hf_cache=tmp_path / "empty-hf",
        mflux_loras_cache=tmp_path / "empty-mflux",
    )
    out = capsys.readouterr().out
    ghibli_lines = [
        line for line in out.splitlines()
        if "openfree/flux-chatgpt-ghibli-lora" in line
    ]
    assert ghibli_lines, "expected ghibli LoRA line"
    assert "(not downloaded)" in ghibli_lines[0]


def test_print_loras_marks_local_path_correctly(tmp_path, capsys, monkeypatch):
    """v0.6.x backlog python NIT-1 regression: a LoRA whose `ref` is an
    absolute local path is not in the HF cache layout — its `ref` IS the
    on-disk file location. Before this fix, ``Path(ref).is_dir()``
    returned False (it's a file, not a directory) and the line read
    ``(not downloaded)`` even for files that obviously existed locally.
    The branch now probes ``is_file()`` and prints ``(local)`` or
    ``(missing)``.
    """
    from imgen import parser
    from imgen.styles import LoraRef, Style

    local_safetensors = tmp_path / "my-custom.safetensors"
    local_safetensors.write_bytes(b"\x00\x00\x00\x00")
    fake_lora = LoraRef(
        ref=str(local_safetensors),
        weight=0.7,
        compatible_with=("flux-1",),
        trigger=None,
    )
    monkeypatch.setattr(parser, "list_styles", lambda: ["custom"])
    monkeypatch.setattr(
        parser,
        "get_style",
        lambda _: Style(loras=(fake_lora,)),
    )
    parser.print_loras(hf_cache=tmp_path)
    out = capsys.readouterr().out
    local_lines = [line for line in out.splitlines() if str(local_safetensors) in line]
    assert local_lines, "expected local-path LoRA line in --list-loras output"
    assert "(local)" in local_lines[0]

    # Now point ref at a non-existent file and re-run.
    missing_lora = LoraRef(
        ref=str(tmp_path / "does-not-exist.safetensors"),
        weight=0.7,
        compatible_with=("flux-1",),
        trigger=None,
    )
    monkeypatch.setattr(
        parser,
        "get_style",
        lambda _: Style(loras=(missing_lora,)),
    )
    parser.print_loras(hf_cache=tmp_path)
    out = capsys.readouterr().out
    missing_lines = [
        line for line in out.splitlines() if "does-not-exist" in line
    ]
    assert missing_lines, "expected missing-path LoRA line in --list-loras output"
    assert "(missing)" in missing_lines[0]


def test_print_loras_help_footer_mentions_lora_flag_and_styles_d(tmp_path, capsys):
    """Tail of output points users at the --lora flag and styles.d/
    extension surface — discoverability for both ad-hoc and persistent
    LoRA additions."""
    from imgen.parser import print_loras
    print_loras(hf_cache=tmp_path)
    out = capsys.readouterr().out
    assert "--lora" in out
    assert "--no-lora" in out
    assert "~/.imgen/styles.d/" in out

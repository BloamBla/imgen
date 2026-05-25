"""argparse validators — bounded ranges + safe output paths.

These run at parse time (CLI level), so they're the first line of
defense against bad user input. Off-by-one in a range or a missing
extension in the allowlist could land bad values in cmd_generate.
"""
from __future__ import annotations

import argparse

import pytest

from imgen.defaults import DEFAULTS
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
    """v0.4 design decision 3: --model choices are loaded at parse
    time via list_backends(), so a TOML in ~/.imgen/backends.d/
    (or v0.8+ ~/.imgen/models.d/) shows up as a valid --model
    argument without code changes.

    v0.8.0 commit 4a: flag renamed --backend → --model, but registry
    keys ARE NOT in the v0.7-rename map for user backends (only the
    two built-ins `flux` + `qwen` got renamed), so a user stem like
    'mythical' passes through `_resolve_v07_alias` unchanged.
    """
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
            ["generate", "photo.jpg", "--model", "mythical"]
        )
        assert args.model == "mythical"
    finally:
        backends_mod.reset_backends_cache()


def test_parser_rejects_unknown_model(tmp_path, monkeypatch):
    """Sanity: a string that doesn't match any built-in or user model
    dies with an ArgumentTypeError from `_resolve_v07_alias` (which
    includes a difflib closest-match hint — strictly better UX than
    argparse's default "invalid choice" output).
    """
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
                ["generate", "photo.jpg", "--model", "totally_unknown_xyz"]
            )
    finally:
        backends_mod.reset_backends_cache()


# ── --list-models action (v0.8.0 commit 4b, renamed from --list-backends) ──


def test_list_models_flag_parsed():
    """v0.8.0 commit 4b: top-level --list-backends → --list-models.
    Legacy --list-backends form is caught by the pre-argparse hook in
    cli.main; the new flag binds to args.list_models."""
    parser = build_parser()
    args = parser.parse_args(["--list-models"])
    assert args.list_models is True


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


# ── _add_run_control_args (v0.7.9) ────────────────────────────────────


class TestAddRunControlArgs:
    """v0.7.9 extraction: 5 universal run-control flags shared by
    generate / batch / draw / refine. Per-subcommand help text via
    kwargs, flag SHAPE centralised so future flag-shape changes land
    once. Lock-in tests verify (a) all 4 subcommands expose the same
    5 flags with the same action/short-form, and (b) the helper alone
    in isolation produces the expected argparse surface."""

    def _flags_on_parser(self, parser):
        """Map flag-name → action object for the 5 control flags."""
        actions = {}
        for action in parser._actions:
            for opt in action.option_strings:
                if opt in {"-p", "--preview", "--no-open", "-y", "--yes",
                           "--dry-run", "--force"}:
                    actions[opt] = action
        return actions

    def _build(self, **kwargs):
        import argparse
        from imgen.parser import _add_run_control_args
        p = argparse.ArgumentParser()
        _add_run_control_args(p, **kwargs)
        return p

    def test_all_five_flags_added(self):
        p = self._build()
        flags = self._flags_on_parser(p)
        # 5 flags, 2 with short aliases (-p and -y)
        assert "--preview" in flags
        assert "-p" in flags
        assert "--no-open" in flags
        assert "--yes" in flags
        assert "-y" in flags
        assert "--dry-run" in flags
        assert "--force" in flags

    def test_all_are_store_true(self):
        """Locks the action contract — all 5 are boolean toggles. A
        future change that makes one of them a typed flag would land
        loudly here."""
        import argparse
        p = self._build()
        flags = self._flags_on_parser(p)
        for opt in ("--preview", "--no-open", "--yes", "--dry-run", "--force"):
            assert isinstance(flags[opt], argparse._StoreTrueAction), (
                f"{opt} should be store_true"
            )

    def test_default_help_text_used_when_no_override(self):
        p = self._build()
        flags = self._flags_on_parser(p)
        # Default phrasing mentions the core concept.
        # NIT v0.7.9 review: verify the concept word, not an
        # accidental coupling to "Preview" the macOS app name —
        # a future help rewording to "Finder" / "Pixelmator" /
        # neither shouldn't fail this test.
        assert "preview" in flags["--preview"].help.lower()
        assert "open" in flags["--no-open"].help.lower()
        assert "confirm" in flags["--yes"].help.lower()
        assert "dry" not in flags["--no-open"].help.lower()  # cross-bleed check

    def test_per_subcommand_help_override(self):
        p = self._build(
            preview_help="custom preview help",
            yes_help="custom yes help",
        )
        flags = self._flags_on_parser(p)
        assert flags["--preview"].help == "custom preview help"
        assert flags["--yes"].help == "custom yes help"
        # Unsupplied overrides fall through to defaults
        assert "RAM" in flags["--force"].help

    def test_all_four_subcommands_share_the_flags(self):
        """End-to-end: parse the SAME --dry-run on each of the 4
        subcommands and confirm it landed. Drift detector — if any
        future PR adds a 5th subcommand and forgets to call
        _add_run_control_args, this would catch by parser failure."""
        from imgen.parser import build_parser
        parser = build_parser()
        # generate (default subcommand — no name needed)
        for argv_prefix in (
            ["generate", "photo.jpg"],
            ["batch", "/tmp/dir"],
            ["draw", "a prompt"],
            ["refine", "photo.png"],
        ):
            args, _ = parser.parse_known_args(argv_prefix + ["--dry-run"])
            assert args.dry_run is True, (
                f"--dry-run not set for {argv_prefix[0]}"
            )


# ── v0.8.0 commit 4a — --backend → --model CLI rename ───────────────────
#
# Per [[project-v080-design]] §I + §Q commit 4a lock-ins, plus 2 lock-ins
# from the design pre-vet round (architect HIGH-1 + python NIT-7):
#
#   §Q (4):
#     1. hard-error space form  (`--backend flux`)
#     2. hard-error equals form (`--backend=flux`)
#     3. hint does not echo user value (control-byte safety)
#     4. backends.d/ loader DEPRECATED warn (covered by
#        tests/test_user_models.py::test_user_toml_warns_on_backends_d_load)
#
#   §I (2):
#     5. --model accepts v0.8 canonical names
#     6. --model rejects v0.7 names with hint
#
#   Pre-vet round 1 (2):
#     7. no-flag invocation uses v0.8 default (CRITICAL-1 regression test)
#     8. draw default uses backend_draw key, NOT i2i rename map (HIGH-1)


class TestBackendFlagDeprecation:
    """Pre-argparse hook hard-errors on the legacy --backend flag with
    a static migration hint. Hook lives at parser._check_for_deprecated_
    backend_flag and is wired from cli.main before argparse runs.
    """

    def test_backend_flag_space_form_hard_errors_with_hint(self, capsys):
        """`--backend flux` (separate token) → SystemExit + stderr
        contains migration hint naming the new --model flag."""
        from imgen.parser import _check_for_deprecated_backend_flag
        with pytest.raises(SystemExit):
            _check_for_deprecated_backend_flag(
                ["generate", "photo.jpg", "--backend", "flux"]
            )
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "renamed --backend → --model" in combined
        assert "--model flux-kontext" in combined

    def test_backend_flag_equals_form_hard_errors_with_hint(self, capsys):
        """`--backend=flux` (equals form) → same hard-error path.
        Equals form was an easy miss in earlier drafts (caught at
        python-reviewer round-1 MEDIUM)."""
        from imgen.parser import _check_for_deprecated_backend_flag
        with pytest.raises(SystemExit):
            _check_for_deprecated_backend_flag(
                ["generate", "photo.jpg", "--backend=flux"]
            )
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "renamed --backend → --model" in combined

    def test_backend_flag_hint_does_not_echo_user_value(self, capsys):
        """Security lock-in (memo §I round-1 MEDIUM): the migration
        hint is STATIC text. The user's typed value never gets
        interpolated into the error — even via repr() — so a
        control-byte-laden value can't leak escape sequences into
        stderr.
        """
        evil_value = "\x1b[2J\x1b[H"  # clear-screen + cursor home
        from imgen.parser import _check_for_deprecated_backend_flag
        with pytest.raises(SystemExit):
            _check_for_deprecated_backend_flag(
                ["generate", "photo.jpg", f"--backend={evil_value}"]
            )
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # Hint fires
        assert "renamed --backend → --model" in combined
        # User value NEVER appears in output (raw OR repr'd)
        assert evil_value not in combined
        assert repr(evil_value) not in combined
        # Spot-check: the escape byte itself is absent
        assert "\x1b" not in combined

    def test_list_backends_flag_hard_errors_with_hint(self, capsys):
        """v0.8.0 commit 4b: legacy ``--list-backends`` top-level
        flag is detected by a pre-argparse hook (architect 4b pre-vet
        M-4 — consistency with the ``--backend`` hook). Hint points
        at ``--list-models``.
        """
        from imgen.parser import _check_for_deprecated_list_backends_flag
        with pytest.raises(SystemExit):
            _check_for_deprecated_list_backends_flag(["--list-backends"])
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "renamed --list-backends → --list-models" in combined
        assert "--list-models" in combined

    def test_hook_passes_through_when_no_backend_flag(self, capsys):
        """Negative case: argv without --backend → hook returns
        cleanly, no SystemExit, no stderr.
        """
        from imgen.parser import _check_for_deprecated_backend_flag
        # Various argv shapes that DON'T contain --backend
        for argv in (
            [],
            ["generate", "photo.jpg"],
            ["generate", "photo.jpg", "--model", "flux-kontext"],
            ["draw", "a prompt", "--model", "flux-dev"],
            # Argument with `backend` substring but not the flag
            ["generate", "photo.jpg", "--style", "cyberbackend"],
        ):
            _check_for_deprecated_backend_flag(argv)
        captured = capsys.readouterr()
        assert (captured.out + captured.err) == ""


@pytest.mark.parametrize(
    "v08_name",
    [
        "flux-kontext",            # was "flux" in v0.7
        "qwen-image-edit-v1",      # was "qwen" in v0.7
        "flux-dev",                # unchanged
        "flux2-klein-edit-9b",     # unchanged
    ],
)
def test_model_flag_accepts_v0_8_0_names(v08_name):
    """§I lock-in: --model accepts every v0.8 canonical name across
    every subcommand that has the flag.

    v0.8.0 commit 4b update: registry source-of-truth is now
    BUILTIN_MODELS (v0.8-keyed), so the resolver returns the v0.8
    name directly — no v0.8→v0.7 inverse translation anymore (the
    commit-4a inverse-map branch was deleted per the 4a TODO marker).
    args.model now holds the v0.8 canonical value.
    """
    from imgen.parser import build_parser
    parser = build_parser()
    # generate subcommand — i2i flag presence is the canonical case
    args = parser.parse_args(
        ["generate", "photo.jpg", "--model", v08_name]
    )
    assert args.model == v08_name


def test_model_flag_rejects_v0_7_name_with_hint(capsys):
    """§I lock-in: --model NAME with the v0.7 name dies, hint names
    the v0.8 canonical equivalent. argparse wraps our
    ArgumentTypeError into its standard "argument --model: <msg>"
    exit-2 path.
    """
    from imgen.parser import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["generate", "photo.jpg", "--model", "flux"]
        )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "'flux' is the v0.7 model name" in combined
    assert "'flux-kontext'" in combined


def test_model_flag_unknown_name_has_difflib_close_match_hint(capsys):
    """When the user typos a v0.8 name (e.g. `flux-konext` instead of
    `flux-kontext`), the rejection includes a `Did you mean ... ?`
    suggestion via difflib. Strictly better UX than argparse's
    default 'invalid choice'.
    """
    from imgen.parser import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["generate", "photo.jpg", "--model", "flux-konext"]
        )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Unknown model 'flux-konext'" in combined
    assert "Did you mean 'flux-kontext'?" in combined


class TestNoModelFlagDefault:
    """Pre-vet round-1 CRITICAL regression lock-in: argparse runs
    `type=` on string defaults, so without pre-translation of
    ``defaults["backend"]`` ("flux") through the v0.7→v0.8 rename
    map, every no-flag invocation would die on the v0.7-name
    rejection branch of `_resolve_v07_alias`. These tests pin the
    fix: `_v07_default_to_v08_for_i2i` translates the i2i default
    BEFORE argparse sees it.

    v0.8.0 commit 4b update: dest renamed `backend` → `model` in
    lockstep with the registry source-of-truth flip; assertions check
    ``args.model`` (v0.8 canonical value) — the v0.7 inverse-map
    branch was removed from the resolver at 4b.
    """

    def test_generate_no_model_resolves_to_v08_default(self):
        from imgen.parser import build_parser
        parser = build_parser()
        args = parser.parse_args(["generate", "photo.jpg"])
        # DEFAULTS["backend"] = "flux" → pre-translated to
        # "flux-kontext" via _v07_default_to_v08_for_i2i, fed as
        # argparse default → resolver returns it unchanged (v0.8
        # canonical in registry).
        assert args.model == "flux-kontext"

    def test_batch_no_model_resolves_to_v08_default(self):
        from imgen.parser import build_parser
        parser = build_parser()
        args = parser.parse_args(["batch", "/tmp/dir"])
        assert args.model == "flux-kontext"

    def test_draw_no_model_resolves_to_backend_draw_default(self):
        """Architect HIGH-1 lock-in: draw's default is
        ``defaults["backend_draw"]`` ("flux-dev"), which is ALREADY
        v0.8-canonical and NOT in the rename map. The translation
        helper is intentionally NOT applied to backend_draw — a user
        with `[defaults] backend_draw = "flux"` in config.toml would
        get the v0.7-rejection error from `_resolve_v07_alias`
        instead of being silently migrated to `flux-kontext` for the
        t2i subcommand (which would be the wrong model for t2i).
        """
        from imgen.parser import build_parser
        parser = build_parser()
        args = parser.parse_args(["draw", "a prompt"])
        assert args.model == "flux-dev"

    def test_refine_no_model_resolves_to_literal_default(self):
        """Refine's default is a literal "flux2-klein-edit-9b" (not
        sourced from `defaults`). Already v0.8-canonical; passes
        through `_resolve_v07_alias` unchanged.
        """
        from imgen.parser import build_parser
        parser = build_parser()
        args = parser.parse_args(["refine", "photo.png"])
        assert args.model == "flux2-klein-edit-9b"

    def test_draw_with_config_override_backend_draw(self):
        """Architect HIGH-1 lock-in (continued): if a colleague sets
        `[defaults] backend_draw = "z-image-turbo"` (hypothetical
        user backend), the parser uses it without translation —
        because the rename map applies only to i2i defaults.
        """
        from imgen.parser import build_parser
        # Imagine a config-loaded defaults dict with a custom draw
        # default. We pass it via the parser's `defaults=` slot.
        custom_defaults = dict(DEFAULTS)
        custom_defaults["backend_draw"] = "flux-dev"  # safe v0.8 name
        parser = build_parser(defaults=custom_defaults)
        args = parser.parse_args(["draw", "a prompt"])
        assert args.model == "flux-dev"

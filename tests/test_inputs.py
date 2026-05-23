"""Tests for src/imgen/inputs.py — v0.3.0 input-side helpers.

Three surfaces, all pure-ish (subprocess is shimmed in HEIC tests):

  * ``discover_inputs(directory)`` — non-recursive directory listing
    filtered to ``SUPPORTED_INPUT_EXTS``, dotfiles skipped, alphabetical.
    Dies (SystemExit 2) on non-existent / not-a-dir; returns ``[]`` on
    empty/no-matches so the caller chooses whether "0 images" is fatal
    in context (cmd_batch will die; future callers may not).
  * ``check_input_stems(paths)`` — die on any stem collision (e.g.
    ``IMG_1234.heic`` + ``IMG_1234.jpg``) because the flat output layout
    ``<run_dir>/<stem>-<style>.png`` would overwrite. All offenders
    surfaced in one shot for one-touch user fix.
  * ``needs_jpeg_conversion`` / ``convert_heic_to_jpeg`` /
    ``resolve_to_mflux_input`` — HEIC support via macOS-native ``sips``.
    mflux subprocess can't auto-register pillow-heif (verified
    2026-05-22 against pillow-heif GH README); converting at the imgen
    layer is the only fragile-free path. The same helper plugs into
    ``cmd_generate`` for the single-input HEIC bug-fix bonus.

Strict TDD per project CLAUDE.md matrix — all functions here are pure
or subprocess-shimmable, so failing test → minimal impl → green is the
contract. No real ``sips`` invocation in the suite.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imgen.inputs import (
    HEIC_EXTS,
    SUPPORTED_INPUT_EXTS,
    check_input_stems,
    convert_heic_to_jpeg,
    discover_inputs,
    needs_jpeg_conversion,
    resolve_single_input_path,
    resolve_to_mflux_input,
)


# ── discover_inputs ─────────────────────────────────────────────────────


def test_discover_inputs_non_existent_dies(tmp_path):
    """A typo'd / missing path is a user-error class (exit 2) — exits
    before any sips invocation or run-dir mkdir."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit) as exc:
        discover_inputs(missing)
    assert exc.value.code == 2


def test_discover_inputs_not_a_dir_dies(tmp_path):
    """`imgen batch <file>` (user typed a file path by mistake) must die
    cleanly — not silently treat a single file as a one-image batch."""
    f = tmp_path / "single.jpg"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        discover_inputs(f)
    assert exc.value.code == 2


def test_discover_inputs_empty_dir_returns_empty(tmp_path):
    """Empty dir returns []. cmd_batch surfaces the user-facing
    "0 supported images" error — keeping the helper non-fatal here lets
    future callers (e.g. doctor / dry-run preflight) detect empty without
    catching SystemExit."""
    assert discover_inputs(tmp_path) == []


def test_discover_inputs_only_non_images_returns_empty(tmp_path):
    """Non-image files (.txt/.md/.zip) silently filtered. Caller turns
    the empty list into a user-friendly "0 supported images" error."""
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "data.zip").write_bytes(b"x")
    (tmp_path / "doc.md").write_text("x")
    assert discover_inputs(tmp_path) == []


def test_discover_inputs_returns_supported_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.heic").write_bytes(b"x")
    (tmp_path / "c.png").write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert {p.name for p in result} == {"a.jpg", "b.heic", "c.png"}


def test_discover_inputs_sorted_alphabetical(tmp_path):
    """Deterministic order — log section indices [1/N], [2/N]... must
    refer to the same input across reruns. Sort by name (not by mtime,
    which is filesystem-dependent and would shuffle batch order after
    `cp -p`)."""
    for name in ("c.jpg", "a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert [p.name for p in result] == ["a.jpg", "b.jpg", "c.jpg"]


def test_discover_inputs_suffix_case_insensitive(tmp_path):
    """macOS HFS+/APFS preserve case but `IMG_1234.HEIC` (camera-export
    convention) and `photo.Png` (some editors) must match the same set
    as `.heic` / `.png`. Lower the suffix before checking."""
    (tmp_path / "PHOTO.JPG").write_bytes(b"x")
    (tmp_path / "IMG.HEIC").write_bytes(b"x")
    (tmp_path / "raw.Png").write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert {p.name for p in result} == {"PHOTO.JPG", "IMG.HEIC", "raw.Png"}


def test_discover_inputs_skips_dotfiles(tmp_path):
    """`.DS_Store` is the chief offender — macOS Finder spawns one in
    every dir it touches, and it'd otherwise need explicit filtering.
    Drop everything starting with `.` to also catch user-hidden files
    that they explicitly don't want batched."""
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")
    (tmp_path / "visible.jpg").write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert [p.name for p in result] == ["visible.jpg"]


def test_discover_inputs_non_recursive(tmp_path):
    """v0.3.0 design decision: non-recursive. Predictable "what you see
    is what you batch". User with a nested tree flattens via shell.
    Tests that a supported file in a subdir is NOT picked up."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.jpg").write_bytes(b"x")
    (tmp_path / "top.jpg").write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert [p.name for p in result] == ["top.jpg"]


def test_discover_inputs_skips_subdir_with_image_suffix(tmp_path):
    """macOS `.photoslibrary` packages are directories with a suffix —
    a naive `glob("*.jpg")` would treat any dir named `foo.jpg` as a
    file. Filter to ``is_file()`` so packages, mounted volumes, etc.
    don't leak through."""
    weird = tmp_path / "looks.jpg"
    weird.mkdir()
    (tmp_path / "real.jpg").write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert [p.name for p in result] == ["real.jpg"]


def test_discover_inputs_supported_extensions_set():
    """Lock the supported set — adding or removing an ext is a v0.3.x
    user-visible change. v0.3.0 ships: jpg, jpeg, png, webp, heic, heif,
    bmp, tif, tiff, gif. AVIF + RAW deferred per design doc."""
    assert SUPPORTED_INPUT_EXTS == frozenset({
        ".jpg", ".jpeg", ".png", ".webp",
        ".heic", ".heif",
        ".bmp", ".tif", ".tiff", ".gif",
    })


def test_discover_inputs_finds_every_supported_extension(tmp_path):
    """End-to-end check that every entry in SUPPORTED_INPUT_EXTS is
    actually accepted by discover_inputs — guards against drift between
    the constant and the filter predicate."""
    names = [f"x{ext}" for ext in sorted(SUPPORTED_INPUT_EXTS)]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    result = discover_inputs(tmp_path)
    assert {p.name for p in result} == set(names)


# ── Control-byte filter on input filenames (v0.3.0 security IMP-3) ──────


@pytest.mark.parametrize("bad_name", [
    "evil\x1b[2J.jpg",       # ANSI clear-screen via ESC + CSI
    "alert\x07.png",          # BEL
    "with\nnewline.jpg",      # newline — log marker injection
    "del\x7fbyte.jpg",        # DEL
    "csi\x9bclear.jpg",       # bare CSI (C1) — clears screen on
                              # 8-bit ECMA-48 terminals
    "win\x9dtitle.jpg",       # OSC C1
    # NUL (0x00) is rejected at the POSIX layer — the kernel won't
    # let us write the file. The predicate guards against it
    # anyway in case a NUL arrives via a non-FS path (UNC mount,
    # crafted dirent injection), but it's untestable at the FS layer.
])
def test_discover_inputs_skips_control_bytes(tmp_path, bad_name, capsys):
    """C0/DEL/C1 in input filenames → warn + skip. Without this filter,
    BatchLogger.write_header / input_section_* and the confirm-gate
    print() would emit raw escapes into the log + terminal, allowing
    fake marker injection and screen-clearing. Mirrors the styles.py
    _is_safe_stem treatment for user-style filenames (v0.2.5)."""
    (tmp_path / bad_name).write_bytes(b"")
    (tmp_path / "ok.jpg").write_bytes(b"")
    result = discover_inputs(tmp_path)
    assert {p.name for p in result} == {"ok.jpg"}
    # Warn surfaced. capsys captures stdout (warn() in this project
    # prints to stdout, not stderr).
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "control bytes" in combined.lower()


def test_discover_inputs_keeps_unicode_stems(tmp_path):
    """Legitimate non-ASCII (CJK, Cyrillic, emoji, accented Latin) all
    live above U+009F, so the filter must NOT reject them. Real users
    name photos like that — `vacaciones_París.jpg`, `寒假.heic`."""
    for name in ("vacaciones_París.jpg", "寒假.heic", "🌅sunset.png", "São_Paulo.webp"):
        (tmp_path / name).write_bytes(b"")
    result = discover_inputs(tmp_path)
    assert len(result) == 4


def test_discover_inputs_warn_uses_repr_not_raw_name(tmp_path, capsys):
    """The warn message itself must not propagate the escape into the
    user's terminal — otherwise the filter half-works (the file is
    skipped but the screen still gets cleared at warn time). Show repr
    of the name so escapes render as `\\x1b` literal."""
    (tmp_path / "x\x1b[2Jbad.jpg").write_bytes(b"")
    discover_inputs(tmp_path)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # The literal escape byte itself should not appear in the
    # captured output — repr() escapes it as \x1b.
    assert "\x1b" not in combined
    assert "\\x1b" in combined


# ── check_input_stems ───────────────────────────────────────────────────


def test_check_input_stems_no_collisions_ok():
    """Standard iPhone HEIC names + jpgs from different sources — all
    unique stems, no death, no side effects."""
    paths = [Path("/p/IMG_1234.heic"), Path("/p/IMG_5678.heic"),
             Path("/p/vacation.jpg"), Path("/p/c.png")]
    check_input_stems(paths)  # must not raise


def test_check_input_stems_two_collisions_die():
    """Two-extension collision is the canonical failure mode — user
    copied `IMG_1234.heic` from Photos and a `.jpg` export of the same
    photo into one folder, gets a clear "rename one" hint."""
    paths = [Path("/p/IMG_1234.heic"), Path("/p/IMG_1234.jpg")]
    with pytest.raises(SystemExit) as exc:
        check_input_stems(paths)
    assert exc.value.code == 2


def test_check_input_stems_three_offenders_named(capsys):
    """Surface every offending stem in one error so the user can fix
    them all in one pass — no game of whack-a-mole where the first run
    reports stem A, the second stem B, etc."""
    paths = [
        Path("/p/IMG_1.heic"), Path("/p/IMG_1.jpg"),
        Path("/p/IMG_2.png"),  Path("/p/IMG_2.webp"),
        Path("/p/IMG_3.jpg"),  Path("/p/IMG_3.heif"),
    ]
    with pytest.raises(SystemExit):
        check_input_stems(paths)
    captured = capsys.readouterr()
    assert "IMG_1" in captured.err
    assert "IMG_2" in captured.err
    assert "IMG_3" in captured.err


def test_check_input_stems_empty_ok():
    """An empty list (e.g. caller about to die on "0 supported images")
    must not itself crash on `collections.Counter([]).most_common()`."""
    check_input_stems([])  # no raise


def test_check_input_stems_triple_collision_one_stem(capsys):
    """Three files with the same stem — uncommon but possible when
    HEIC + JPEG + PNG of the same source coexist. Stem named once in
    the error (not 3×), so the message stays short."""
    paths = [
        Path("/p/X.heic"), Path("/p/X.jpg"), Path("/p/X.png"),
    ]
    with pytest.raises(SystemExit):
        check_input_stems(paths)
    err_text = capsys.readouterr().err
    assert err_text.count("'X'") == 1, (
        f"expected the colliding stem to be named exactly once; got: {err_text}"
    )


# ── HEIC: needs_jpeg_conversion ─────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "x.heic", "x.heif", "X.HEIC", "X.HEIF",
    "weird.Heic", "/full/path/photo.heic",
])
def test_needs_jpeg_conversion_heic_variants(name):
    assert needs_jpeg_conversion(Path(name))


@pytest.mark.parametrize("name", [
    "x.jpg", "x.jpeg", "x.png", "x.webp",
    "x.bmp", "x.tif", "x.tiff", "x.gif",
    "x",  # no suffix
])
def test_needs_jpeg_conversion_non_heic_passes_through(name):
    assert not needs_jpeg_conversion(Path(name))


def test_heic_exts_constant_locked():
    """HEIC_EXTS exposed so test_inputs (and any future caller that wants
    to UI-list which inputs require conversion) can introspect."""
    assert HEIC_EXTS == frozenset({".heic", ".heif"})


# ── HEIC: convert_heic_to_jpeg ──────────────────────────────────────────


def test_convert_heic_to_jpeg_invokes_sips_with_correct_argv(
    monkeypatch, tmp_path,
):
    """Argv shape is locked by mflux's downstream PIL expectation: sips
    output must be a real JPEG (not just renamed HEIC). `-s format jpeg`
    is the documented sips invocation. List form, never shell=True,
    matches project subprocess discipline.

    v0.3.0 review additions: ``timeout=60`` (hang guard on corrupt HEIC)
    and explicit ``env=`` minimal allow-list (don't inherit HF_TOKEN).
    """
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        # sips would have written the output; fake it so callers
        # depending on Path.exists() see the file
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"fakejpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "in.heic"
    dst = tmp_path / "out.jpg"
    src.write_bytes(b"heic-bytes")
    convert_heic_to_jpeg(src, dst)
    assert captured["cmd"] == [
        "sips", "-s", "format", "jpeg", str(src), "--out", str(dst),
    ]
    assert captured["kwargs"].get("check") is True
    assert captured["kwargs"].get("capture_output") is True
    assert captured["kwargs"].get("timeout") == 60


def test_convert_heic_to_jpeg_does_not_leak_hf_token_to_sips(
    monkeypatch, tmp_path,
):
    """v0.3.0 security review: sips inherits parent env by default,
    which would forward HF_TOKEN to a binary that has no business
    with it (and could surface in a sips crash report under
    ~/Library/Logs/DiagnosticReports/). Enforce a minimal env
    allow-list — HF_TOKEN never reaches sips."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak_here_1234567890")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_also_not_here")
    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "x.heic"
    dst = tmp_path / "out.jpg"
    src.write_bytes(b"")
    convert_heic_to_jpeg(src, dst)
    env = captured["env"]
    assert env is not None, "env must be explicitly passed, not None (inherit)"
    assert "HF_TOKEN" not in env
    assert "HUGGINGFACE_HUB_TOKEN" not in env
    # And the allow-list is non-empty when those vars exist.
    assert "PATH" in env  # PATH always set in test runs


def test_convert_heic_to_jpeg_propagates_timeout(monkeypatch, tmp_path):
    """sips stalled (corrupt HEIC) → TimeoutExpired propagates out so the
    caller / batch loop sees an explicit failure instead of an opaque
    hang at the per-iteration BatchLogger gap."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=60)

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "huge.heic"
    dst = tmp_path / "out.jpg"
    src.write_bytes(b"")
    with pytest.raises(subprocess.TimeoutExpired):
        convert_heic_to_jpeg(src, dst)


def test_convert_heic_to_jpeg_raises_on_sips_failure(monkeypatch, tmp_path):
    """sips fails (corrupted HEIC, unsupported format) → CalledProcessError
    propagates so caller can warn-and-skip. We never swallow here — caller
    layer decides whether the whole batch fails or this input is dropped."""

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, output=b"", stderr=b"sips: invalid input",
        )

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "bad.heic"
    dst = tmp_path / "out.jpg"
    src.write_bytes(b"")
    with pytest.raises(subprocess.CalledProcessError):
        convert_heic_to_jpeg(src, dst)


# ── HEIC: resolve_to_mflux_input ────────────────────────────────────────


def test_resolve_to_mflux_input_heic_converts_to_cache(monkeypatch, tmp_path):
    """HEIC input → cache_dir/<stem>.jpg, sips invoked once. Stem is
    preserved (without the .heic suffix) so cmd_generate's output naming
    keeps the original-file identity."""
    cache_dir = tmp_path / "cache"
    runs: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"jpeg")
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "vacation.heic"
    src.write_bytes(b"heic")
    out = resolve_to_mflux_input(src, cache_dir)
    assert out == cache_dir / "vacation.jpg"
    assert out.exists()
    assert len(runs) == 1


def test_resolve_to_mflux_input_non_heic_returns_original(tmp_path):
    """JPEG / PNG / WEBP passed through verbatim — no sips, no cache_dir
    touched. Avoids the 100-200ms per-image cost when nothing needs it."""
    cache_dir = tmp_path / "cache"
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"jpeg")
    out = resolve_to_mflux_input(src, cache_dir)
    assert out == src
    assert not cache_dir.exists()


def test_resolve_to_mflux_input_creates_cache_dir_0o700(monkeypatch, tmp_path):
    """Cache dir is created with 0o700 (matches ~/.imgen mode). The
    HEIC's intermediate JPEG can contain identifiable subject matter —
    no point exposing it to other users on a shared Mac."""
    cache_dir = tmp_path / "fresh_cache"

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "x.heic"
    src.write_bytes(b"")
    resolve_to_mflux_input(src, cache_dir)
    assert cache_dir.is_dir()
    assert (cache_dir.stat().st_mode & 0o777) == 0o700


def test_resolve_to_mflux_input_reuses_existing_cache_dir(
    monkeypatch, tmp_path,
):
    """Caller passes a TemporaryDirectory path that already exists.
    mkdir must be parents=True, exist_ok=True so a second input in
    the same batch doesn't trip OSError."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(mode=0o700)

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "a.heic"
    src.write_bytes(b"")
    resolve_to_mflux_input(src, cache_dir)  # must not raise


def test_resolve_to_mflux_input_heif_also_converts(monkeypatch, tmp_path):
    """`.heif` is the same container as `.heic` in iOS — must trigger
    the same conversion path. Some Android exports use `.heif`."""
    cache_dir = tmp_path / "cache"

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    src = tmp_path / "shot.heif"
    src.write_bytes(b"")
    out = resolve_to_mflux_input(src, cache_dir)
    assert out == cache_dir / "shot.jpg"


# ── resolve_single_input_path (v0.7.7 Sec #S2) ────────────────────────


class TestResolveSingleInputPath:
    """v0.7.7 Sec #S2: shared validator for cmd_generate + cmd_refine
    single-file input paths. Mirrors batch's discover_inputs filter
    so the cross-cutting control-byte guard is uniform across the
    three i2i subcommands."""

    def test_clean_path_returns_resolved(self, tmp_path):
        p = tmp_path / "photo.png"
        p.write_bytes(b"x")
        result = resolve_single_input_path(str(p), subcommand="refine")
        assert result == p.resolve()

    def test_missing_dies_code_2(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            resolve_single_input_path(
                str(tmp_path / "missing.png"), subcommand="refine",
            )
        assert exc.value.code == 2

    def test_directory_dies_code_2(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            resolve_single_input_path(str(tmp_path), subcommand="generate")
        assert exc.value.code == 2

    def test_control_byte_in_filename_dies_code_2(self, tmp_path):
        """ESC C0 byte in the filename → reject, do NOT proceed even
        if the file exists. Same threat model as discover_inputs's
        warn-and-skip for batch."""
        unsafe = tmp_path / "photo\x1b[2J.png"
        unsafe.write_bytes(b"x")
        with pytest.raises(SystemExit) as exc:
            resolve_single_input_path(str(unsafe), subcommand="refine")
        assert exc.value.code == 2

    def test_del_byte_in_filename_dies(self, tmp_path):
        """DEL (0x7f) is on the same reject list as C0/C1."""
        unsafe = tmp_path / "photo\x7f.png"
        unsafe.write_bytes(b"x")
        with pytest.raises(SystemExit) as exc:
            resolve_single_input_path(str(unsafe), subcommand="generate")
        assert exc.value.code == 2

    def test_c1_byte_in_filename_dies(self, tmp_path):
        """C1 controls (0x80-0x9f) include CSI (0x9b) — reject."""
        unsafe = tmp_path / "photo\x9b.png"
        unsafe.write_bytes(b"x")
        with pytest.raises(SystemExit) as exc:
            resolve_single_input_path(str(unsafe), subcommand="refine")
        assert exc.value.code == 2

    def test_unicode_emoji_passes_through(self, tmp_path):
        """Emoji / accented Latin / CJK are above U+009F and are
        legitimate input. Don't reject them."""
        ok = tmp_path / "Привет 🎨 写真.png"
        ok.write_bytes(b"x")
        result = resolve_single_input_path(str(ok), subcommand="refine")
        assert result == ok.resolve()

    def test_subcommand_label_in_error(self, tmp_path, capsys):
        """The 'subcommand' kwarg scopes the diagnostic so users see
        the verb they typed."""
        with pytest.raises(SystemExit):
            resolve_single_input_path(
                str(tmp_path / "x.png"), subcommand="refine",
            )
        err = capsys.readouterr()
        assert "refine:" in (err.out + err.err)

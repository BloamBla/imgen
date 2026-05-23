"""Input-side helpers for v0.3.0 ``imgen batch <dir>`` (and a bonus
HEIC fix-up for ``imgen generate <file>``).

Three small surfaces:

* :func:`discover_inputs` — non-recursive directory scan filtered to
  :data:`SUPPORTED_INPUT_EXTS`. Dies (SystemExit 2) on non-existent /
  not-a-dir; returns ``[]`` on "empty / no supported files" so callers
  decide whether that's fatal in context. Skips dotfiles (``.DS_Store``
  et al.) and subdirectories (including ``.photoslibrary`` packages
  that look file-shaped to a naive ``glob``).
* :func:`check_input_stems` — fail-fast collision detector. Because
  the v0.3.0 output layout is flat (``<run_dir>/<stem>-<style>.png``),
  two inputs sharing a stem would silently overwrite each other.
  Reports every offending stem in one error so the user fixes the
  whole thing in one pass.
* :func:`resolve_to_mflux_input` / :func:`needs_jpeg_conversion` /
  :func:`convert_heic_to_jpeg` — sips-based HEIC→JPEG conversion at the
  imgen layer. pillow-heif can't be auto-registered inside the mflux
  child process (verified against pillow-heif's GH README 2026-05-22),
  so we pre-convert and hand mflux a path it can ``PIL.Image.open``
  natively. Same helper plugs into ``cmd_generate``'s single-input path
  to fix the v0.2.x cryptic-error-on-HEIC bug for free.
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path

from .colors import die, warn

__all__ = [
    "HEIC_EXTS",
    "SUPPORTED_INPUT_EXTS",
    "check_input_stems",
    "convert_heic_to_jpeg",
    "discover_inputs",
    "needs_jpeg_conversion",
    "resolve_single_input_path",
    "resolve_to_mflux_input",
]


# ── Supported input formats ─────────────────────────────────────────────

# AVIF + RAW (CR2/NEF/ARW) deliberately deferred per v0.3.0 design doc:
# RAW needs rawpy (heavy dep), AVIF is rare on macOS Photos exports.
SUPPORTED_INPUT_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp",
    ".heic", ".heif",
    ".bmp", ".tif", ".tiff", ".gif",
})

# HEIC variants — input via sips, output JPEG that mflux's PIL can open
# without any subprocess-side plugin registration.
HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif"})


# ── discover_inputs ─────────────────────────────────────────────────────


def _has_unsafe_controls(name: str) -> bool:
    """True if ``name`` contains C0, DEL, or C1 control bytes.

    Mirrors :func:`imgen.styles._is_safe_stem` but checks the whole
    filename (not just the stem) because input filenames reach the
    terminal AND log via three surfaces:

      * :meth:`~imgen.runs.BatchLogger.write_header` renders the
        ``# inputs (N):  ...`` line with raw ``p.name``.
      * :meth:`~imgen.runs.BatchLogger.input_section_start` /
        :meth:`~imgen.runs.BatchLogger.input_section_end` markers.
      * :func:`~imgen.commands.batch._confirm_dir_batch` prints
        ``p.name`` to the terminal before the run.

    Without this filter, an attacker (or accidental rename) producing
    ``IMG\\x1b[2J.jpg`` could clear-screen the user's terminal at the
    confirm prompt, or inject fake ``# imgen batch`` / ``=== INPUT ===``
    marker lines into the per-batch log (defeating any tooling that
    parses it). v0.3.0 security review IMP-3.

    Three ranges blocked (same as the styles helper):

      * ``c < ' '``           — C0 controls (NUL, BEL, ESC, ...)
      * ``c == '\\x7f'``      — DEL
      * ``'\\x80' <= c <= '\\x9f'`` — C1 (0x9B alone is CSI on
        ECMA-48 terminals)

    Legitimate Unicode (emoji, CJK, accented Latin) passes through —
    all those characters live above U+009F.
    """
    return any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in name
    )


def resolve_single_input_path(image_arg: str, *, subcommand: str) -> Path:
    """Resolve, expand ~, and verify a single-file input path supplied
    via CLI. Shared by cmd_generate and cmd_refine (v0.7.7).

    v0.7.7 security #S2: the resolved stem ends up in the output
    filename, the `ok()` terminal display line, AND the
    `history.jsonl` record. Filenames with C0/DEL/C1 control bytes
    can inject ANSI escapes into the terminal (clear-screen, fake
    confirm-gate) or break tools that parse history. Cross-cutting
    hardening — batch path already filters via :func:`discover_inputs`,
    single-file paths through generate/refine did not until now.

    Exits with code 2 on any failure (missing, not-a-file, unsafe
    name). The ``subcommand`` parameter scopes the diagnostic message
    so users see the right verb in the error.
    """
    input_path = Path(image_arg).expanduser().resolve()
    if not input_path.exists():
        die(
            f"{subcommand}: input not found: {input_path}",
            code=2,
            hint="Check the path. Use absolute path if unsure.",
        )
    if not input_path.is_file():
        die(
            f"{subcommand}: input is not a file: {input_path}", code=2,
        )
    if _has_unsafe_controls(input_path.name):
        # repr() escapes the unsafe bytes for the error message so the
        # message itself doesn't re-emit them (otherwise rejecting an
        # ANSI-injection payload would itself perform the injection
        # in the warn output).
        die(
            f"{subcommand}: input filename contains unsafe control bytes: "
            f"{input_path.name!r}",
            code=2,
            hint="Rename the file — C0/DEL/C1 bytes can inject "
                 "terminal escape sequences into logs and prompts.",
        )
    return input_path


def discover_inputs(directory: Path) -> list[Path]:
    """Return supported image files directly under ``directory``.

    Non-recursive: ``directory/sub/photo.jpg`` is NOT returned. v0.3.0
    design — "what you see is what you batch" so behaviour is
    predictable and packages / mounts / hidden trees can't leak in.

    Filtering rules (in order):

    1. ``directory`` must exist and be a real directory — otherwise
       :func:`die` with exit code 2 (user-input class).
    2. Entries starting with ``.`` are skipped (covers ``.DS_Store``
       and any user-hidden files).
    3. Entries must be files (``is_file()``) — rejects subdirectories,
       including ``foo.jpg/`` directories and ``.photoslibrary``
       packages whose suffix would otherwise sneak past a naive glob.
    4. ``path.suffix.lower()`` must be in :data:`SUPPORTED_INPUT_EXTS`.
    5. Entries whose name contains C0/DEL/C1 control bytes are
       warn-and-skipped via :func:`_has_unsafe_controls` (security
       defence-in-depth — those names would inject escape sequences
       into the per-batch log and the user's terminal).
    6. Result sorted by name so log section indices ``[k/N]`` are
       stable across reruns.

    "Empty directory" or "no supported files" returns ``[]`` rather
    than dying — keeps the helper reusable for callers (e.g. doctor /
    dry-run preflight) that want to detect emptiness without catching
    SystemExit. ``cmd_batch`` turns ``[]`` into the user-facing
    "0 supported images in <dir>" error itself.

    Symlinks (both as the directory itself and as entries inside)
    are followed by design. The user explicitly passes ``<dir>``, so
    pointing at a symlinked photo collection is normal use; Lightroom /
    Capture One smart-preview workflows produce in-dir symlinks that
    callers expect to be processed. Compare ``LOGS_DIR`` which imgen
    picks internally and DOES symlink-guard — that asymmetry is
    intentional. (security NIT, v0.3.0 review)
    """
    if not directory.exists():
        die(f"Directory not found: {directory}",
            code=2,
            hint="Check the path. Use absolute path if unsure.")
    if not directory.is_dir():
        die(f"Not a directory: {directory}",
            code=2,
            hint="`imgen batch` takes a directory; for a single file "
                 "use `imgen generate <file>`.")

    out: list[Path] = []
    for entry in directory.iterdir():
        if entry.name.startswith("."):
            continue
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_INPUT_EXTS:
            continue
        if _has_unsafe_controls(entry.name):
            # Show the printable-repr so the warn() itself doesn't
            # propagate the escape into the user's terminal. Same
            # treatment as load_user_styles_dir for control-byte
            # style filenames.
            warn(f"Skipping {entry.name!r}: control bytes in filename "
                 "(unsafe — would inject escapes into log + terminal)")
            continue
        out.append(entry)
    out.sort(key=lambda p: p.name)
    return out


# ── check_input_stems ───────────────────────────────────────────────────


def check_input_stems(input_paths: list[Path]) -> None:
    """Die if any two inputs share a stem.

    v0.3.0 output layout is flat:
    ``<run_dir>/<input.stem>-<style>.png``. Two inputs with the same
    stem (e.g. ``IMG_1234.heic`` + ``IMG_1234.jpg``) would overwrite
    each other silently. Cheap preflight, no half-done batches.

    All offenders surfaced in one shot so the user fixes everything in
    one pass instead of re-running to discover the next collision.
    """
    if not input_paths:
        return
    stems = [p.stem for p in input_paths]
    duplicates = sorted({s for s, n in Counter(stems).items() if n > 1})
    if not duplicates:
        return
    offenders = ", ".join(f"'{s}'" for s in duplicates)
    die(f"Input stem collision: {offenders}. "
        "The flat output layout would overwrite — rename one of each pair.",
        code=2,
        hint="e.g. `mv IMG_1234.heic IMG_1234-orig.heic` before re-running.")


# ── HEIC support via sips ───────────────────────────────────────────────


def needs_jpeg_conversion(path: Path) -> bool:
    """True iff ``path`` is HEIC/HEIF and needs pre-conversion for mflux."""
    return path.suffix.lower() in HEIC_EXTS


def convert_heic_to_jpeg(src: Path, dst: Path) -> None:
    """Run ``sips -s format jpeg <src> --out <dst>``.

    sips is macOS-native (in every install since 10.3), so no extra
    dep. ``check=True`` raises ``CalledProcessError`` on non-zero exit;
    ``capture_output=True`` keeps sips' chatter out of our stdout.
    Caller decides whether to warn-and-skip or fail the whole batch.

    Two defensive measures (v0.3.0 pre-release review):

    * ``env=`` minimal allow-list — sips has no business with
      ``HF_TOKEN`` (or anything else in the parent shell). Inheriting
      the full ``os.environ`` would also leak HF credentials into
      sips' crash report under ``~/Library/Logs/DiagnosticReports/``
      should sips ever abort, and breaks the discipline established
      for mflux's env in :func:`~imgen.subprocess_helpers.build_minimal_env`.
    * ``timeout=60`` — a corrupt or pathologically large HEIC could
      stall sips indefinitely. At N=50 in a batch that's an opaque
      hang (the iteration's BatchLogger marker never fires because
      ``_run_one_iteration`` is never reached). 60s is generous: a
      real HEIC on Apple Silicon finishes in well under a second.
      A ``TimeoutExpired`` propagates as an unhandled exception so
      the user gets a traceback + batch abort rather than a silent
      stall.
    """
    minimal_env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "TMPDIR")
        if k in os.environ
    }
    subprocess.run(
        ["sips", "-s", "format", "jpeg", str(src), "--out", str(dst)],
        check=True,
        capture_output=True,
        env=minimal_env,
        timeout=60,
    )


def resolve_to_mflux_input(path: Path, cache_dir: Path) -> Path:
    """Return a path mflux can open directly.

    HEIC → convert into ``cache_dir/<stem>.jpg`` and return the cache
    path. Anything else → return ``path`` unchanged (no sips, no cache
    touched — avoids the 100-200ms per-image cost when nothing needs
    it).

    ``cache_dir`` is created with mode 0o700 on first HEIC if missing
    (parents=True, exist_ok=True so a second HEIC in the same batch
    doesn't trip EEXIST). Callers should pass a
    :class:`tempfile.TemporaryDirectory` path so the converted JPEGs
    are wiped on batch exit — they can contain identifiable subject
    matter and have no value beyond this run.
    """
    if not needs_jpeg_conversion(path):
        return path
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    jpeg = cache_dir / f"{path.stem}.jpg"
    convert_heic_to_jpeg(path, jpeg)
    return jpeg

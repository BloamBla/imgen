"""argparse parser construction + argument validators + --list-styles handler.

Lives separately from cli.py so command modules can stay focused on logic;
adding/changing a flag is one edit in this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from typing import Any

from . import __version__
from .backends import BUILTIN_BACKENDS, get_backend, list_backends
from .colors import C, step
from .defaults import DEFAULTS, MFLUX_PIN, PREVIEW_OVERRIDES
from .paths import DEFAULT_OUTPUT_DIR, SAFE_OUTPUT_EXTS
from .styles import get_style, list_styles, parse_style_list


def _style_list_type(value: str) -> list[str]:
    """argparse adapter for parse_style_list.

    argparse swallows ValueError messages and re-wraps them as the
    unhelpful `invalid X value: 'Y'`. Catching here and re-raising as
    ArgumentTypeError surfaces our detailed error (which names the
    offending styles plus the known set) directly to the user.

    parse_style_list itself stays pure (no argparse import), so future
    non-argparse callers — config validation, replay path, future
    `imgen batch` — can use it without dragging argparse along.
    """
    try:
        return parse_style_list(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


# ── Argparse validators ──────────────────────────────────────────────────

def _int_range(lo: int, hi: int):
    def validator(s: str) -> int:
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be an integer, got '{s}'")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"must be {lo}..{hi}, got {v}")
        return v
    return validator


def _float_range(lo: float, hi: float):
    def validator(s: str) -> float:
        try:
            v = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a number, got '{s}'")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"must be {lo}..{hi}, got {v}")
        return v
    return validator


def _safe_output_path(s: str) -> str:
    """argparse validator: reject output paths with non-image extensions."""
    ext = Path(s).suffix.lower()
    if ext not in SAFE_OUTPUT_EXTS:
        raise argparse.ArgumentTypeError(
            f"output extension must be one of "
            f"{sorted(SAFE_OUTPUT_EXTS)}, got '{ext or '(none)'}'")
    return s


# ── Parser ───────────────────────────────────────────────────────────────

def build_parser(
    epilog: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    """Build the top-level argparse parser.

    `epilog` is the usage-examples text shown after the help options. cli.py
    passes its module docstring here so the source of truth for that text
    stays with the entry module.

    `defaults` is the effective DEFAULTS dict (config.toml `[defaults]`
    merged over the module DEFAULTS). Used for argparse `default=` slots
    on `--style`/`--backend` so the CLI default reflects any user config.
    """
    if defaults is None:
        defaults = DEFAULTS

    p = argparse.ArgumentParser(
        prog="imgen",
        description="Photo style transfer for Apple Silicon Macs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    # Top-level utility flags
    p.add_argument("--list-styles", action="store_true",
                   help="List style presets and exit")
    p.add_argument("--list-backends", action="store_true",
                   help="List image-gen backends (built-in + ~/.imgen/backends.d/) and exit")
    # v0.3.5: `-v` short flag added — `node -v`/`npm -v`/`pip -V` all
    # use a single letter for version; users naturally try `imgen -v`
    # first and were getting "unrecognized arguments". `-v` doesn't
    # collide with any other flag in this parser (no -verbose mode).
    p.add_argument("-v", "--version", action="version",
                   version=f"imgen {__version__}")

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # setup / doctor / upgrade
    sub.add_parser("setup", help="First-time install & token setup")
    sub.add_parser("doctor", help="Check environment & cached models")
    u = sub.add_parser(
        "upgrade",
        help=f"Self-update imgen (git pull + reinstall) + refresh mflux "
             f"(pinned {MFLUX_PIN})",
    )
    u.add_argument("--latest", action="store_true",
                   help="Install newest mflux instead of pinned version "
                        "(may have breaking changes)")

    # clean
    c = sub.add_parser("clean", help="Cleanup HuggingFace cache")
    c.add_argument("--all", action="store_true",
                   help="Also delete cached models (with confirmation)")
    c.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted without deleting")

    # history / last / replay
    h = sub.add_parser("history", help="Show generation history")
    h.add_argument("--last", type=int, default=20,
                   help="Show last N entries (default 20)")
    sub.add_parser("last", help="Repeat last generation with new seed")
    r = sub.add_parser("replay", help="Repeat generation by id")
    r.add_argument("id", type=int)

    # generate (default — no subcommand, positional image)
    g = sub.add_parser("generate",
                       help="Generate styled image (default command)")
    _add_generate_args(g, defaults)

    # batch — N inputs × M styles (v0.3.0). Shares every generate flag
    # except `--output FILE` (single-file mutex doesn't apply when the
    # batch fan-out always produces multiple files).
    b = sub.add_parser(
        "batch",
        help="Apply M styles to every supported image in a directory "
             "(non-recursive). HEIC inputs auto-converted via sips.",
    )
    _add_batch_args(b, defaults)

    return p


def _add_generate_args(
    p: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
    p.add_argument("image", help="Path to input photo")
    # --style accepts a comma-list. `--style anime` → 1 generation,
    # `--style anime,ghibli,pixar` → 3 generations into the same run
    # folder. parse_style_list validates each name, dedupes (stable,
    # first occurrence wins, warn on dups). `choices=` is intentionally
    # NOT used — argparse compares the whole token against the list
    # which would reject the comma form.
    p.add_argument(
        "-s", "--style", type=_style_list_type, default=None,
        metavar="STYLE[,STYLE,...]",
        help=f"Style preset(s), comma-separated for multi-style "
             f"(default: {defaults['style']}). See: imgen --list-styles",
    )
    p.add_argument("--custom-prompt",
                   help="Custom prompt text. With an explicit --style and "
                        "a full preset, AUGMENTS the preset prompt (appended "
                        "as a final detail — v0.3.5+). Without --style, "
                        "becomes the sole prompt. Pass '-' to read from "
                        "stdin (hides the prompt from `ps auxww`).")
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="Read prompt from PATH instead of an argv string. "
                        "Mutually exclusive with --custom-prompt. Keeps "
                        "sensitive prompts out of process arguments.")
    # --output FILE writes to exactly that path (bypasses the
    # folder-per-invocation layout); --output-dir PATH overrides the
    # parent of the timestamped run folder. Mutex — only one shape
    # makes sense per invocation.
    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output", type=_safe_output_path,
        help=f"Output path with .png/.jpg/.jpeg/.webp suffix "
             f"(bypasses run-folder layout; default: "
             f"{DEFAULT_OUTPUT_DIR}/<start-ts>/<basename>-<style>.png)",
    )
    output_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Parent directory for the auto-named run folder. "
             "Overrides $IMGEN_OUTPUT_DIR and [defaults] output_dir.",
    )
    # Override args use default=None so we can tell "user set" from "use default"
    p.add_argument("--steps", type=_int_range(1, 200), default=None,
                   help=f"Inference steps 1..200 (default {defaults['steps']}, "
                        f"preview {PREVIEW_OVERRIDES['steps']})")
    p.add_argument("-g", "--guidance", type=_float_range(0.5, 15.0), default=None,
                   help=f"Guidance scale 0.5..15 (default {defaults['guidance']}, "
                        "style preset may override)")
    p.add_argument("--strength", type=_float_range(0.0, 1.0), default=None,
                   help=f"Image strength 0..1 (default {defaults['strength']}, "
                        "style preset may override)")
    p.add_argument("--seed", type=_int_range(0, 2**32 - 1),
                   help="Seed (default: random)")
    p.add_argument("--backend", choices=list_backends(),
                   default=defaults["backend"],
                   help=f"Backend (default {defaults['backend']})")
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
                   default=None,
                   help=f"Quantization (default {defaults['quantize']}, "
                        f"preview {PREVIEW_OVERRIDES['quantize']})")
    p.add_argument("--scope", choices=["person", "scene"], default="scene",
                   help="scene=transform whole image (default — most photos "
                        "are scenes, not portraits); person=keep background "
                        "photorealistic and unchanged")
    p.add_argument("-p", "--preview", action="store_true",
                   help="Fast preview mode: smaller resolution, fewer steps, "
                        "lower quantization (~5x faster, lower quality)")
    p.add_argument("--width", type=_int_range(64, 4096),
                   help="Override output width (64..4096)")
    p.add_argument("--height", type=_int_range(64, 4096),
                   help="Override output height (64..4096)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't open result in Preview")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the [y/N] confirm gate that fires when generating "
                        "multiple images (M ≥ 2 styles).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show mflux command without running")
    p.add_argument("--force", action="store_true",
                   help="Skip resource checks (RAM, parallel mflux, etc.) "
                        "and try anyway. Use at your own risk.")
    _add_enhance_args(p)


def _add_batch_args(
    p: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
    """Argparse stanza for `imgen batch <dir>` — superset of generate's
    flags minus `--output FILE` (which is mutex with batch's many-files
    fan-out)."""
    p.add_argument("directory",
                   help="Directory containing input photos (non-recursive). "
                        "Supported: jpg/jpeg/png/webp/heic/heif/bmp/tif/"
                        "tiff/gif; dotfiles skipped.")
    p.add_argument(
        "-s", "--style", type=_style_list_type, default=None,
        metavar="STYLE[,STYLE,...]",
        help=f"Style preset(s), comma-separated for multi-style "
             f"(default: {defaults['style']}). See: imgen --list-styles",
    )
    p.add_argument("--custom-prompt",
                   help="Custom prompt text. With an explicit --style and "
                        "a full preset, AUGMENTS the preset prompt (appended "
                        "as a final detail — v0.3.5+). Without --style, "
                        "becomes the sole prompt. Pass '-' to read from "
                        "stdin (hides from `ps auxww`).")
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="Read prompt from PATH instead of an argv string. "
                        "Mutually exclusive with --custom-prompt.")
    # No `--output FILE` here — batch always uses run-folder layout.
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Parent directory for the auto-named run folder. "
             "Overrides $IMGEN_OUTPUT_DIR and [defaults] output_dir.",
    )
    p.add_argument("--steps", type=_int_range(1, 200), default=None,
                   help=f"Inference steps 1..200 (default {defaults['steps']}, "
                        f"preview {PREVIEW_OVERRIDES['steps']})")
    p.add_argument("-g", "--guidance", type=_float_range(0.5, 15.0),
                   default=None,
                   help=f"Guidance scale 0.5..15 (default "
                        f"{defaults['guidance']}, style preset may override)")
    p.add_argument("--strength", type=_float_range(0.0, 1.0), default=None,
                   help=f"Image strength 0..1 (default {defaults['strength']}, "
                        "style preset may override)")
    p.add_argument("--seed", type=_int_range(0, 2**32 - 1),
                   help="Seed shared across the whole N×M batch "
                        "(default: random)")
    p.add_argument("--backend", choices=list_backends(),
                   default=defaults["backend"],
                   help=f"Backend (default {defaults['backend']})")
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
                   default=None,
                   help=f"Quantization (default {defaults['quantize']}, "
                        f"preview {PREVIEW_OVERRIDES['quantize']})")
    p.add_argument("--scope", choices=["person", "scene"], default="scene",
                   help="scene=transform whole image (default); "
                        "person=keep background photorealistic and unchanged")
    p.add_argument("-p", "--preview", action="store_true",
                   help="Fast preview mode applied uniformly across all "
                        "N×M generations (~5x faster, lower quality)")
    p.add_argument("--width", type=_int_range(64, 4096),
                   help="Override output width (uniform across the batch)")
    p.add_argument("--height", type=_int_range(64, 4096),
                   help="Override output height (uniform across the batch)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't open the run folder in Finder")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the N×M confirm gate")
    p.add_argument("--dry-run", action="store_true",
                   help="Show mflux command for every N×M iteration "
                        "without running")
    p.add_argument("--force", action="store_true",
                   help="Skip resource checks (RAM, parallel mflux, etc.) "
                        "and try anyway. Use at your own risk.")
    _add_enhance_args(p)


def _add_enhance_args(p: argparse.ArgumentParser) -> None:
    """LLM prompt enhancer flags. Shared by generate + batch.

    The enabled flag is a mutex pair (``--enhance-prompt`` /
    ``--no-enhance``) both writing to ``args.enhance``. ``None`` = no
    CLI override, fall back to ``[enhance] default`` from config.
    """
    group = p.add_argument_group(
        "Smart prompts",
        "Pipe the constructed prompt through a local AI model "
        "(Qwen2.5-7B-Instruct-4bit by default) to expand it into a "
        "richer, model-tuned version before mflux sees it. Opt-in.",
    )
    enable = group.add_mutually_exclusive_group()
    enable.add_argument(
        "--enhance-prompt", dest="enhance", action="store_true", default=None,
        help="Expand the prompt via the local LLM before generating.",
    )
    enable.add_argument(
        "--no-enhance", dest="enhance", action="store_false", default=None,
        help="Disable the enhancer for this run (overrides "
             "`[enhance] default = true` in config.toml).",
    )
    group.add_argument(
        "--enhance-model", type=str, default=None, metavar="REF",
        help="HF repo or local path for the enhancer LLM (overrides "
             "[enhance] model in config.toml).",
    )
    group.add_argument(
        "--enhance-temperature", type=_float_range(0.0, 2.0), default=None,
        metavar="T",
        help="Sampler temperature for the enhancer (0.0 = greedy = "
             "deterministic; default 0.0 for replay reproducibility).",
    )


def print_styles() -> int:
    """Handler for the top-level --list-styles flag."""
    step("Available styles")
    for name in list_styles():
        preset = get_style(name)
        prompt = preset.get("prompt") or "(param-only — pass --custom-prompt)"
        print(f"  {C.BOLD}{name:14}{C.END} "
              f"{C.DIM}(guidance={preset.get('guidance')}, "
              f"strength={preset.get('strength')}){C.END}")
        print(f"             {prompt[:80]}...")
    return 0


def print_backends() -> int:
    """Handler for the top-level --list-backends flag (v0.4).

    Shows every entry in the merged registry. Built-in vs custom is
    indicated with a marker; backends with a declared secret env var
    surface that too so the user knows which env vars they need set
    before running.
    """
    step("Available backends")
    for name in list_backends():
        be = get_backend(name)
        origin = "" if name in BUILTIN_BACKENDS else " (custom)"
        secret_marker = ""
        if be.secret_env_var is not None:
            req = "required" if be.secret_required else "optional"
            secret_marker = f"  [secret: ${be.secret_env_var} ({req})]"
        print(f"  {C.BOLD}{name:14}{C.END} "
              f"{C.DIM}({be.binary}{origin}){C.END}{secret_marker}")
    return 0

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
from .paths import DEFAULT_OUTPUT_DIR, HF_CACHE, SAFE_OUTPUT_EXTS
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


def _clean_model_ref(s: str) -> str:
    """argparse validator for ``--enhance-model``: reject empty and any
    C0/DEL/C1 control bytes. Symmetric with the ``[enhance] model``
    schema validator in ``config.py``. (v0.5 security IMP-4.)"""
    if not s.strip():
        raise argparse.ArgumentTypeError(
            "--enhance-model must be a non-empty HF repo or absolute path"
        )
    if any(c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f' for c in s):
        raise argparse.ArgumentTypeError(
            "--enhance-model contains control bytes (C0/DEL/C1) — "
            "reject so they don't reach mlx_lm.load or terminal output"
        )
    return s


def _lora_ref_arg(s: str):
    """argparse validator for ``--lora REF[:WEIGHT]``. Returns a
    :class:`imgen.styles.LoraRef` instance (repeatable; argparse's
    ``action='append'`` collects them into a list).

    Syntax:

    * Bare ref → weight defaults to 1.0:
      ``--lora "alvarobartt/ghibli-characters-flux-lora"``
    * ``REF:WEIGHT`` for explicit weight (single colon separator):
      ``--lora "alvarobartt/ghibli-characters-flux-lora:0.8"``

    Note: HF repo ids cannot contain ``:`` (only alphanumerics, ``-``,
    ``_``, ``/``, ``.``) so split-on-rightmost-colon disambiguates a
    weight suffix from any colon that might appear in an absolute
    path (``/Users/x/...`` doesn't contain ``:``; we only split if the
    suffix parses as a float).

    Compatible_with defaults to ``("flux-1",)`` — same as :class:`LoraRef`'s
    own default. Users who need a different compat group for a CLI-
    supplied LoRA should put it in a styles.d/*.toml entry instead
    where the full ``[[loras]]`` shape with explicit ``compatible_with``
    is available.

    Defence-in-depth: reject control bytes + oversized refs + weights
    outside [-2.0, 2.0] at parse time, matching the user-style schema.
    """
    from .styles import LoraRef

    # Inline byte caps + control-byte check matching the user-style
    # schema (styles._LORA_REF_MAX_LEN + styles._is_safe_stem).
    # Cross-module duplication here is intentional — the v0.6 design
    # memo flagged ``_safe.py`` extraction as v0.5+ candidate; until
    # that lands, parser.py keeps its own copy of the two constants
    # rather than reaching into styles.py's private names.
    _MAX_LEN = 4096

    def _has_control_bytes(s: str) -> bool:
        return any(
            c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f' for c in s
        )

    raw = s.strip()
    if not raw:
        raise argparse.ArgumentTypeError(
            "--lora value must be non-empty"
        )

    # Try to split a trailing ``:WEIGHT`` — ONLY for non-absolute refs.
    # v0.6 python-reviewer IMP-1: an absolute path like
    # ``/Users/x/lora-v1.0:2024`` (timestamped folder) or
    # ``/Volumes/.timemachine/disk:0.5`` ends in ``:<digits>``; the v0.5
    # rightmost-colon split would silently strip the suffix and load a
    # DIFFERENT file than the user pointed at. macOS allows ``:`` in
    # filenames at the APFS layer, so this is reachable in practice.
    # Restrict the split to refs that DON'T start with ``/`` — HF repo
    # ids never contain ``:`` (only alphanumerics + ``-_/.``), so the
    # split is safe and unambiguous for the HF-id case. Absolute paths
    # must use the upcoming styles.d/*.toml ``[[loras]] weight = ...``
    # shape if they need a non-default weight; CLI weight syntax stays
    # HF-only.
    ref = raw
    weight = 1.0
    if not raw.startswith("/") and ":" in raw:
        head, _, tail = raw.rpartition(":")
        try:
            candidate_weight = float(tail)
        except ValueError:
            pass
        else:
            if head.strip():
                ref = head.strip()
                weight = candidate_weight

    # v0.6 security-reviewer IMP-1: reject flag-shaped refs.
    # ``--lora "--config /etc/passwd"`` would land verbatim on mflux's
    # argv (build_mflux_cmd emits ``--lora-paths <ref>``); mflux's own
    # argparser may interpret a ``--``-prefixed token as an argparse
    # flag rather than a positional value, masking legitimate args
    # (``--negative-prompt``, ``--seed``, etc.) the iteration was
    # supposed to set. Absolute paths starting with ``/`` are fine —
    # any other ``-``-prefix is a flag shape and gets rejected at
    # validation time before it can reach mflux. Symmetric defence
    # with v0.4's ``_validate_binary_field`` posture in backends.py.
    if ref.startswith("-"):
        raise argparse.ArgumentTypeError(
            "--lora ref must not start with '-' (flag-shaped refs are "
            "rejected to prevent argv injection into mflux). Use an "
            "absolute path or HF repo id."
        )

    if len(ref) > _MAX_LEN:
        raise argparse.ArgumentTypeError(
            f"--lora ref too long ({len(ref)} bytes; cap {_MAX_LEN})"
        )
    if _has_control_bytes(ref):
        raise argparse.ArgumentTypeError(
            "--lora ref contains control bytes (C0/DEL/C1) — "
            "reject so they don't reach mlx_lm via subprocess argv"
        )
    if not (-2.0 <= weight <= 2.0):
        raise argparse.ArgumentTypeError(
            f"--lora weight out of range: {weight} (must be -2.0..2.0)"
        )

    return LoraRef(ref=ref, weight=weight, compatible_with=("flux-1",))


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
    p.add_argument("--list-loras", action="store_true",
                   help="List LoRA weight deltas referenced by built-in + user "
                        "styles, with HF cache status, and exit")
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
    _add_lora_args(p)


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
    _add_lora_args(p)


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
        "--enhance-model", type=_clean_model_ref, default=None, metavar="REF",
        help="HF repo or local path for the enhancer LLM (overrides "
             "[enhance] model in config.toml).",
    )
    group.add_argument(
        "--enhance-temperature", type=_float_range(0.0, 2.0), default=None,
        metavar="T",
        help="Sampler temperature for the enhancer (0.0 = greedy = "
             "deterministic; default 0.0 for replay reproducibility).",
    )


def _add_lora_args(p: argparse.ArgumentParser) -> None:
    """LoRA weight-delta flags. Shared by generate + batch.

    ``--lora`` is repeatable; each occurrence appends one
    :class:`styles.LoraRef` to ``args.lora`` (an actual list, not None).
    ``--no-lora`` is mutex with ``--lora`` and signals "drop both
    style-declared and CLI-declared LoRAs for this run". When neither
    is passed, the style's own ``loras`` field applies as-is.
    """
    group = p.add_argument_group(
        "LoRA stack",
        "LoRA weight deltas applied on top of the base diffusion "
        "model. Built-in styles may ship with curated LoRAs; "
        "--lora APPENDS additional ones; --no-lora drops them "
        "entirely for this run. Repeatable: --lora A --lora B:0.5.",
    )
    mode = group.add_mutually_exclusive_group()
    mode.add_argument(
        "--lora", action="append", type=_lora_ref_arg, default=None,
        metavar="REF[:WEIGHT]",
        help="LoRA HF repo id (e.g. 'strangerzonehf/Flux-Animeo-v1-LoRA') "
             "or absolute path to .safetensors, with optional :WEIGHT "
             "suffix (default 1.0). Repeatable to stack multiple LoRAs.",
    )
    mode.add_argument(
        "--no-lora", dest="no_lora", action="store_true", default=False,
        help="Drop any LoRAs the chosen style declares — generate from "
             "the base model alone. Useful for A/B comparing style + LoRA "
             "vs style + text-only.",
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


def _lora_hf_cache_dir(repo: str, hf_cache: Path) -> Path:
    """Return the ``models--<author>--<name>`` directory for an HF repo.

    Mirrors the same convention as ``doctor._hf_cache_dir_for``. Local
    absolute paths (LoraRef.ref can also be one) bypass the HF cache
    mapping — the ``ref`` IS the on-disk location.

    Security note (v0.6 security-reviewer IMP-2): the only consumer
    today is ``print_loras`` which calls ``cache_dir.is_dir()`` — a
    stat-only probe with no read or write. Even if ``repo`` is
    user-attacker-controlled and points at ``//host/share`` or
    ``/Volumes/external``, the worst outcome is "this path is reported
    as cached/not-cached in --list-loras" — information disclosure
    bounded to "does that filesystem path exist", which same-uid
    attackers can already determine via plain ``stat()``. Do NOT add
    file-read consumers without first anchoring under ``HF_CACHE``.

    The ``not repo`` branch is dead code reachable only by a hand-
    constructed LoraRef bypassing the user-style + parser schemas
    (both reject empty refs). Kept for symmetry with doctor's helper.
    """
    if not repo or repo.startswith("/"):
        return Path(repo) if repo else hf_cache
    return hf_cache / ("models--" + repo.replace("/", "--"))


def print_loras(hf_cache: Path | None = None) -> int:
    """Handler for the top-level --list-loras flag (v0.6).

    Walks every style in the merged registry and surfaces its
    ``loras`` tuple (empty for text-only styles). For each LoRA shows
    the HF repo / local path, weight, optional trigger phrase, compat
    group(s), and whether the weights are already cached locally so
    the user can predict cold-download cost.

    ``hf_cache`` parameter exists for tests (so they can point at a
    tmp directory and not depend on the real ``~/.cache/huggingface/
    hub/`` state). Production calls with ``None`` → ``HF_CACHE``.
    """
    if hf_cache is None:
        hf_cache = HF_CACHE
    step("Available LoRAs")

    text_only: list[str] = []
    with_loras: list[tuple[str, tuple]] = []
    for name in list_styles():
        preset = get_style(name)
        loras = preset.get("loras", ())
        if loras:
            with_loras.append((name, loras))
        else:
            text_only.append(name)

    if with_loras:
        print(f"  {C.BOLD}Styles shipping LoRAs:{C.END}")
        for style_name, loras in with_loras:
            for lora in loras:
                cache_dir = _lora_hf_cache_dir(lora.ref, hf_cache)
                cached = "cached" if cache_dir.is_dir() else "not downloaded"
                trigger = f' trigger="{lora.trigger}"' if lora.trigger else ""
                compat = ",".join(lora.compatible_with)
                print(f"    {C.BOLD}{style_name:14}{C.END} "
                      f"{lora.ref} "
                      f"{C.DIM}@{lora.weight:.2f}  [{compat}]{trigger}  "
                      f"({cached}){C.END}")
    if text_only:
        print(f"  {C.BOLD}Text-only styles (no LoRA):{C.END} "
              f"{C.DIM}{', '.join(text_only)}{C.END}")
    print()
    print(f"  {C.DIM}Override per-invocation with "
          f"--lora REF[:WEIGHT] (repeatable) or --no-lora.{C.END}")
    print(f"  {C.DIM}User styles in ~/.imgen/styles.d/*.toml may declare "
          f"[[loras]] entries — see README.{C.END}")
    return 0

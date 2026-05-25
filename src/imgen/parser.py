"""argparse parser construction + argument validators + --list-styles handler.

Lives separately from cli.py so command modules can stay focused on logic;
adding/changing a flag is one edit in this file.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from types import MappingProxyType

from typing import Any, Mapping

from . import __version__
from .backends import BUILTIN_BACKENDS, get_backend, list_backends
from .colors import C, die, step
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


def _clean_prompt_arg(s: str) -> str:
    """argparse validator for ``imgen refine --prompt``: reject any
    C0/DEL/C1 control bytes (v0.7.7 Sec #S3).

    Prompts flow through three rendering surfaces — ``--dry-run``
    output, ``imgen refine``'s ``ok()`` display line, and
    ``~/.imgen/history.jsonl`` ― any of which can render terminal
    escape sequences if the prompt carries them. ANSI ESC (0x1b),
    CSI (0x9b), and SOH/STX/etc would let a crafted prompt
    clear-screen or fake the confirm gate.

    Symmetric with :func:`_clean_model_ref` and the styles.d
    schema's ``no control bytes`` predicate. Empty is allowed
    here — refine's cmd-level resolution still falls back to the
    baked-in default when ``args.prompt is None`` (an empty string
    is a user choice "use empty prompt" and mflux accepts that;
    we just reject *unsafe* bytes, not lack of content).
    """
    if any(c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f' for c in s):
        raise argparse.ArgumentTypeError(
            "--prompt contains control bytes (C0/DEL/C1) — reject so "
            "they don't inject terminal escape sequences into logs / "
            "confirm gate / dry-run output"
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

    # v0.7.0 (architect §A): CLI default widened from `("flux-1",)` to
    # `("flux-1", "flux-dev")` so a user's `--lora foo/bar` works on
    # BOTH FLUX-Kontext (i2i, lora_compat_group="flux-1") AND FLUX.1-dev
    # (t2i, lora_compat_group="flux-dev"). User-style `~/.imgen/
    # styles.d/*.toml` `[[loras]] compatible_with = ["flux-1"]` entries
    # remain restrictive (authored with intent). Narrowing happens via
    # user-style TOML; CLI ships broad-by-default because the CLI user
    # doesn't know the architectural distinction. Per-LoRA verification
    # round (mirror of v0.6.3 work, flipped to flux-dev) is a v0.7.1
    # candidate ([[feedback-kontext-lora-compat]] discipline).
    return LoraRef(
        ref=ref, weight=weight, compatible_with=("flux-1", "flux-dev"),
    )


def _lora_refs_arg(s: str):
    """argparse validator for ``--lora`` accepting comma-list values.

    v0.7.0 (architect §C): `--lora` accepts both repeated flag AND
    comma-separated value, mirroring how `--style anime,ghibli,pixar`
    works in v0.2.3+. Examples:

      * ``--lora a/b`` → ``[LoraRef("a/b", 1.0)]``
      * ``--lora a/b:0.7,c/d`` → ``[LoraRef("a/b", 0.7),
        LoraRef("c/d", 1.0)]``
      * ``--lora a/b --lora c/d`` → 2 refs (repeated flag; argparse's
        ``action='append'`` collects, ``resolve_effective_loras``
        flattens the resulting list-of-lists)

    Each comma-split element is parsed via :func:`_lora_ref_arg`, so
    every per-element guard (control bytes, oversized refs, flag-shape
    rejection, weight range) fires identically.

    Whitespace around each element is stripped (``"a/b , c/d"`` parses
    as two clean refs). An empty element (e.g. ``"a,,b"``) is rejected
    — the caller probably meant to type a ref, and silently dropping
    empties would hide typos.

    Returns ``list[LoraRef]`` per call. The argparse stanza uses
    ``action="append"`` so the dest collects ``list[list[LoraRef]]``;
    ``cmd_helpers.resolve_effective_loras`` flattens at use-site.

    Note: HF repo ids contain ``-_/.`` only (no commas), so comma-split
    is unambiguous for the HF-id case. Absolute paths on macOS APFS
    can contain commas at the filesystem level; if a user has a LoRA
    at ``/Volumes/disk/lora-v1,backup.safetensors``, the comma-split
    will mis-parse. Documented limitation matching the colon-split
    one (v0.6 python IMP-1) — CLI weight + comma syntax are HF-only;
    absolute paths with separators must use styles.d TOML.
    """
    parts = [p.strip() for p in s.split(",")]
    if any(not p for p in parts):
        raise argparse.ArgumentTypeError(
            "--lora value contains an empty comma-element "
            f"(check for stray commas in {s!r})"
        )
    return [_lora_ref_arg(p) for p in parts]


# ── v0.8.0 — CLI rename helpers (commit 4a) ─────────────────────────────
#
# Per [[project-v080-design]] §I + §Q commit 4a. The user-facing flag
# `--backend NAME` becomes `--model NAME` across every subcommand, and
# two built-in names move to honest v0.8 spellings:
#
#   v0.7   →  v0.8
#   flux   →  flux-kontext        (FLUX.1-Kontext-dev, i2i)
#   qwen   →  qwen-image-edit-v1  (Qwen-Image-Edit v1, distinct from 2512)
#
# Other names (flux-dev, flux2-klein-edit-9b, user TOML stems) are
# unchanged. The registry source-of-truth STAYS at ``BUILTIN_BACKENDS``
# (keyed by v0.7 names) through commit 4a — commit 4b flips it to
# ``BUILTIN_MODELS`` (keyed by v0.8 names), at which point the inverse
# translation in ``_resolve_v07_alias`` becomes identity and the
# ``_V08_TO_V07_REGISTRY_KEY`` constant goes away (see TODO marker).

# MappingProxyType for runtime immutability — matches the project
# discipline of `frozenset` for `_DANGEROUS_ENV_VARS` in backends.py.
_V07_TO_V08_MODEL_RENAMES: Mapping[str, str] = MappingProxyType({
    "flux": "flux-kontext",
    "qwen": "qwen-image-edit-v1",
})

# Inverse map: v0.8 canonical name → v0.7 registry key. USED ONLY AT
# COMMIT 4a while ``BUILTIN_BACKENDS`` is the live registry. Commit 4b
# flips the registry to ``BUILTIN_MODELS`` (keyed by v0.8 names), at
# which point this constant + the inverse-map branch of
# ``_resolve_v07_alias`` collapse to identity and should be deleted.
# TODO commit 4b: remove _V08_TO_V07_REGISTRY_KEY + the .get() call
# inside _resolve_v07_alias once registry source-of-truth flips.
_V08_TO_V07_REGISTRY_KEY: Mapping[str, str] = MappingProxyType({
    v08: v07 for v07, v08 in _V07_TO_V08_MODEL_RENAMES.items()
})


def _check_for_deprecated_backend_flag(argv: list[str]) -> None:
    """Pre-argparse hook: detect ``--backend`` (space form) and
    ``--backend=VALUE`` (equals form), die with a STATIC migration
    hint.

    Called from ``cli.py`` BEFORE ``build_parser().parse_args(argv)`` —
    argparse alone would die with the uninformative "unrecognized
    arguments: --backend" once we drop the flag. This hook gives the
    user an actionable message naming the new flag.

    Security: the hint is STATIC text. The user's typed value is never
    echoed back, even via ``repr()``. This matches the
    ``backends.py:_validate_binary_field`` discipline (round-1 security
    MEDIUM) — if the user typed ``--backend=$'\\x1b[2J'``, echoing the
    value into stderr would leak the escape sequence to the terminal.

    Equals-form coverage: ``a.startswith("--backend=")`` matches the
    full ``--backend=foo`` token. A hypothetical future flag like
    ``--backend-something`` would NOT match because we check
    ``a == "--backend"`` (exact) and the equals-form prefix is
    ``--backend=`` (with the equals sign), not ``--backend``.
    """
    has_backend = any(
        a == "--backend" or a.startswith("--backend=") for a in argv
    )
    if has_backend:
        die(
            "imgen v0.8.0 renamed --backend → --model. Update your "
            "command:\n"
            "  Old: imgen photo.jpg --backend flux\n"
            "  New: imgen photo.jpg --model flux-kontext\n"
            "Some model names also changed; run `imgen --list-models` "
            "for the current registry."
        )


def _resolve_v07_alias(name: str) -> str:
    """argparse ``type=`` validator for ``--model``. Returns the v0.7
    registry key (suitable for ``get_backend()`` at commit 4a; becomes
    near-identity at commit 4b when registry flips to v0.8 names).

    Three branches:

    1. v0.7-typed name (``flux``, ``qwen``) → ``ArgumentTypeError``
       with rename hint. argparse converts this into a clean exit-2
       error message scoped to the ``--model`` argument.
    2. v0.8 canonical name in the rename map's inverse (``flux-kontext``
       → ``flux``) → translate to v0.7 registry key for lookup.
    3. Unchanged name (``flux-dev``, ``flux2-klein-edit-9b``, user TOML
       stems like ``z-image``) → pass through. Validated against
       ``list_backends()`` with a difflib closest-match hint on miss,
       which strictly beats argparse's default ``invalid choice``
       output (no fuzzy-match suggestion there).

    Note: argparse passes string ``default=`` values through ``type=``
    too — see ``build_parser`` where each i2i subcommand pre-translates
    ``defaults["backend"]`` through ``_V07_TO_V08_MODEL_RENAMES`` so
    the no-flag invocation path doesn't trip branch 1.
    """
    if name in _V07_TO_V08_MODEL_RENAMES:
        v08 = _V07_TO_V08_MODEL_RENAMES[name]
        raise argparse.ArgumentTypeError(
            f"{name!r} is the v0.7 model name. Use {v08!r} instead. "
            f"(--backend → --model rename in v0.8.0; some names also "
            f"changed.)"
        )
    resolved = _V08_TO_V07_REGISTRY_KEY.get(name, name)
    available = list_backends()
    if resolved not in available:
        # Show v0.8 canonical names in the hint so a typo on the new
        # spelling gets matched against the new spelling, not the
        # registry's still-v0.7 keys.
        v08_canonical = sorted({
            _V07_TO_V08_MODEL_RENAMES.get(n, n) for n in available
        })
        closest = difflib.get_close_matches(name, v08_canonical, n=1)
        hint = f" Did you mean {closest[0]!r}?" if closest else ""
        raise argparse.ArgumentTypeError(
            f"Unknown model {name!r}.{hint}"
        )
    return resolved


def _v07_default_to_v08_for_i2i(name: str) -> str:
    """Translate a v0.7 i2i backend default to its v0.8 canonical name
    for use as an argparse ``default=`` value.

    argparse runs ``type=`` on string defaults too — see the Python
    docs caveat on "default parsed as if it were a command-line
    argument". Without this pre-translation, ``default="flux"`` would
    crash on ``_resolve_v07_alias("flux")``'s v0.7-rename branch when
    the user invokes a subcommand without ``--model``.

    The rename map is applied ONLY to i2i defaults (generate / batch's
    ``defaults["backend"]``). Draw's ``defaults["backend_draw"]``
    default is ``flux-dev`` (already v0.8 canonical, not in the map)
    AND deliberately kept OUT of the translation — a user with
    ``[defaults] backend_draw = "flux"`` in their config was already
    broken on v0.7 (FLUX.1-Kontext is i2i, not t2i), and silently
    migrating to ``flux-kontext`` for the t2i subcommand would replace
    one wrong with a different wrong. Better to surface the
    v0.7-rename error via ``_resolve_v07_alias`` and let the user fix
    their config explicitly. (Architect HIGH-1 from v0.8.0 commit 4a
    design review.)

    Refine's hardcoded ``default="flux2-klein-edit-9b"`` is not in the
    rename map either, so passes through without translation.
    """
    return _V07_TO_V08_MODEL_RENAMES.get(name, name)


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

    # draw — v0.7.0 text-to-image. New subcommand for the first
    # non-i2i surface on imgen. Custom-prompt-only (no --style); LoRAs
    # are the expression vector. Default backend `flux-dev` (FLUX.1-dev
    # gated, same HF token as Kontext). See memory/project_v070_design.md.
    d = sub.add_parser(
        "draw",
        help="Text-to-image: generate from a prompt with no input photo. "
             "v0.7.0+ — uses FLUX.1-dev via mflux-generate.",
    )
    _add_draw_args(d, defaults)

    # refine — v0.7.5 Hires-Fix path: take an existing 1024² image and
    # produce a higher-res (--scale 1.5 → 1536, --scale 2 → 2048)
    # version with low-strength i2i refine for sharper detail. Closes
    # the canonical explore→refine pipeline: `imgen draw "..." -n 5
    # --preview` → pick winner → `imgen refine <winner.png>`. Default
    # backend = flux2-klein-edit-9b (FLUX.2-klein-9B distilled edit
    # variant — native ~4 MP support, sweet spot for 1.5-2K refine).
    r = sub.add_parser(
        "refine",
        help="Hires-Fix: upscale + refine an existing image to ~1.5K/2K. "
             "v0.7.5+ — uses FLUX.2-klein-9B via mflux-generate-flux2-edit.",
    )
    _add_refine_args(r, defaults)

    return p


def _add_run_control_args(
    p: argparse.ArgumentParser,
    *,
    preview_help: str = "Fast preview mode (smaller resolution + steps, lower quantization; ~5x faster, lower quality)",
    no_open_help: str = "Don't open the result in Preview",
    yes_help: str = "Skip the [y/N] confirm gate",
    dry_run_help: str = "Show mflux command without running",
    force_help: str = "Skip resource checks (RAM, parallel mflux, etc.) and try anyway. Use at your own risk.",
) -> None:
    """Universal run-control flags shared by every subcommand
    (generate / batch / draw / refine). v0.7.9 extraction — closes
    python NIT #6 from the v0.7.5 review trail (deferred until
    pattern emerged; the 3rd subcommand `refine` crossed the
    threshold and v0.7.8 architect re-confirmed the rule).

    Per-subcommand help text via kwargs — flag SHAPE (action,
    dashes, short-form) centralised so a future flag-shape change
    (e.g. ``--dry-run`` becomes a choice flag, ``--no-open`` gains
    a short alias) lands in one place rather than four. Each
    keyword default is a generic phrasing that works without
    customisation when a future subcommand wants minimum-overhead
    onboarding.
    """
    p.add_argument(
        "-p", "--preview", action="store_true", help=preview_help,
    )
    p.add_argument("--no-open", action="store_true", help=no_open_help)
    p.add_argument("-y", "--yes", action="store_true", help=yes_help)
    p.add_argument("--dry-run", action="store_true", help=dry_run_help)
    p.add_argument("--force", action="store_true", help=force_help)


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
        help="Style preset(s), comma-separated for multi-style. "
             "v0.7.13+: explicit opt-in. Omit --style AND pass "
             "--custom-prompt TEXT (or --prompt-file PATH) for bare "
             "mode (no preset baggage — raw prompt straight to mflux, "
             "no preset prefix / negative_prompt leak / LoRA stack). "
             "See: imgen --list-styles",
    )
    p.add_argument("--custom-prompt", type=_clean_prompt_arg,
                   help="Custom prompt text. With an explicit --style and "
                        "a full preset, AUGMENTS the preset prompt (appended "
                        "as a final detail — v0.3.5+). Without --style, "
                        "becomes the sole prompt and triggers bare mode "
                        "(v0.7.13+: no preset baggage). Pass '-' to read "
                        "from stdin (hides the prompt from `ps auxww`).")
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
    p.add_argument("-g", "--guidance", type=_float_range(0.0, 15.0), default=None,
                   help=f"Guidance scale 0..15 (default {defaults['guidance']}; "
                        f"0 disables guidance — required for distilled models "
                        f"like FLUX-schnell/Z-Image-Turbo, "
                        "style preset may override)")
    p.add_argument("--strength", type=_float_range(0.0, 1.0), default=None,
                   help=f"Image strength 0..1 (default {defaults['strength']}, "
                        "style preset may override)")
    p.add_argument("--seed", type=_int_range(0, 2**32 - 1),
                   help="Seed (default: random)")
    # v0.8.0 commit 4a: --backend → --model. `dest="backend"` keeps the
    # args attribute name unchanged at this commit (registry source-of-
    # truth flip is commit 4b — don't pre-empt). Default pre-translated
    # to v0.8 canonical name so argparse's default-passes-through-type=
    # behaviour doesn't crash on the v0.7 rename in _resolve_v07_alias.
    _v08_default = _v07_default_to_v08_for_i2i(defaults["backend"])
    p.add_argument(
        "--model", type=_resolve_v07_alias, dest="backend",
        default=_v08_default, metavar="NAME",
        help=f"Model (default {_v08_default}). "
             f"Run --list-backends for the full set.",
    )
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
                   default=None,
                   help=f"Quantization (default {defaults['quantize']}, "
                        f"preview {PREVIEW_OVERRIDES['quantize']})")
    p.add_argument("--scope", choices=["person", "scene"], default="scene",
                   help="scene=transform whole image (default — most photos "
                        "are scenes, not portraits); person=keep background "
                        "photorealistic and unchanged")
    p.add_argument("--width", type=_int_range(64, 4096),
                   help="Override output width (64..4096)")
    p.add_argument("--height", type=_int_range(64, 4096),
                   help="Override output height (64..4096)")
    _add_run_control_args(
        p,
        preview_help="Fast preview mode: smaller resolution, fewer steps, lower quantization (~5x faster, lower quality)",
        no_open_help="Don't open result in Preview",
        yes_help="Skip the [y/N] confirm gate that fires when generating multiple images (M ≥ 2 styles).",
    )
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
        help="Style preset(s), comma-separated for multi-style. "
             "v0.7.13+: explicit opt-in. Omit --style AND pass "
             "--custom-prompt TEXT (or --prompt-file PATH) for bare "
             "mode (no preset baggage — raw prompt straight to mflux, "
             "no preset prefix / negative_prompt leak / LoRA stack). "
             "See: imgen --list-styles",
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
    p.add_argument("-g", "--guidance", type=_float_range(0.0, 15.0),
                   default=None,
                   help=f"Guidance scale 0..15 (default {defaults['guidance']}; "
                        f"0 disables guidance — required for distilled models "
                        f"like FLUX-schnell/Z-Image-Turbo, "
                        "style preset may override)")
    p.add_argument("--strength", type=_float_range(0.0, 1.0), default=None,
                   help=f"Image strength 0..1 (default {defaults['strength']}, "
                        "style preset may override)")
    p.add_argument("--seed", type=_int_range(0, 2**32 - 1),
                   help="Seed shared across the whole N×M batch "
                        "(default: random)")
    # v0.8.0 commit 4a: same --backend → --model rename as generate.
    # Shares the same i2i default (defaults["backend"]) → pre-translate.
    _v08_default = _v07_default_to_v08_for_i2i(defaults["backend"])
    p.add_argument(
        "--model", type=_resolve_v07_alias, dest="backend",
        default=_v08_default, metavar="NAME",
        help=f"Model (default {_v08_default}). "
             f"Run --list-backends for the full set.",
    )
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
                   default=None,
                   help=f"Quantization (default {defaults['quantize']}, "
                        f"preview {PREVIEW_OVERRIDES['quantize']})")
    p.add_argument("--scope", choices=["person", "scene"], default="scene",
                   help="scene=transform whole image (default); "
                        "person=keep background photorealistic and unchanged")
    p.add_argument("--width", type=_int_range(64, 4096),
                   help="Override output width (uniform across the batch)")
    p.add_argument("--height", type=_int_range(64, 4096),
                   help="Override output height (uniform across the batch)")
    _add_run_control_args(
        p,
        preview_help="Fast preview mode applied uniformly across all N×M generations (~5x faster, lower quality)",
        no_open_help="Don't open the run folder in Finder",
        yes_help="Skip the N×M confirm gate",
        dry_run_help="Show mflux command for every N×M iteration without running",
    )
    _add_enhance_args(p)
    _add_lora_args(p)


def _add_draw_args(
    p: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
    """Argparse stanza for `imgen draw <prompt>` — v0.7.0 t2i.

    Differences from cmd_generate: positional prompt (not --image),
    no --style (LoRAs are the expression vector), no --scope (i2i-only
    parser flag), no --strength (no source photo to interpolate
    against). --width / --height carry defaults since there's no input
    to detect from.
    """
    # Positional prompt is the primary input — mutex with --prompt-file.
    # The mutex check fires in cmd_draw (argparse mutex groups don't
    # compose cleanly with optional positionals).
    p.add_argument(
        "prompt", nargs="?", default=None,
        help="Text prompt for image generation. Mutually exclusive "
             "with --prompt-file. Pass '-' as the positional to read "
             "from stdin (hides the prompt from `ps auxww`).",
    )
    p.add_argument(
        "--prompt-file", type=Path, default=None,
        help="Read prompt from PATH instead of the positional. "
             "Mutually exclusive with the positional prompt.",
    )

    # Output: --output PATH single-file canonical, --output-dir DIR
    # uses the existing folder-per-invocation layout.
    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output", type=_safe_output_path,
        help=f"Output path with .png/.jpg/.jpeg/.webp suffix "
             f"(bypasses run-folder layout; default: "
             f"{DEFAULT_OUTPUT_DIR}/<start-ts>/<prompt-slug>.png)",
    )
    output_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Parent directory for the auto-named run folder. "
             "Overrides $IMGEN_OUTPUT_DIR and [defaults] output_dir.",
    )

    p.add_argument(
        "--steps", type=_int_range(1, 200), default=None,
        help=f"Inference steps 1..200 (default {defaults['steps']}, "
             f"preview {PREVIEW_OVERRIDES['steps']})",
    )
    p.add_argument(
        "-g", "--guidance", type=_float_range(0.0, 15.0), default=None,
        help=f"Guidance scale 0..15 (default {defaults['guidance']}; "
             f"FLUX.1-dev canonical is 3.5 — pass explicitly if you "
             f"want the tighter scale. Pass 0 to disable guidance — "
             f"required for distilled backends like Z-Image-Turbo).",
    )
    # v0.7.11 (gap 1): expose --negative-prompt to imgen draw.
    # Pre-v0.7.11 the CLI had no way to set a negative prompt for t2i,
    # even though :func:`backends.build_mflux_cmd` emits it when present
    # on backends with ``supports_negative=True`` (flux-dev qualifies).
    # Z-Image and FLUX.1-dev model cards both recommend negatives for
    # quality steering; this closes that gap. Reuses the existing
    # ``_clean_prompt_arg`` validator (control-byte stripping — same
    # discipline as refine's `--prompt`).
    p.add_argument(
        "--negative-prompt", type=_clean_prompt_arg, default=None,
        help="Negative prompt — concepts to steer the model AWAY from "
             "(e.g. 'low quality, blurry, deformed'). Only honoured on "
             "backends with supports_negative=True (flux, flux-dev, etc.; "
             "qwen + flux2-klein-edit-9b silently drop it).",
    )
    p.add_argument(
        "--seed", type=_int_range(0, 2**32 - 1), default=None,
        help="Seed (default: random). With --num-iterations N, this is "
             "the BASE seed; subsequent iterations use base+1, base+2, ...",
    )
    # v0.7.3: --num-iterations N — explore-mode randomness ladder.
    # The canonical workflow: --preview --num-iterations 5 → 5 variations
    # of the same prompt at preview cost (~2:50 each on M2 Pro), pick the
    # best, then refine via cmd_generate i2i at Q8. Cap at 32 — protects
    # against accidental `-n 9999`; nobody legitimately needs more in
    # one invocation.
    p.add_argument(
        "-n", "--num-iterations", type=_int_range(1, 32), default=1,
        metavar="N",
        help="Generate N variations of the prompt (different seeds). "
             "Default 1. Cap 32. With --seed X: seeds are X, X+1, ..., "
             "X+N-1 (reproducible ladder). Without --seed: random base. "
             "Output naming: N=1 keeps <slug>.png; N>=2 emits "
             "<slug>-1.png ... <slug>-N.png.",
    )
    # v0.7.0: default `--backend flux-dev`. The shared `defaults["backend"]`
    # entry is "flux" (Kontext, i2i); draw's default lives at the
    # separate ``defaults["backend_draw"]`` key so config.toml
    # ``[defaults] backend_draw = "..."`` can override per-subcommand
    # without crossing i2i's default. Architect IMP-3 from pre-tag
    # review.
    #
    # v0.8.0 commit 4a: --backend → --model. ``backend_draw`` default
    # (``flux-dev``) is already v0.8 canonical, so it does NOT go
    # through ``_v07_default_to_v08_for_i2i`` — that helper translates
    # i2i defaults only. Architect HIGH-1 from 4a design review: a
    # user with ``[defaults] backend_draw = "flux"`` was already broken
    # on v0.7 (Kontext is i2i, not t2i); silently migrating to
    # ``flux-kontext`` would replace one wrong with a different wrong.
    # The ``_resolve_v07_alias`` ArgumentTypeError surfaces the rename
    # explicitly so the user fixes their config.
    _draw_default = defaults.get("backend_draw", "flux-dev")
    p.add_argument(
        "--model", type=_resolve_v07_alias, dest="backend",
        default=_draw_default, metavar="NAME",
        help=f"Model (default {_draw_default}). "
             f"Run --list-backends to see all.",
    )
    p.add_argument(
        "-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8], default=None,
        help=f"Quantization (default {defaults['quantize']}, "
             f"preview {PREVIEW_OVERRIDES['quantize']})",
    )
    # t2i: --width/--height carry defaults — there's no input to detect
    # from. 1024x1024 is FLUX.1-dev canonical.
    p.add_argument(
        "--width", type=_int_range(64, 4096), default=1024,
        help="Output width 64..4096 (default 1024)",
    )
    p.add_argument(
        "--height", type=_int_range(64, 4096), default=1024,
        help="Output height 64..4096 (default 1024)",
    )
    _add_run_control_args(
        p,
        preview_help="Fast preview mode: smaller resolution, fewer steps, lower quantization (~5x faster, lower quality)",
    )
    _add_enhance_args(p)
    _add_lora_args(p)


def _add_refine_args(
    p: argparse.ArgumentParser,
    defaults: dict[str, Any],
) -> None:
    """Argparse stanza for `imgen refine <input>` — v0.7.5 Hires-Fix path.

    Positional <input> = existing image (typically 1024² from a prior
    ``imgen draw`` or any source). --scale OR --width/--height (mutex)
    sets target resolution. Default backend FLUX.2-klein-edit-9b.
    """
    p.add_argument(
        "input", help="Path to existing image to refine (PNG/JPG/etc).",
    )
    # --scale OR --width/--height. Mutex enforced in cmd_refine; argparse
    # mutex groups don't play cleanly with three flags.
    p.add_argument(
        "--scale", type=_float_range(1.0, 4.0), default=None,
        help="Multiply input dimensions by N (rounded to multiple of 16). "
             "Default 1.5 (1024² → 1536²). 2.0 → 2048² (native FLUX.2-klein "
             "cap of 4 MP). Mutex with --width/--height.",
    )
    p.add_argument(
        "--width", type=_int_range(64, 4096), default=None,
        help="Explicit output width (64..4096). Mutex with --scale.",
    )
    p.add_argument(
        "--height", type=_int_range(64, 4096), default=None,
        help="Explicit output height. Mutex with --scale.",
    )
    p.add_argument(
        "--prompt", type=_clean_prompt_arg, default=None,
        help="Refine prompt override. Default focuses on detail/sharpness "
             "while preserving composition; print it with --dry-run.",
    )
    output_group = p.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output", type=_safe_output_path,
        help=f"Output path with .png/.jpg suffix (bypasses run-folder layout; "
             f"default: {DEFAULT_OUTPUT_DIR}/<start-ts>/<input-stem>-refined.png)",
    )
    output_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Parent directory for the auto-named run folder.",
    )
    p.add_argument(
        "--steps", type=_int_range(1, 200), default=None,
        help=f"Inference steps (default {defaults['steps']}). FLUX.2-klein "
             f"distilled converges fast; 20-25 is typical for refine.",
    )
    p.add_argument(
        "-g", "--guidance", type=_float_range(0.0, 15.0), default=None,
        help=f"Guidance scale (default {defaults['guidance']}). Ignored "
             f"on the default flux2-klein-edit-9b backend — mflux pins "
             f"guidance to 1.0 for non-base FLUX.2 models.",
    )
    p.add_argument(
        "--strength", type=_float_range(0.0, 1.0), default=0.3,
        help="Refine strength (default 0.3 — low so input composition is "
             "preserved; raise to 0.5+ for more aggressive restyling). "
             "FLUX.2-klein-edit doesn't directly consume this — currently "
             "recorded in history for metadata only.",
    )
    p.add_argument(
        "--seed", type=_int_range(0, 2**32 - 1), default=None,
        help="Seed (default: random)",
    )
    # v0.8.0 commit 4a: --backend → --model. Refine's hardcoded default
    # ``flux2-klein-edit-9b`` is already v0.8 canonical (not in the
    # rename map) so passes through ``_resolve_v07_alias`` unchanged.
    p.add_argument(
        "--model", type=_resolve_v07_alias, dest="backend",
        default="flux2-klein-edit-9b", metavar="NAME",
        help="Model (default flux2-klein-edit-9b — FLUX.2-klein-9B distilled "
             "edit variant). Override with --model flux-kontext to use "
             "FLUX.1-Kontext-dev (faster, already cached, lower native res "
             "ceiling).",
    )
    p.add_argument(
        "-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
        default=4,
        help="Quantization (default 4 — safe for 2K² activations on 32GB "
             "Mac with klein-9B). Use 8 for max quality if you have headroom.",
    )
    _add_run_control_args(
        p,
        preview_help="Fast preview mode (smaller resolution + steps).",
        force_help="Skip resource checks (RAM, parallel mflux, etc.)",
    )
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
        "--enhance-prompt", dest="enhance", action="store_true",
        help="Expand the prompt via the local LLM before generating.",
    )
    enable.add_argument(
        "--no-enhance", dest="enhance", action="store_false",
        help="Disable the enhancer for this run (overrides "
             "`[enhance] default = true` in config.toml).",
    )
    # v0.5 python I-5: argparse takes the LAST ``default=`` it sees on
    # the dest when args share a dest via ``add_mutually_exclusive_group``,
    # which makes ``default=None`` on each arg order-fragile. Set the
    # default once on the parser instead so a future re-order can't
    # silently flip the "no CLI override" sentinel.
    p.set_defaults(enhance=None)
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
        "--lora", action="append", type=_lora_refs_arg, default=None,
        metavar="REF[:WEIGHT][,REF[:WEIGHT]...]",
        help="LoRA HF repo id (e.g. 'strangerzonehf/Flux-Animeo-v1-LoRA') "
             "or absolute path to .safetensors, with optional :WEIGHT "
             "suffix (default 1.0). Repeatable AND comma-split: "
             "--lora A,B:0.5 --lora C stacks 3 LoRAs in arg order.",
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
        prompt = preset.prompt or "(param-only — pass --custom-prompt)"
        print(f"  {C.BOLD}{name:14}{C.END} "
              f"{C.DIM}(guidance={preset.guidance}, "
              f"strength={preset.strength}){C.END}")
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


# Security note (v0.6 security-reviewer IMP-2): the only consumer of
# ``hf_cache_dir_for`` from this module is ``print_loras`` below, which
# calls ``cache_dir.is_dir()`` — a stat-only probe with no read or write.
# Even if ``repo`` is user-attacker-controlled and points at
# ``//host/share`` or ``/Volumes/external``, the worst outcome is "this
# path is reported as cached/not-cached in --list-loras" — information
# disclosure bounded to "does that filesystem path exist", which same-uid
# attackers can already determine via plain ``stat()``. Do NOT add
# file-read consumers without first anchoring under ``HF_CACHE``.
from .hf_cache import hf_cache_dir_for


def print_loras(
    hf_cache: Path | None = None,
    mflux_loras_cache: Path | None = None,
) -> int:
    """Handler for the top-level --list-loras flag (v0.6).

    Walks every style in the merged registry and surfaces its
    ``loras`` tuple (empty for text-only styles). For each LoRA shows
    the HF repo / local path, weight, optional trigger phrase, compat
    group(s), and whether the weights are already cached locally so
    the user can predict cold-download cost.

    Two cache roots are probed (v0.6.4 task #21): the standard HF hub
    cache (``~/.cache/huggingface/hub/``) which is where most
    ``huggingface_hub.snapshot_download`` calls land, AND mflux's
    private LoRA cache (``~/Library/Caches/mflux/loras/``) which is
    where mflux's own LoRA downloader writes. v0.6.3 only probed the
    HF hub cache and so reported "not downloaded" for every built-in
    LoRA after a successful smoke run — the weights were in mflux's
    cache, not HF's. We now report "cached" when EITHER root has the
    ``models--<author>--<name>`` directory.

    Both ``hf_cache`` and ``mflux_loras_cache`` parameters exist for
    tests (point at tmp directories). Production passes ``None`` for
    both → :data:`HF_CACHE` + :data:`MFLUX_LORAS_CACHE` from paths.py.
    """
    if hf_cache is None:
        hf_cache = HF_CACHE
    if mflux_loras_cache is None:
        from .paths import MFLUX_LORAS_CACHE
        mflux_loras_cache = MFLUX_LORAS_CACHE
    step("Available LoRAs")

    text_only: list[str] = []
    with_loras: list[tuple[str, tuple]] = []
    for name in list_styles():
        preset = get_style(name)
        loras = preset.loras
        if loras:
            with_loras.append((name, loras))
        else:
            text_only.append(name)

    if with_loras:
        print(f"  {C.BOLD}Styles shipping LoRAs:{C.END}")
        for style_name, loras in with_loras:
            for lora in loras:
                # Local absolute paths bypass the HF cache layout — the
                # `ref` IS the on-disk location, and it's a .safetensors
                # FILE not a directory. Probing `is_dir()` on a file
                # would falsely report "not downloaded" for an existing
                # local LoRA. (v0.6.x backlog python NIT-1.)
                if lora.ref.startswith("/"):
                    cached = "local" if Path(lora.ref).is_file() else "missing"
                else:
                    # v0.6.4 task #21: check both caches; mflux writes
                    # to its own LoRA cache, not the HF hub cache.
                    hf_dir = hf_cache_dir_for(lora.ref, hf_cache)
                    mflux_dir = hf_cache_dir_for(lora.ref, mflux_loras_cache)
                    cached = (
                        "cached" if hf_dir.is_dir() or mflux_dir.is_dir()
                        else "not downloaded"
                    )
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

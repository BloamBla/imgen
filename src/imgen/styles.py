"""
Style presets for imgen.

Built-in presets live in BUILTIN_STYLES — modifying that dict needs a
code change. Users can drop additional `.toml` files into
`~/.imgen/styles.d/`; load_user_styles_dir() reads them and
merge_user_styles() folds them into the final dict surfaced via
list_styles() / get_style() (cached per process).

Each preset is a fully-formed instruction for FLUX Kontext / Qwen Image Edit
to transform a person photo into a target art style while preserving identity.

Per-style tuning of `guidance` and `strength` is allowed when defaults don't
work well (e.g. Simpsons needs higher guidance to nail the distinctive look).
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Callable

# Forward-declared so loader can warn without a circular import on .colors.
# Resolved below the BUILTIN_STYLES dict.


__all__ = [
    "BUILTIN_STYLES",
    "STYLES",
    "USER_STYLE_MAX_BYTES",
    "StyleNotFound",
    "UserStyleError",
    "get_style",
    "list_styles",
    "load_user_style_file",
    "load_user_styles_dir",
    "merge_user_styles",
    "parse_style_list",
    "reset_styles_cache",
]


class StyleNotFound(KeyError):
    """Raised by ``get_style`` when the requested style name is unknown.

    Subclass of ``KeyError`` so existing ``except KeyError`` handlers
    (e.g. ``cmd_helpers.resolve_styles_list``) keep matching, but with
    a clean ``__str__`` that avoids ``KeyError``'s historical
    repr-quoting (``"'msg'"`` instead of ``msg``). (python #21 from
    the v0.1.x review.)
    """

    def __str__(self) -> str:
        # str(...) wrap defends against a non-string first arg — CPython's
        # __str__ contract requires returning str, so a bare `return
        # self.args[0]` would raise TypeError if a caller constructed
        # StyleNotFound with a non-string positional. Today's raise site
        # passes an f-string, but the subclass surface is public and
        # future callers shouldn't have to read the source to discover
        # the implicit str-only contract. (v0.3.6 python-reviewer CRITICAL.)
        return str(self.args[0]) if self.args else ""

# Cap on per-file size for ~/.imgen/styles.d/*.toml. Real style files are
# under 2 KB (one prompt + a few tunings); 256 KB is way above realistic
# use, while still bounded to defend against a rogue/oversized file
# OOM'ing tomllib.
USER_STYLE_MAX_BYTES = 256 * 1024


# Prompt structure follows BFL's official FLUX.1 Kontext guidance
# (docs.bfl.ai/guides/prompting_guide_kontext_i2i, June 2025) and the
# mflux Kontext README. Three-layer pattern, applied uniformly to every
# built-in preset:
#
#   ACTION  → "Restyle this person as <STYLE NOUN>"
#   ──────────  ("Restyle"/"Convert" preferred over "Transform [person]",
#                which BFL explicitly flags as identity-drift risk:
#                ❌ "Transform the person into a Viking"
#                ✅ "Change the clothes to a Viking warrior while
#                    preserving facial features")
#
#   PRESERVATION → "while preserving the exact facial features,
#   ──────────     hairstyle, body proportions, and pose"
#                  (anchored mid-prompt with "while" connector — Kontext
#                  weights mid-sentence preservation tokens consistently
#                  vs tail tokens. Four explicit identity anchors:
#                  face / hair / body / pose. v0.3.4 user goal: preserve
#                  face + figure as much as possible.)
#
#   STYLE DESCRIPTORS → ", with <concrete visual attributes>"
#   ─────────────────  (named style + concrete attributes per BFL:
#                       "name the style + describe characteristics".
#                       Lands at the tail — also leaves a clean slot
#                       for v0.3.5 augmentation via --custom-prompt.)
#
# Negative prompts stay focused on artifact rejection (deformed, blurry,
# watermark) — they were not the diagnosed issue and don't need surgery.

BUILTIN_STYLES: dict[str, dict] = {
    "pixar": {
        # Pixar restructures facial geometry (rounded features, big
        # eyes are core to the style). "Exact facial features" would
        # contradict the style descriptors below; "facial identity"
        # tells the model to keep WHO it is while letting the style
        # reshape HOW. (v0.3.4 review HIGH-1.)
        "prompt": (
            "Restyle this person as a polished Pixar 3D animated "
            "character, while preserving the facial identity, "
            "hairstyle, body proportions, and pose, with soft volumetric "
            "lighting, smooth rounded features, expressive large eyes, "
            "stylized cartoon proportions, and high-quality CGI rendering"
        ),
        "negative": (
            "deformed, blurry, photorealistic skin, flat lighting, missing eye, "
            "extra limbs, distorted face, low quality, artifacts, watermark, text"
        ),
        "guidance": 3.5,
        "strength": 0.55,
    },

    "anime": {
        # Anime enlarges eyes + reshapes face geometry — see pixar
        # comment. Same "facial identity" anchor instead of "exact
        # facial features". (v0.3.4 review HIGH-1.)
        "prompt": (
            "Restyle this person as a Japanese anime character, while "
            "preserving the facial identity, hairstyle, body "
            "proportions, and pose, with cel-shaded illustration, "
            "expressive large eyes, detailed line art, vibrant colors, "
            "clean shading, and manga aesthetic"
        ),
        "negative": (
            "realistic photo, 3d render, deformed face, bad anatomy, extra "
            "limbs, blurry, low quality, watermark, text"
        ),
        "guidance": 4.0,
        "strength": 0.60,
    },

    "simpsons": {
        # Simpsons style fundamentally restructures facial features
        # (yellow skin + large round white eyes are core to the style),
        # so "exact facial features" would create contradictory
        # instructions. Use "recognizable expression" as the identity
        # anchor instead — that's the most stylistically-flexible token
        # the model can hold onto while still rebuilding the face.
        "prompt": (
            "Restyle this person as a Matt Groening Simpsons character, "
            "while preserving the recognizable expression, hairstyle, "
            "body proportions, and pose, with bright yellow skin, large "
            "round white eyes with small black pupils, bold thick black "
            "outlines, flat saturated colors, characteristic overbite, "
            "simple cartoon proportions, and 1990s Springfield aesthetic"
        ),
        "negative": (
            "realistic, 3d render, photo, soft shading, gradients, complex "
            "details, deformed, blurry, watermark, text"
        ),
        "guidance": 4.5,
        "strength": 0.65,
    },

    "ghibli": {
        # Ghibli simplifies features (expressive but simple) — would
        # conflict with "exact facial features". Same "facial identity"
        # anchor as pixar/anime. (v0.3.4 review HIGH-1.)
        "prompt": (
            "Restyle this person as a Studio Ghibli character in Hayao "
            "Miyazaki's style, while preserving the facial identity, "
            "hairstyle, body proportions, and pose, with soft "
            "watercolor textures, gentle pastel colors, hand-drawn 2D "
            "animation, expressive but simple features, dreamy "
            "atmosphere, and painterly background"
        ),
        "negative": (
            "photorealistic, 3d render, harsh lighting, sharp edges, deformed, "
            "blurry, low quality, watermark, text"
        ),
        "guidance": 3.5,
        "strength": 0.55,
    },

    "vangogh": {
        "prompt": (
            "Restyle this person as a Vincent Van Gogh oil painting "
            "subject, while preserving the exact facial features, "
            "hairstyle, body proportions, and pose, with thick visible "
            "impasto brushstrokes, swirling textured patterns, vibrant "
            "post-impressionist colors, painterly distortion, and "
            "expressive yellows and blues"
        ),
        "negative": (
            "smooth, flat, photo, 3d render, digital art, clean lines, "
            "deformed face, blurry, watermark, text"
        ),
        "guidance": 4.0,
        "strength": 0.55,
    },

    "pencil": {
        "prompt": (
            "Restyle this person as a detailed graphite pencil sketch "
            "portrait, while preserving the exact facial features, "
            "hairstyle, body proportions, and pose, with fine "
            "cross-hatching, careful shading gradations, monochrome "
            "grayscale, realistic drawing on paper texture, and "
            "hand-drawn precision"
        ),
        "negative": (
            "colorful, painting, 3d render, photo, smooth gradients, deformed, "
            "blurry, watermark, text"
        ),
        "guidance": 3.5,
        "strength": 0.50,
    },
}


# Backwards-compatible alias. Points at the built-in dict only — DO NOT
# read from this in code that needs to see user styles. Use get_style()
# / list_styles() instead, which transparently include user TOMLs from
# ~/.imgen/styles.d/. Kept so the existing test_styles.py and any
# downstream code expecting `STYLES` keeps working — those callers only
# care about the built-in set.
STYLES: dict[str, dict] = BUILTIN_STYLES


class UserStyleError(Exception):
    """Raised when a user TOML in ~/.imgen/styles.d/ has bad shape/values."""


# ── Field validators ─────────────────────────────────────────────────────

def _is_number_not_bool(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_USER_STYLE_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "prompt": ("string", lambda v: isinstance(v, str) and v.strip() != ""),
    "negative": ("string", lambda v: isinstance(v, str)),
    "guidance": (
        "number 0.5..15.0",
        lambda v: _is_number_not_bool(v) and 0.5 <= v <= 15.0,
    ),
    "strength": (
        "number 0.0..1.0",
        lambda v: _is_number_not_bool(v) and 0.0 <= v <= 1.0,
    ),
}


def load_user_style_file(path: Path) -> dict[str, Any]:
    """Parse one .toml file into a preset dict.

    All fields are OPTIONAL — a TOML with only `guidance = 4.0` is a valid
    "param-only" preset. cmd_generate checks at use time whether the
    selected style has a prompt or whether the user supplied
    --custom-prompt to fill the gap.

    Unknown fields are dropped with a warn (forward-compat with future
    schema additions). Known fields with bad values raise UserStyleError.
    """
    # Local import to avoid the styles → colors → … cycle risk
    from .colors import warn

    try:
        size = path.stat().st_size
    except OSError as e:
        raise UserStyleError(f"{path}: {e}") from e
    if size > USER_STYLE_MAX_BYTES:
        raise UserStyleError(
            f"{path}: too large ({size} bytes; cap {USER_STYLE_MAX_BYTES})"
        )

    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise UserStyleError(f"{path}: {e}") from e

    validated: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _USER_STYLE_SCHEMA:
            warn(f"{path}: unknown field '{key}' — ignored")
            continue
        expected_desc, predicate = _USER_STYLE_SCHEMA[key]
        if not predicate(value):
            raise UserStyleError(
                f"{path}: {key}: expected {expected_desc}, got {value!r}"
            )
        validated[key] = value
    return validated


def _is_safe_stem(stem: str) -> bool:
    """Reject C0 controls, DEL, and C1 controls in a style filename stem.

    macOS APFS allows these bytes in filenames; if they end up as style
    names they ride into BatchLogger.write_header / _print_batch_summary
    output, surviving in logs and stdout where they can clear screens,
    inject window-title escapes, or otherwise mess up the user's terminal
    when they later `cat ~/.imgen/logs/<id>.log`.

    Three ranges blocked:
      * ``c < ' '``          — C0 controls 0x00–0x1F (NUL, BEL, ESC, ...)
      * ``c == '\\x7f'``     — DEL
      * ``'\\x80' <= c <= '\\x9f'`` — C1 controls; on 8-bit ECMA-48
        terminals 0x9B alone acts as CSI (= `ESC[`), so a filename
        like ``evil\\x9b[2Jname.toml`` could clear screen even without
        a leading ESC. macOS Terminal.app + iTerm2 default UTF-8 mode
        renders these as replacement chars, but defence-in-depth.

    Initial filter landed in v0.2.4 closure (security N3); C1 range
    added in v0.2.5 review (python NIT-2 / security NIT-1).
    """
    return not any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in stem
    )


def load_user_styles_dir(dir_path: Path) -> dict[str, dict]:
    """Scan a directory for `*.toml` files; return {filename_stem: preset}.

    Files are processed in alphabetical filename order so the conflict-
    resolution suffixes are deterministic. A single bad file warns but
    doesn't kill the rest of the load. Stems containing C0 controls
    (0x00–0x1F) or DEL (0x7F) are rejected for safety — see
    `_is_safe_stem`.
    """
    from .colors import warn

    if not dir_path.exists() or not dir_path.is_dir():
        return {}
    result: dict[str, dict] = {}
    for path in sorted(dir_path.iterdir()):
        if path.suffix != ".toml" or not path.is_file():
            continue
        if not _is_safe_stem(path.stem):
            # Show the printable-repr so the warn() itself doesn't
            # propagate the escape into the user's terminal.
            warn(f"Skipping {path.name!r}: control bytes in filename "
                 "(unsafe to use as a style name)")
            continue
        try:
            result[path.stem] = load_user_style_file(path)
        except UserStyleError as e:
            warn(f"Skipping {path.name}: {e}")
            continue
    return result


_SUFFIX_RE = re.compile(r"_\d{4}$")


def _strip_auto_suffix(name: str) -> str:
    """Drop a trailing `_NNNN` (4-digit) so re-suffixing produces clean
    `anime_0002` rather than `anime_0001_0001` when a user file already
    happens to be named `anime_0001`."""
    return _SUFFIX_RE.sub("", name)


def _find_free_suffix(base: str, taken: dict) -> str:
    """Return base + `_NNNN` for smallest N >= 1 such that the result is
    not already a key in `taken`."""
    n = 1
    while f"{base}_{n:04d}" in taken:
        n += 1
    return f"{base}_{n:04d}"


def merge_user_styles(
    builtins: dict[str, dict],
    user: dict[str, dict],
) -> dict[str, dict]:
    """Combine built-in styles with user styles. Built-in names always win.

    A user-style whose desired name clashes with an existing entry gets
    renamed to `<name>_NNNN` (4-digit zero-padded counter). The built-in
    or earlier user style with that name stays accessible under its
    original name.

    Does NOT mutate either input.
    """
    from .colors import warn

    merged: dict[str, dict] = dict(builtins)
    for name, preset in user.items():
        if name not in merged:
            merged[name] = preset
            continue
        # Strip any existing _NNNN before re-suffixing so we don't stack:
        # anime_0001 → anime_0002, not anime_0001_0001.
        base = _strip_auto_suffix(name)
        new_name = _find_free_suffix(base, merged)
        warn(
            f"styles.d: '{name}' already taken (built-in or earlier user file), "
            f"registered as '{new_name}'"
        )
        merged[new_name] = preset
    return merged


# ── Public accessors (cached merge of built-ins + user styles) ───────────

_cached_merged: dict[str, dict] | None = None


def _load_merged_styles() -> dict[str, dict]:
    """Lazy-merge built-ins + ~/.imgen/styles.d/. Cached per process."""
    global _cached_merged
    if _cached_merged is None:
        # Local import to avoid module-load circularity with paths.py
        from .paths import STATE_DIR
        user = load_user_styles_dir(STATE_DIR / "styles.d")
        _cached_merged = merge_user_styles(BUILTIN_STYLES, user)
    return _cached_merged


def list_styles() -> list[str]:
    """Return sorted list of available style keys (built-in + user)."""
    return sorted(_load_merged_styles().keys())


def get_style(name: str) -> dict:
    """Return preset dict by name. Raises StyleNotFound (KeyError) if unknown."""
    merged = _load_merged_styles()
    if name not in merged:
        available = ", ".join(sorted(merged.keys()))
        raise StyleNotFound(f"Unknown style '{name}'. Available: {available}")
    return merged[name]


def parse_style_list(value: str) -> list[str]:
    """Parse the `--style anime,ghibli,pixar` argument into a deduped list.

    Behaviour (locked by tests):
      - Comma is the separator. Whitespace around items is stripped.
      - Empty items (`",,"`, trailing comma, all-whitespace) raise ValueError.
      - Each item must match a known style — unknown names raise ValueError
        listing the offending names plus the known set (so the user can
        fix the typo without re-running with `--list-styles`).
      - Duplicates are silently dropped, **stable** (first occurrence wins),
        with a one-line warn. `anime,ghibli,anime` → `["anime","ghibli"]`.
      - Order is preserved — `--style ghibli,anime` runs ghibli first.

    Raises ValueError on any bad input. argparse `type=` converts that
    to a user-facing error; direct callers can `die()` on it.
    """
    if not isinstance(value, str):
        raise ValueError(f"--style: expected string, got {type(value).__name__}")

    items = [item.strip() for item in value.split(",")]
    if any(item == "" for item in items):
        raise ValueError(
            "--style: empty name (check for stray commas or whitespace-only items)"
        )

    known = list_styles()
    unknown = [item for item in items if item not in known]
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise ValueError(
            f"--style: unknown name{plural}: {', '.join(unknown)}. "
            f"Known: {', '.join(known)}"
        )

    seen: set[str] = set()
    deduped: list[str] = []
    dropped: list[str] = []
    for item in items:
        if item in seen:
            dropped.append(item)
            continue
        seen.add(item)
        deduped.append(item)

    if dropped:
        # Local import dodges the styles → colors → config → styles cycle —
        # config validation calls list_styles() during load.
        from .colors import warn
        plural = "s" if len(set(dropped)) > 1 else ""
        warn(f"--style: duplicate name{plural} dropped: "
             f"{', '.join(sorted(set(dropped)))}")

    return deduped


def reset_styles_cache() -> None:
    """Drop the cached merge of built-in + user TOMLs.

    Tests use this between cases that touch the on-disk styles.d/
    contents. Future `imgen serve`-style long-lived processes can call
    this from a file-watcher to pick up user-added presets without a
    restart. CLI is single-threaded, so no lock needed.
    """
    global _cached_merged
    _cached_merged = None

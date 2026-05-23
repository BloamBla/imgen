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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._schema import validate_against_schema

# Forward-declared so loader can warn without a circular import on .colors.
# Resolved below the BUILTIN_STYLES dict.


__all__ = [
    "BUILTIN_STYLES",
    "LoraRef",
    "STYLES",
    "Style",
    "USER_STYLE_MAX_BYTES",
    "StyleNotFound",
    "UserStyleError",
    "get_style",
    "list_styles",
    "load_user_style_file",
    "load_user_styles_dir",
    "merge_user_styles",
    "parse_lora_refs",
    "parse_style_list",
    "reset_styles_cache",
]


@dataclass(frozen=True, slots=True)
class Style:
    """v0.6.2 (architect I-2): promotion of the v0.1.x ``dict[str, Any]``
    preset shape into a frozen+slots dataclass. Six fields cover the
    full surface of every built-in style as of v0.6.1 plus the v0.6
    LoRA stack:

      * ``prompt`` — the base prompt sent to mflux (with optional scope
        substitution + LoRA trigger prepend + v0.5 LLM enhancement).
        Can be ``None`` for param-only user styles that lean on
        ``--custom-prompt`` to provide the prompt text.
      * ``negative`` — negative prompt; empty string when unused.
      * ``guidance`` — Kontext / Qwen-Edit CFG guidance scale.
        ``None`` falls back to ``merged_defaults["guidance"]``.
      * ``strength`` — image strength (img2img-style how-much-of-the-
        source-to-keep dial). ``None`` falls back to merged_defaults.
      * ``scene_suffix`` — v0.5 per-style background directive used
        when ``--scope scene``; ``None`` falls back to
        ``SCOPE_SCENE_SUFFIX_GENERIC`` from ``images.py``.
      * ``loras`` — v0.6 LoRA stack (frozen tuple). Empty tuple = no
        LoRA = identical to v0.5 behaviour. ``parse_lora_refs``
        validates user-supplied entries before they land here.

    Dict-compat API kept for v0.6.2 migration: ``preset.get("prompt")``
    and ``preset["prompt"]`` keep working alongside attribute access
    (``preset.prompt``). The hybrid surface lets legacy test code and
    legacy ``cmd_helpers.build_iterations`` call sites carry on
    unchanged while new code prefers attribute access. Future cleanup
    can shed the dict API once all consumers move to attributes.

    Frozen + slots matches the project's other config dataclasses
    (Iteration, BatchContext, EnhanceResult, LoraRef, Backend) — cheap
    in memory, hash/equal by value, immutable so callers can store
    them in caches without fear of mutation.
    """
    prompt: str | None = None
    negative: str = ""
    guidance: float | None = None
    strength: float | None = None
    scene_suffix: str | None = None
    loras: tuple = ()  # tuple[LoraRef, ...] — forward ref dodges class-body ref

    # Dict-compat read surface. Read-only by virtue of frozen=True; the
    # underlying fields cannot be set, so __setitem__ would always raise.
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return hasattr(self, key) and not key.startswith("_")

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass(frozen=True, slots=True)
class LoraRef:
    """One LoRA weight delta to apply on top of the base diffusion model.

    Style presets carry a tuple of these. The mflux argv-builder turns
    each into a (``--lora-paths <ref>``, ``--lora-scales <weight>``)
    pair, filtered by ``compatible_with`` matching the active backend's
    ``lora_compat_group``.

    Fields:
      * ``ref`` — either a HuggingFace repo id (e.g.
        ``"strangerzonehf/Flux-Animeo-v1-LoRA"``) or an absolute path
        to a local ``.safetensors`` file. mflux's ``--lora-paths``
        accepts both shapes.
      * ``weight`` — scalar multiplier passed via ``--lora-scales``.
        1.0 = full strength; lower = subtler effect; higher = overshoot
        (rarely useful, often produces artifacts).
      * ``compatible_with`` — set of backend ``lora_compat_group``
        values this LoRA was trained against. A FLUX.1-dev LoRA also
        works on FLUX.1-Kontext-dev (same architecture family), so the
        common case is ``("flux-1",)``. FLUX.2 LoRAs are a separate
        ecosystem (``"flux-2"``); Qwen LoRAs another (``"qwen"``); etc.
        Mismatched LoRAs are warn-skipped at command-construction time,
        not silently mis-applied.
      * ``trigger`` — optional trigger word/phrase the LoRA was trained
        to activate on. When set, ``build_iterations`` prepends it to
        the prompt if not already present. Many style LoRAs only
        produce their effect when the trigger word appears in the
        prompt (e.g. "Pixar 3D" for Canopus-Pixar-3D-Flux-LoRA).

    Frozen + slots matches the project's other config dataclasses
    (Iteration, BatchContext, EnhanceResult, BackendHealth,
    EnhanceHealth). ``__hash__`` stays default-frozen-hashable since
    every field is hashable — tuples of LoraRef can live in sets if a
    future caller wants dedup.
    """
    ref: str
    weight: float = 1.0
    compatible_with: tuple[str, ...] = ("flux-1",)
    trigger: str | None = None


# Cap on a single LoraRef's ref-string length. HF repo ids are well
# under 200 chars; absolute paths can be longer but 4 KB is more than
# enough headroom for the longest realistic path. Reject anything
# beyond that at schema time — a 1 MB string in a TOML field would
# burn argv space and is almost certainly corruption.
_LORA_REF_MAX_LEN = 4096


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

BUILTIN_STYLES: dict[str, "Style"] = {
    # Pixar restructures facial geometry (rounded features, big eyes are
    # core to the style). "Exact facial features" would contradict the
    # style descriptors below; "facial identity" tells the model to keep
    # WHO it is while letting the style reshape HOW. (v0.3.4 review HIGH-1.)
    # v0.6.0 shipped a Canopus-Pixar-3D-Flux-LoRA built-in; v0.6.1 reverted
    # to text-only after A/B revealed it's FLUX.1-dev base (filename:
    # Canopus-Pixar-3D-FluxDev-LoRA) and crashes mflux Kontext at attention
    # shape mismatch — see [[project-v06x-backlog]].
    "pixar": Style(
        prompt=(
            "Restyle this person as a polished Pixar 3D animated "
            "character, while preserving the facial identity, "
            "hairstyle, body proportions, and pose, with soft volumetric "
            "lighting, smooth rounded features, expressive large eyes, "
            "stylized cartoon proportions, and high-quality CGI rendering"
        ),
        negative=(
            "deformed, blurry, photorealistic skin, flat lighting, missing eye, "
            "extra limbs, distorted face, low quality, artifacts, watermark, text"
        ),
        guidance=3.5,
        strength=0.55,
        # v0.5: scope=scene PRESERVES identity-anchor (it stays in the
        # prompt above) and only ADDS a background-restyling directive
        # tuned for this style's visual school. Pixar backgrounds are
        # painterly 3D environments with cinematic depth-of-field.
        scene_suffix=(
            ", and transform the background and surroundings into a "
            "Pixar-style 3D-animated environment with soft volumetric "
            "lighting, painterly textures, and cinematic depth-of-field"
        ),
    ),

    # Anime enlarges eyes + reshapes face geometry. Same "facial identity"
    # anchor instead of "exact facial features" (v0.3.4 review HIGH-1).
    # v0.6.0 shipped a strangerzonehf/Flux-Animeo-v1-LoRA built-in;
    # v0.6.1 reverted to text-only after A/B revealed the LoRA is
    # FLUX.1-dev base and crashes mflux Kontext at attention shape
    # mismatch (matmul (1,4992,16) vs (64,3072) — Kontext's modified
    # attention doesn't fit the LoRA's rank-16 tensors). All 912 LoRA
    # keys "matched" at load time, but inference exploded on first
    # denoise step.
    "anime": Style(
        prompt=(
            "Restyle this person as a Japanese anime character, while "
            "preserving the facial identity, hairstyle, body "
            "proportions, and pose, with cel-shaded illustration, "
            "expressive large eyes, detailed line art, vibrant colors, "
            "clean shading, and manga aesthetic"
        ),
        negative=(
            "realistic photo, 3d render, deformed face, bad anatomy, extra "
            "limbs, blurry, low quality, watermark, text"
        ),
        guidance=4.0,
        strength=0.60,
        # Anime backgrounds (背景画) are hand-painted detailed scenery —
        # vivid skies, soft cloud shapes, lush illustrated nature.
        # Cel-shaded but rich in painterly detail, not flat.
        scene_suffix=(
            ", and transform the background and surroundings into a "
            "hand-painted anime cel-shaded environment with vibrant "
            "skies, soft cloud shapes, and detailed illustrated scenery"
        ),
    ),

    # Simpsons style fundamentally restructures facial features (yellow
    # skin + large round white eyes), so "exact facial features" would
    # create contradictory instructions. Use "recognizable expression"
    # as the identity anchor — most stylistically-flexible token the
    # model can hold onto while still rebuilding the face.
    "simpsons": Style(
        prompt=(
            "Restyle this person as a Matt Groening Simpsons character, "
            "while preserving the recognizable expression, hairstyle, "
            "body proportions, and pose, with bright yellow skin, large "
            "round white eyes with small black pupils, bold thick black "
            "outlines, flat saturated colors, characteristic overbite, "
            "simple cartoon proportions, and 1990s Springfield aesthetic"
        ),
        negative=(
            "realistic, 3d render, photo, soft shading, gradients, complex "
            "details, deformed, blurry, watermark, text"
        ),
        guidance=4.5,
        strength=0.65,
        # Simpsons backgrounds are deliberately flat — bright saturated
        # color planes, bold outlines, simplified geometric props. NOT
        # painterly. Echoes the style's flat-color aesthetic so the
        # background doesn't compete with the foreground subject.
        scene_suffix=(
            ", and transform the background and surroundings into a "
            "Simpsons-style flat-color cartoon scene with bold black "
            "outlines, simplified geometric shapes, and saturated "
            "1990s Springfield aesthetic"
        ),
    ),

    # Ghibli simplifies features (expressive but simple) — would conflict
    # with "exact facial features". Same "facial identity" anchor as
    # pixar/anime (v0.3.4 review HIGH-1).
    # v0.6: openfree/flux-chatgpt-ghibli-lora — Kontext-compat verified
    # by real inference (the only built-in LoRA that survived v0.6.0
    # → v0.6.1 hotfix). Trigger phrase "Ghibli style" activates the
    # weight delta; auto-prepended if absent. License:
    # flux-1-dev-non-commercial-license (non-commercial only).
    "ghibli": Style(
        prompt=(
            "Restyle this person as a Studio Ghibli character in Hayao "
            "Miyazaki's style, while preserving the facial identity, "
            "hairstyle, body proportions, and pose, with soft "
            "watercolor textures, gentle pastel colors, hand-drawn 2D "
            "animation, expressive but simple features, dreamy "
            "atmosphere, and painterly background"
        ),
        negative=(
            "photorealistic, 3d render, harsh lighting, sharp edges, deformed, "
            "blurry, low quality, watermark, text"
        ),
        guidance=3.5,
        strength=0.55,
        # Ghibli's signature is lush watercolor environments with soft
        # pastels, dramatic skies, lush nature. Atmospheric haze is part
        # of the brand — Miyazaki's painters layer warm light diffusion
        # over the whole scene.
        scene_suffix=(
            ", and transform the background and surroundings into a "
            "Studio Ghibli watercolor environment with soft pastel "
            "skies, lush painterly nature, and warm atmospheric haze"
        ),
        loras=(
            LoraRef(
                ref="openfree/flux-chatgpt-ghibli-lora",
                weight=0.8,
                compatible_with=("flux-1",),
                trigger="Ghibli style",
            ),
        ),
    ),

    "vangogh": Style(
        prompt=(
            "Restyle this person as a Vincent Van Gogh oil painting "
            "subject, while preserving the exact facial features, "
            "hairstyle, body proportions, and pose, with thick visible "
            "impasto brushstrokes, swirling textured patterns, vibrant "
            "post-impressionist colors, painterly distortion, and "
            "expressive yellows and blues"
        ),
        negative=(
            "smooth, flat, photo, 3d render, digital art, clean lines, "
            "deformed face, blurry, watermark, text"
        ),
        guidance=4.0,
        strength=0.55,
        # Van Gogh's backgrounds are the SAME oil paint as the subject —
        # swirling impasto skies (Starry Night), textured fields, bold
        # complementary colors.
        scene_suffix=(
            ", and transform the background and surroundings as a Van "
            "Gogh oil painting with thick visible impasto brushstrokes, "
            "swirling textured skies, and bold post-impressionist colors"
        ),
    ),

    "pencil": Style(
        prompt=(
            "Restyle this person as a detailed graphite pencil sketch "
            "portrait, while preserving the exact facial features, "
            "hairstyle, body proportions, and pose, with fine "
            "cross-hatching, careful shading gradations, monochrome "
            "grayscale, realistic drawing on paper texture, and "
            "hand-drawn precision"
        ),
        negative=(
            "colorful, painting, 3d render, photo, smooth gradients, deformed, "
            "blurry, watermark, text"
        ),
        guidance=3.5,
        strength=0.50,
        # Pencil is "render", not "transform" — it's a drawing technique.
        # Background gets the same monochrome graphite treatment.
        scene_suffix=(
            ", and render the background and surroundings as a "
            "detailed graphite pencil sketch with fine cross-hatching, "
            "varied tonal density, and visible paper texture"
        ),
    ),
}


# Backwards-compatible alias. Points at the built-in dict only — DO NOT
# read from this in code that needs to see user styles. Use get_style()
# / list_styles() instead, which transparently include user TOMLs from
# ~/.imgen/styles.d/. Kept so the existing test_styles.py and any
# downstream code expecting `STYLES` keeps working — those callers only
# care about the built-in set. v0.6.2: values are now :class:`Style`
# instances; dict-compat read API on Style (``["..."]``/``.get(...)``)
# preserves the v0.1.x test surface.
STYLES: dict[str, "Style"] = BUILTIN_STYLES


class UserStyleError(Exception):
    """Raised when a user TOML in ~/.imgen/styles.d/ has bad shape/values."""


# ── Field validators ─────────────────────────────────────────────────────

def _is_number_not_bool(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_lora_list(v: Any) -> bool:
    """Predicate for the ``loras`` field on user style TOMLs. Must be a
    list of dicts; each dict's individual fields are validated in
    :func:`parse_lora_refs`. The schema-level check only confirms the
    outer shape so an obviously wrong type (string, int, etc.) gets a
    clean error message before parse_lora_refs is reached."""
    if not isinstance(v, list):
        return False
    return all(isinstance(item, dict) for item in v)


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
    # v0.5: optional per-style background directive for scope=scene.
    # When absent, scope=scene falls back to ``SCOPE_SCENE_SUFFIX_GENERIC``
    # from images.py. Authors can tune this to match their style's
    # visual school (watercolor / flat-color / impasto / etc.) — see
    # built-in BUILTIN_STYLES for examples. The value is appended
    # verbatim to the prompt, so include a leading separator (e.g.
    # ``, and transform the background...``). Control-byte filter
    # (C0/DEL/C1) because the suffix flows into the prompt → subprocess
    # argv → per-batch log files → terminal display via dry-run output;
    # an ESC byte here could clear the user's screen on `cat <log>`.
    # Symmetric with the v0.4 `extra_args` defence in backends.
    # (v0.5 security-reviewer IMP-3.)
    "scene_suffix": (
        "string (no control bytes)",
        lambda v: isinstance(v, str) and _is_safe_stem(v),
    ),
    # v0.6: optional list of LoRA weight-delta references. Each entry
    # is a TOML table ([[lora]]) with fields ref / weight /
    # compatible_with / trigger — see :class:`LoraRef`. The per-entry
    # validation happens in :func:`parse_lora_refs` which is called
    # AFTER schema validation has confirmed the outer shape (list of
    # dicts). Per-entry errors raise :class:`UserStyleError` carrying
    # the bad field name for diagnostics.
    "loras": (
        "list of TOML tables (see [[lora]] entries)",
        _is_lora_list,
    ),
}


# Per-field bounds for a single LoRA entry. Reused by parse_lora_refs.
_LORA_REF_FIELD_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    # v0.6 security-reviewer IMP-1: ref values starting with ``-`` are
    # rejected (flag-shaped refs would land on mflux's argv as ``--lora-
    # paths --config /etc/passwd`` and confuse mflux's argparser).
    # Absolute paths (``/...``) are allowed; HF repo ids never start
    # with ``-`` anyway. Symmetric with parser._lora_ref_arg.
    "ref": (
        "non-empty string, <= 4 KB, no control bytes, not flag-shaped "
        "(HF repo id or absolute path)",
        lambda v: (
            isinstance(v, str)
            and v.strip() != ""
            and not v.strip().startswith("-")
            and len(v) <= _LORA_REF_MAX_LEN
            and _is_safe_stem(v)
        ),
    ),
    "weight": (
        "number -2.0..2.0 (1.0 = full strength; rarely useful "
        "outside ~0.3..1.2)",
        lambda v: _is_number_not_bool(v) and -2.0 <= v <= 2.0,
    ),
    "compatible_with": (
        "list of non-empty strings (lora_compat_group names like "
        "'flux-1', 'flux-2', 'qwen')",
        lambda v: (
            isinstance(v, list)
            and len(v) >= 1
            and all(
                isinstance(g, str) and g.strip() != "" and _is_safe_stem(g)
                for g in v
            )
        ),
    ),
    "trigger": (
        "string (no control bytes) or omitted",
        lambda v: isinstance(v, str) and _is_safe_stem(v),
    ),
}


def parse_lora_refs(
    raw_loras: list[dict],
    source: Path | str,
) -> tuple[LoraRef, ...]:
    """Validate + convert the ``loras`` field's list-of-dicts shape into
    a tuple of :class:`LoraRef` instances.

    Each entry's ``ref`` is required; every other field has a sensible
    default (``weight=1.0``, ``compatible_with=("flux-1",)``,
    ``trigger=None``). Per-entry validation uses the same (desc,
    predicate) pattern as the top-level schema; bad values raise
    :class:`UserStyleError` carrying the source + entry index for
    diagnostics.

    Returns a tuple (matches :class:`LoraRef`'s frozen-dataclass
    expectations); the caller stores it on the style preset dict.
    """
    result: list[LoraRef] = []
    for idx, entry in enumerate(raw_loras):
        if "ref" not in entry:
            raise UserStyleError(
                f"{source}: loras[{idx}] missing required field 'ref'"
            )
        # Validate every present field against the per-field schema.
        # Use the existing _schema helper so error messages match the
        # style of the rest of the styles.py validation surface.
        validated = validate_against_schema(
            entry, _LORA_REF_FIELD_SCHEMA, UserStyleError,
            source=f"{source} loras[{idx}]",
        )
        # Defaults for omitted optional fields.
        weight = validated.get("weight", 1.0)
        # Dedupe compatible_with preserving stable order (first
        # occurrence wins). `dict.fromkeys` is Python 3.7+ insertion-
        # ordered; matches `parse_style_list` dedupe semantics.
        # (v0.6.x backlog python NIT-3.)
        compat = tuple(dict.fromkeys(
            validated.get("compatible_with", ("flux-1",))
        ))
        trigger = validated.get("trigger")
        result.append(LoraRef(
            ref=validated["ref"],
            weight=float(weight),
            compatible_with=compat,
            trigger=trigger,
        ))
    return tuple(result)


def load_user_style_file(path: Path) -> "Style":
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

    validated = validate_against_schema(
        raw, _USER_STYLE_SCHEMA, UserStyleError, source=str(path),
    )
    # v0.6: post-process the raw ``loras`` list-of-dicts into a tuple of
    # :class:`LoraRef` instances. The schema only validates the outer
    # shape (list of dicts) so per-entry field errors carry a more
    # specific "loras[N].field" diagnostic location.
    if "loras" in validated:
        validated["loras"] = parse_lora_refs(validated["loras"], path)
    # v0.6.2 (architect I-2): assemble the validated mapping into a
    # :class:`Style` instance with only the keys Style knows about. Any
    # unknown key was already filtered by ``validate_against_schema`` so
    # this is just a slot-by-slot construction.
    style_fields = {
        f: validated[f]
        for f in ("prompt", "negative", "guidance", "strength",
                  "scene_suffix", "loras")
        if f in validated
    }
    return Style(**style_fields)


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


def load_user_styles_dir(dir_path: Path) -> dict[str, "Style"]:
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
    result: dict[str, "Style"] = {}
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
    builtins: dict[str, "Style"],
    user: dict[str, "Style"],
) -> dict[str, "Style"]:
    """Combine built-in styles with user styles. Built-in names always win.

    A user-style whose desired name clashes with an existing entry gets
    renamed to `<name>_NNNN` (4-digit zero-padded counter). The built-in
    or earlier user style with that name stays accessible under its
    original name.

    Does NOT mutate either input.
    """
    from .colors import warn

    merged: dict[str, "Style"] = dict(builtins)
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

_cached_merged: dict[str, "Style"] | None = None


def _load_merged_styles() -> dict[str, "Style"]:
    """Lazy-merge built-ins + ~/.imgen/styles.d/. Cached per process."""
    global _cached_merged
    if _cached_merged is None:
        # Local import to avoid module-load circularity with paths.py
        from .paths import STYLES_D
        _cached_merged = merge_user_styles(
            BUILTIN_STYLES, load_user_styles_dir(STYLES_D)
        )
    return _cached_merged


def list_styles() -> list[str]:
    """Return sorted list of available style keys (built-in + user)."""
    return sorted(_load_merged_styles().keys())


def get_style(name: str) -> "Style":
    """Return :class:`Style` instance by name. Raises StyleNotFound (a
    KeyError subclass) when ``name`` is not in the merged registry.

    v0.6.2 (architect I-2): return type changed from ``dict`` to
    :class:`Style`. Dict-compat read API on Style preserves the
    ``preset.get("prompt")`` / ``preset["prompt"]`` call shape for the
    duration of the migration.
    """
    merged = _load_merged_styles()
    if name not in merged:
        available = ", ".join(sorted(merged.keys()))
        raise StyleNotFound(f"Unknown style '{name}'. Available: {available}")
    return merged[name]


# ── Import-time invariant: every built-in style's LoRA stack must round-
# trip through the same validation user styles go through (v0.6.x backlog
# python NIT-7). Catches future hand-written BUILTIN_STYLES picks that
# bypass parse_lora_refs (e.g. a typo'd compat group or weight out of
# range). Runs exactly once per process at import time — cheap.

def _validate_builtin_loras() -> None:
    """Walk every BUILTIN_STYLES preset's loras tuple back through
    :func:`parse_lora_refs` to verify the same constraints user TOMLs
    must satisfy. Raises AssertionError on any violation so the test
    suite surfaces the regression at collection time, not via an
    obscure runtime crash inside cmd_helpers.build_iterations.
    """
    for name, style in BUILTIN_STYLES.items():
        loras = style.loras
        if not loras:
            continue
        as_dicts = [
            {
                "ref": lora.ref,
                "weight": lora.weight,
                "compatible_with": list(lora.compatible_with),
                **({"trigger": lora.trigger} if lora.trigger is not None else {}),
            }
            for lora in loras
        ]
        try:
            roundtrip = parse_lora_refs(as_dicts, f"BUILTIN_STYLES[{name!r}]")
        except UserStyleError as e:  # pragma: no cover — import-time invariant
            raise AssertionError(
                f"BUILTIN_STYLES[{name!r}].loras failed parse_lora_refs "
                f"validation: {e}"
            ) from e
        assert roundtrip == loras, (
            f"BUILTIN_STYLES[{name!r}].loras did not round-trip through "
            f"parse_lora_refs (schema drift?): "
            f"input={loras} roundtrip={roundtrip}"
        )


_validate_builtin_loras()


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

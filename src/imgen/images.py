"""Resolution mapping + scope-modifier for prompts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .colors import warn
from .paths import VENV_BIN

__all__ = [
    "PREVIEW_RESOLUTIONS",
    "RESOLUTIONS",
    "SCOPE_PERSON_SUFFIX",
    "SCOPE_SCENE_SUFFIX_GENERIC",
    "apply_scope",
    "detect_resolution",
]

# FLUX-friendly resolutions (close to 1024² total pixels, aspect-aware)
RESOLUTIONS = [
    # (w, h, aspect)
    (1024, 1024, 1.00),
    (1152, 896, 1.29),
    (896, 1152, 0.78),
    (1216, 832, 1.46),
    (832, 1216, 0.68),
    (1280, 768, 1.67),
    (768, 1280, 0.60),
    (1344, 704, 1.91),
    (704, 1344, 0.52),
]

# Smaller resolutions for --preview mode (~768² pixels, multiples of 64)
PREVIEW_RESOLUTIONS = [
    (768, 768, 1.00),
    (896, 640, 1.40),
    (640, 896, 0.71),
    (960, 576, 1.67),
    (576, 960, 0.60),
]

# --scope: how to modify the prompt
#
# v0.5 redesign — both scopes preserve identity-anchor unconditionally;
# scope only governs background treatment. Previously (v0.3.2–v0.4.0)
# scope=scene actively REWROTE identity-anchor language into composition-
# anchor language via ``SCOPE_SCENE_REPLACEMENTS``, which silently broke
# face preservation on every default invocation (scope=scene is the
# default). With v0.5 the identity-anchor stays in the style preset
# unchanged for BOTH scopes; ``apply_scope`` just appends a background-
# directive suffix:
#
#   * scope=person → identity preserved + background unchanged
#   * scope=scene  → identity preserved + background also restyled
#                    (per-style ``scene_suffix`` field if present, else
#                    ``SCOPE_SCENE_SUFFIX_GENERIC`` fallback)
#
# The per-style suffix lives in ``BUILTIN_STYLES[name]["scene_suffix"]``
# so each style's background gets a treatment tuned to its visual
# school (Pixar 3D-painterly, Simpsons flat-color, Ghibli watercolor,
# etc.). User styles in ``~/.imgen/styles.d/*.toml`` may declare their
# own ``scene_suffix``; absent that field, they fall to the generic.

# Person mode: identity preserved (from the style preset itself) + a
# directive to leave the background untouched. Same wording v0.3.x
# established — unchanged in v0.5.
SCOPE_PERSON_SUFFIX = ", keep the background photorealistic and unchanged"

# Generic background-restyling directive for scope=scene. Used when the
# selected style doesn't carry an explicit ``scene_suffix`` field — i.e.
# user styles in ``~/.imgen/styles.d/*.toml`` that pre-date v0.5 or
# don't bother tuning per-style background language. Built-in styles
# ALL carry their own ``scene_suffix`` tuned to their visual school,
# so the generic is a fallback path, not the primary one.
SCOPE_SCENE_SUFFIX_GENERIC = (
    ", and transform the background and surroundings to match the "
    "same artistic style as the subject"
)


def detect_resolution(image_path: Path, preview: bool = False) -> tuple[int, int]:
    """Return best (width, height) matching source aspect ratio.

    Uses the venv's Python (which has Pillow via mflux deps) so the launcher
    doesn't require a separate system-Python Pillow.
    """
    fallback = (768, 768) if preview else (1024, 1024)
    py = VENV_BIN / "python"
    if not py.exists():
        warn(f"venv missing — defaulting to {fallback[0]}x{fallback[1]} "
             "(run: imgen setup)")
        return fallback
    try:
        out = subprocess.check_output(
            [str(py), "-c",
             "import sys; from PIL import Image, ImageOps; "
             "i = ImageOps.exif_transpose(Image.open(sys.argv[1])); "
             "print(i.size[0], i.size[1])",
             str(image_path)],
            stderr=subprocess.DEVNULL, text=True, timeout=15,
        )
        w, h = map(int, out.split())
    except (subprocess.CalledProcessError, ValueError,
            subprocess.TimeoutExpired) as e:
        warn(f"Couldn't read image dimensions ({e}); defaulting to "
             f"{fallback[0]}x{fallback[1]}")
        return fallback

    aspect = w / h
    table = PREVIEW_RESOLUTIONS if preview else RESOLUTIONS
    best = min(table, key=lambda r: abs(r[2] - aspect))
    return best[0], best[1]


def apply_scope(
    prompt: str,
    scope: str | None,
    scene_suffix: str | None = None,
) -> str:
    """Append the scope-specific background directive to ``prompt``.

    v0.5: both scopes leave the prompt's identity-anchor language
    (e.g. ``while preserving the facial identity, hairstyle, body
    proportions, and pose``) untouched. Scope only governs what
    happens with the background:

    * ``scope="person"`` → ``prompt + SCOPE_PERSON_SUFFIX``
      ("keep the background photorealistic and unchanged")
    * ``scope="scene"`` → ``prompt + (scene_suffix or
      SCOPE_SCENE_SUFFIX_GENERIC)``. Built-in styles carry their own
      ``scene_suffix`` tuned to the visual school (Pixar 3D-painterly,
      Simpsons flat-color, Ghibli watercolor, etc.); user styles in
      ``styles.d/*.toml`` either declare their own or fall back to
      the generic ``match the same artistic style`` directive.
    * Any other / None scope → ``prompt`` unchanged.

    The v0.3.2–v0.4.0 ``SCOPE_SCENE_REPLACEMENTS`` substring-rewrite
    table is gone. That table actively STRIPPED identity-anchor
    language from scope=scene prompts (the most common default
    invocation) and replaced it with composition-anchor language,
    silently losing face preservation on every person-photo run.
    Whether the photo contains a person or not, keeping the
    identity-anchor in the prompt costs nothing for landscape inputs
    (FLUX Kontext's image-conditioning ignores person-directives
    when no person is on the input) and gains everything for
    person-photo inputs.

    Pure function — no I/O, no mutation. ``scene_suffix`` is read
    from the active style preset by the caller (``build_iterations``)
    and threaded through here as a string.
    """
    if scope == "person":
        return prompt + SCOPE_PERSON_SUFFIX
    if scope == "scene":
        return prompt + (scene_suffix or SCOPE_SCENE_SUFFIX_GENERIC)
    return prompt

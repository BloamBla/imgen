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
    "SCOPE_SCENE_REPLACEMENTS",
    "SCOPE_SCENE_SUFFIX",
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
SCOPE_PERSON_SUFFIX = ", keep the background photorealistic and unchanged"

# Targeted substring rewrites for prompts that follow the v0.1.x built-in
# wording convention ("Transform this person ... keep face identity, keep
# pose"). When a prompt contains the trigger substrings, these in-place
# rewrites surgically reframe person-focused wording into scene-focused
# wording without growing the prompt.
SCOPE_SCENE_REPLACEMENTS = [
    ("this person", "this entire scene"),
    ("the person", "the whole scene"),
    ("keep face identity", "keep all subjects recognizable"),
    ("keep pose and composition", "keep overall composition"),
    ("keep pose", "keep composition"),
]

# Fallback append for scene-mode when no SCOPE_SCENE_REPLACEMENTS trigger
# matched — i.e. a user-supplied style in ``~/.imgen/styles.d/*.toml`` or
# a drifted built-in whose wording no longer hits the table. Without this
# the v0.3.2 ``--scope=scene`` default would be a silent no-op on those
# prompts, and the user would get a person-focused image despite asking
# for a scene-wide transform. Symmetric in shape with
# :data:`SCOPE_PERSON_SUFFIX` (leading comma + directive + preservation
# clause). (v0.3.3 — closes the apply_scope fragility flagged as a v0.1.x
# review nit and made more acute by v0.3.2 making scene the default.)
SCOPE_SCENE_SUFFIX = (
    ", transform the entire scene including background and any "
    "additional subjects, keep overall composition"
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


def apply_scope(prompt: str, scope: str | None) -> str:
    """Modify ``prompt`` based on ``scope``: person-only, full scene,
    or unchanged.

    Person mode appends :data:`SCOPE_PERSON_SUFFIX` — works for any
    prompt because it's a pure append, no substring assumption.

    Scene mode is hybrid (v0.3.3):

    1. Try every :data:`SCOPE_SCENE_REPLACEMENTS` substring rewrite in
       order. Built-in presets all follow the v0.1.x wording
       convention and trigger at least one rewrite, so this path
       preserves the targeted v0.1.x-tuned phrasing.
    2. If NO rewrite fired (e.g. a user style in ``styles.d/`` whose
       prompt is structured differently, or a built-in whose wording
       drifted), append :data:`SCOPE_SCENE_SUFFIX` as a fallback
       directive so scene-mode is never a silent no-op.

    The hybrid keeps the precise built-in behaviour but eliminates the
    fragility that became acute when v0.3.2 made scene the default —
    every default invocation now needs scene-mode to actually have an
    effect, no matter the prompt's wording.

    Unknown scope values (anything other than ``"person"``/``"scene"``)
    fall through with no modification, matching argparse's enforcement
    at the CLI boundary.
    """
    if scope == "person":
        return prompt + SCOPE_PERSON_SUFFIX
    if scope == "scene":
        modified = prompt
        applied_any = False
        for old, new in SCOPE_SCENE_REPLACEMENTS:
            if old in modified:
                modified = modified.replace(old, new)
                applied_any = True
        if not applied_any:
            modified = modified + SCOPE_SCENE_SUFFIX
        return modified
    return prompt

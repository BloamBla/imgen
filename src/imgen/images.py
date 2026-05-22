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
SCOPE_SCENE_REPLACEMENTS = [
    ("this person", "this entire scene"),
    ("the person", "the whole scene"),
    ("keep face identity", "keep all subjects recognizable"),
    ("keep pose and composition", "keep overall composition"),
    ("keep pose", "keep composition"),
]


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
    """Modify prompt based on scope: person-only, full scene, or unchanged."""
    if scope == "person":
        return prompt + SCOPE_PERSON_SUFFIX
    if scope == "scene":
        modified = prompt
        for old, new in SCOPE_SCENE_REPLACEMENTS:
            modified = modified.replace(old, new)
        return modified
    return prompt

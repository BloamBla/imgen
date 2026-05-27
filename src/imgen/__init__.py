"""imgen — local image + video generation CLI for Apple Silicon Macs.

Modes as of v0.9.4: text-to-image (``imgen draw``) via FLUX.1-dev,
Hires-Fix upsample (``imgen refine``) via FLUX.2-klein-edit-9b,
photo restyle (``imgen generate`` / ``imgen batch``) via FLUX.1-Kontext
or Qwen-Image-Edit, and text-or-image-to-video (``imgen video``) via
LTX-Video. v0.9.4 is a polish-bundle release (image-arc audit closures
+ pre-tag review fixups) — no user-facing CLI surface change. Image
path on-device via mflux + MLX; video path via HuggingFace
``diffusers`` on MPS in a separate ``.venv-diffusers/`` to avoid the
torch ↔ MLX dependency conflict.
"""

__version__ = "0.9.4"

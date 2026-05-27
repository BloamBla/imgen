"""imgen — local image + video generation CLI for Apple Silicon Macs.

Four modes as of v0.9.0: text-to-image (``imgen draw``) via FLUX.1-dev,
Hires-Fix upsample (``imgen refine``) via FLUX.2-klein-edit-9b,
photo restyle (``imgen generate`` / ``imgen batch``) via FLUX.1-Kontext
or Qwen-Image-Edit, and text-to-video (``imgen video``) via
LTX-Video. Image path on-device via mflux + MLX; video path via
HuggingFace ``diffusers`` on MPS in a separate ``.venv-diffusers/``
to avoid the torch ↔ MLX dependency conflict.
"""

__version__ = "0.9.1"

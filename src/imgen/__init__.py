"""imgen — local image + video generation CLI for Apple Silicon Macs.

Modes as of v0.9.5: text-to-image (``imgen draw``) via FLUX.1-dev,
Hires-Fix upsample (``imgen refine``) via FLUX.2-klein-edit-9b,
photo restyle (``imgen generate`` / ``imgen batch``) via FLUX.1-Kontext
or Qwen-Image-Edit, and text-or-image-to-video (``imgen video``) via
LTX-Video. v0.9.5 is a polish-bundle release closing the 4 architect
deferrals from the v0.9.4 image-arc audit (Engine registry, replay
dispatch table, BatchContext kw_only, scope getattr helper) — no
user-facing CLI surface change. Image path on-device via mflux + MLX;
video path via HuggingFace ``diffusers`` on MPS in a separate
``.venv-diffusers/`` to avoid the torch ↔ MLX dependency conflict.
"""

__version__ = "0.9.5"

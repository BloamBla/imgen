"""imgen — local image + video generation CLI for Apple Silicon Macs.

Modes as of v0.10.0: text-to-image (``imgen draw``) via FLUX.1-dev,
Hires-Fix upsample (``imgen refine``) via FLUX.2-klein-edit-9b,
photo restyle (``imgen generate`` / ``imgen batch``) via FLUX.1-Kontext
or Qwen-Image-Edit, text-or-image-to-video (``imgen video``) via
LTX-Video, and LoRA fine-tuning (``imgen train``) on FLUX.2-klein-4b
via ``mflux-train``. v0.10.0 adds ``imgen train``: a folder of photos
+ a trigger word → a personal LoRA at ``~/.imgen/loras/<name>.safetensors``
that round-trips into ``imgen draw --model flux2-klein-4b --lora <name>``
with the trigger auto-prepended. Validated end-to-end by a real
M2 Pro 32 GB smoke (§M.1). Image + training paths on-device via
mflux + MLX; video path via HuggingFace ``diffusers`` on MPS in a
separate ``.venv-diffusers/`` to avoid the torch ↔ MLX conflict.
"""

__version__ = "0.10.1"

"""imgen — local image generation CLI for Apple Silicon Macs.

Three modes since v0.7.5: text-to-image (`imgen draw`) via FLUX.1-dev,
Hires-Fix upsample (`imgen refine`) via FLUX.2-klein-edit-9b, and
photo restyle (`imgen generate` / `imgen batch`) via FLUX.1-Kontext or
Qwen-Image-Edit. All on-device via mflux + MLX.
"""

__version__ = "0.8.4"

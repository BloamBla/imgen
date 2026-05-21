"""
Style presets for imgen.

Each preset is a fully-formed instruction for FLUX Kontext / Qwen Image Edit
to transform a person photo into a target art style while preserving identity.

Per-style tuning of `guidance` and `strength` is allowed when defaults don't
work well (e.g. Simpsons needs higher guidance to nail the distinctive look).
"""

STYLES = {
    "pixar": {
        "prompt": (
            "Transform this person into a polished Pixar 3D animation style, "
            "soft volumetric lighting, smooth rounded features, expressive large "
            "eyes, stylized cartoon character, high-quality CGI rendering, "
            "keep face identity, keep pose and composition"
        ),
        "negative": (
            "deformed, blurry, photorealistic skin, flat lighting, missing eye, "
            "extra limbs, distorted face, low quality, artifacts, watermark, text"
        ),
        "guidance": 3.5,
        "strength": 0.55,
    },

    "anime": {
        "prompt": (
            "Transform this person into Japanese anime art style, cel-shaded "
            "illustration, expressive large eyes, detailed line art, vibrant "
            "colors, clean shading, manga aesthetic, keep face identity, "
            "keep pose and composition"
        ),
        "negative": (
            "realistic photo, 3d render, deformed face, bad anatomy, extra "
            "limbs, blurry, low quality, watermark, text"
        ),
        "guidance": 4.0,
        "strength": 0.60,
    },

    "simpsons": {
        "prompt": (
            "Transform this person into The Simpsons cartoon style by Matt "
            "Groening, bright yellow skin, large round white eyes with small "
            "black pupils, bold thick black outlines, flat saturated colors, "
            "characteristic overbite, simple cartoon proportions, 1990s "
            "Springfield aesthetic, keep face identity, keep pose"
        ),
        "negative": (
            "realistic, 3d render, photo, soft shading, gradients, complex "
            "details, deformed, blurry, watermark, text"
        ),
        "guidance": 4.5,
        "strength": 0.65,
    },

    "ghibli": {
        "prompt": (
            "Transform this person into Studio Ghibli animation style by Hayao "
            "Miyazaki, soft watercolor textures, gentle pastel colors, "
            "hand-drawn 2D animation, expressive but simple features, dreamy "
            "atmosphere, painterly background, keep face identity, keep pose"
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
            "Transform this person into Vincent Van Gogh oil painting style, "
            "thick visible impasto brushstrokes, swirling textured patterns, "
            "vibrant post-impressionist colors, painterly distortion, expressive "
            "yellows and blues, keep face identity, keep pose and composition"
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
            "Transform this person into a detailed graphite pencil sketch, fine "
            "cross-hatching, careful shading gradations, monochrome grayscale, "
            "realistic drawing on paper texture, hand-drawn precision, keep "
            "face identity, keep pose"
        ),
        "negative": (
            "colorful, painting, 3d render, photo, smooth gradients, deformed, "
            "blurry, watermark, text"
        ),
        "guidance": 3.5,
        "strength": 0.50,
    },
}


def list_styles() -> list[str]:
    """Return sorted list of available style keys."""
    return sorted(STYLES.keys())


def get_style(name: str) -> dict:
    """Return preset dict by name. Raises KeyError if unknown."""
    if name not in STYLES:
        available = ", ".join(list_styles())
        raise KeyError(f"Unknown style '{name}'. Available: {available}")
    return STYLES[name]

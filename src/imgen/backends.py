"""mflux backend registry.

Each Backend captures the variant-specific behavior of an mflux binary:
which executable to call, whether HF token is needed, the spelling of the
image-input flag, and which optional flags it supports. Adding a new
backend = one row in BACKENDS.

`frozen=True` so the registry can't be mutated at runtime (constants
should stay constant). `slots=True` for tighter memory + early errors on
typo'd attribute access.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Backend", "BACKENDS", "build_mflux_cmd"]


@dataclass(frozen=True, slots=True)
class Backend:
    binary: str                  # basename of the mflux entry-point script
    needs_token: bool            # gated HF repo → require ~/.imgen/hf_token / $HF_TOKEN
    image_flag: str              # mflux's input-image flag (--image-path vs --image-paths)
    supports_strength: bool      # accepts --image-strength
    supports_negative: bool      # accepts --negative-prompt
    extra_args: tuple[str, ...]  # fixed flags appended unconditionally (e.g. --model X)
    # v0.4: custom backends (registered via ~/.imgen/backends.d/*.toml)
    # may declare a single env var that imgen forwards from the parent
    # environment into the subprocess. None on built-ins; FLUX keeps
    # using the legacy ``needs_token=True`` path with ~/.imgen/hf_token
    # because that path also owns whoami validation + atomic save —
    # generalizing those is out of scope for v0.4 (see
    # project_v040_design.md, decision 2 + schema migration trap).
    secret_env_var: str | None = None
    # When ``secret_env_var`` is set: die at command-construction time
    # if the env var is missing from os.environ. False means "best
    # effort" — forward if set, silently skip if not, let the backend
    # binary report its own auth failure.
    secret_required: bool = True


BACKENDS: dict[str, Backend] = {
    "flux": Backend(
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
    ),
    "qwen": Backend(
        binary="mflux-generate-qwen-edit",
        needs_token=False,
        image_flag="--image-paths",
        supports_strength=False,
        supports_negative=False,
        extra_args=("--model", "qwen"),
    ),
}


def build_mflux_cmd(
    *,
    binary: Path,
    backend: Backend,
    input_path: Path,
    output_path: Path,
    prompt: str,
    negative: str,
    quantize: int,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
    width: int,
    height: int,
    mlx_cache_gb: int,
    battery_stop: int,
) -> list[str]:
    """Build the mflux argv for `backend` from already-resolved parameters.

    Pure: no I/O, no env reads, no subprocess. Keyword-only because 15
    positional args would be a footgun.

    Order preserved from v0.1.x: common args first, then strength (if
    supported), then `extra_args` (e.g. `--model dev`), then negative
    prompt (if supported and non-empty). Locked in by test_generate_cmd.

    v0.3.2: dropped ``--metadata``. mflux's ``--metadata`` writes a
    ``<output>.metadata.json`` sidecar next to every generated image,
    which clutters the user's gallery folder. The PNG itself still
    gets metadata embedded via mflux's ``_embed_metadata`` +
    ``MetadataBuilder.embed_metadata`` calls (these fire whenever a
    ``metadata`` dict exists, independent of the ``--metadata`` flag);
    the sidecar JSON was duplicate data. We also already store every
    run's params in ``~/.imgen/history.jsonl`` for replay, making the
    sidecars triply redundant.
    """
    cmd = [
        str(binary),
        "--quantize", str(quantize),
        backend.image_flag, str(input_path),
        "--prompt", prompt,
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(seed),
        "--width", str(width),
        "--height", str(height),
        "--mlx-cache-limit-gb", str(mlx_cache_gb),
        "--battery-percentage-stop-limit", str(battery_stop),
        "--output", str(output_path),
    ]
    if backend.supports_strength:
        cmd += ["--image-strength", str(strength)]
    cmd += list(backend.extra_args)
    if backend.supports_negative and negative:
        cmd += ["--negative-prompt", negative]
    return cmd

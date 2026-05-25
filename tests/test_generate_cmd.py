"""build_mflux_cmd — the cmd-construction footgun this exists to prevent.

architect #7: "flux gets --image-path + --image-strength + --negative-prompt,
qwen gets --image-paths and NO strength flag — was footgun-prone split".

These tests lock the exact argv order and the supports_*/extra_args
semantics. If a future Backend field changes the construction order or a
new conditional gets added, these tests must be updated explicitly —
that's the design.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.backends import BACKENDS, Backend, build_mflux_cmd


# ── Test fixture: canonical params for build_mflux_cmd ───────────────

@pytest.fixture
def params():
    """Return a dict of build_mflux_cmd kwargs. Tests override one key
    at a time to keep assertions focused."""
    return dict(
        binary=Path("/venv/bin/mflux-generate-x"),
        model=BACKENDS["flux"],
        input_path=Path("/in/photo.jpg"),
        output_path=Path("/out/photo.png"),
        prompt="anime prompt text",
        negative="negative text",
        quantize=8,
        steps=20,
        guidance=3.5,
        strength=0.55,
        seed=42,
        width=1024,
        height=1024,
        mlx_cache_gb=12,
        battery_stop=20,
    )


# ── Common arg structure ────────────────────────────────────────────

def test_first_arg_is_binary_path(params):
    cmd = build_mflux_cmd(**params)
    assert cmd[0] == "/venv/bin/mflux-generate-x"


def test_returns_list_of_strings(params):
    cmd = build_mflux_cmd(**params)
    assert isinstance(cmd, list)
    assert all(isinstance(x, str) for x in cmd)


@pytest.mark.parametrize("flag,value_provider", [
    ("--quantize", lambda p: str(p["quantize"])),
    ("--prompt", lambda p: p["prompt"]),
    ("--steps", lambda p: str(p["steps"])),
    ("--guidance", lambda p: str(p["guidance"])),
    ("--seed", lambda p: str(p["seed"])),
    ("--width", lambda p: str(p["width"])),
    ("--height", lambda p: str(p["height"])),
    ("--mlx-cache-limit-gb", lambda p: str(p["mlx_cache_gb"])),
    ("--battery-percentage-stop-limit", lambda p: str(p["battery_stop"])),
    ("--output", lambda p: str(p["output_path"])),
])
def test_common_flags_present_with_correct_value(params, flag, value_provider):
    cmd = build_mflux_cmd(**params)
    assert flag in cmd, f"{flag} not in cmd"
    idx = cmd.index(flag)
    assert cmd[idx + 1] == value_provider(params)


def test_metadata_flag_absent(params):
    """v0.3.2: ``--metadata`` removed — mflux otherwise drops a
    ``<output>.metadata.json`` sidecar next to every image, cluttering
    the gallery. PNG-embedded metadata is preserved by mflux regardless
    of this flag (see backends.build_mflux_cmd docstring), so the
    sidecar was duplicate data. We also write run params to
    ``~/.imgen/history.jsonl`` for replay, making the sidecar triply
    redundant."""
    cmd = build_mflux_cmd(**params)
    assert "--metadata" not in cmd


# ── v0.7.0: Optional[Path] input_path for t2i ─────────────────────────

def test_input_path_none_omits_image_flag(params):
    """v0.7.0: t2i path. `input_path=None` (the `imgen draw` shape)
    skips the `backend.image_flag <path>` argv pair entirely. The
    rest of the cmd is unchanged."""
    params["input_path"] = None
    params["model"] = BACKENDS["flux-dev"]
    cmd = build_mflux_cmd(**params)
    # Neither --image-path nor its value lands in argv.
    assert "--image-path" not in cmd
    # Sanity: prompt + output still present.
    assert "--prompt" in cmd
    assert "--output" in cmd


def test_input_path_set_emits_image_flag(params):
    """Symmetric lock-in for the i2i path — the None gate doesn't
    accidentally break the populated case."""
    params["input_path"] = Path("/in/photo.jpg")
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    assert "--image-path" in cmd
    idx = cmd.index("--image-path")
    assert cmd[idx + 1] == "/in/photo.jpg"


def test_flux_dev_t2i_no_strength_no_image(params):
    """Combined t2i case: flux-dev backend + input_path=None →
    no --image-path, no --image-strength (supports_strength=False on
    flux-dev), prompt + output + quantize + steps still land."""
    params["input_path"] = None
    params["model"] = BACKENDS["flux-dev"]
    cmd = build_mflux_cmd(**params)
    assert "--image-path" not in cmd
    assert "--image-strength" not in cmd
    # Core t2i flags still present.
    assert "--prompt" in cmd
    assert "--output" in cmd
    assert "--quantize" in cmd
    assert "--steps" in cmd


# ── FLUX-specific ───────────────────────────────────────────────────

def test_flux_uses_image_path_singular(params):
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    assert "--image-path" in cmd
    assert "--image-paths" not in cmd


def test_flux_includes_image_strength(params):
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    assert "--image-strength" in cmd
    idx = cmd.index("--image-strength")
    assert cmd[idx + 1] == str(params["strength"])


def test_flux_includes_negative_prompt_when_set(params):
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    assert "--negative-prompt" in cmd
    idx = cmd.index("--negative-prompt")
    assert cmd[idx + 1] == params["negative"]


def test_flux_omits_negative_prompt_when_empty(params):
    params["model"] = BACKENDS["flux"]
    params["negative"] = ""
    cmd = build_mflux_cmd(**params)
    assert "--negative-prompt" not in cmd


def test_flux_appends_model_dev(params):
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "dev"


def test_flux_tail_order_strength_model_negative(params):
    """v0.1.x order: ...--image-strength X --model dev --negative-prompt Y.
    Regression guard against accidentally reordering."""
    params["model"] = BACKENDS["flux"]
    cmd = build_mflux_cmd(**params)
    strength_idx = cmd.index("--image-strength")
    model_idx = cmd.index("--model")
    negative_idx = cmd.index("--negative-prompt")
    assert strength_idx < model_idx < negative_idx


# ── QWEN-specific ───────────────────────────────────────────────────

def test_qwen_uses_image_paths_plural(params):
    params["model"] = BACKENDS["qwen"]
    cmd = build_mflux_cmd(**params)
    assert "--image-paths" in cmd
    assert "--image-path" not in cmd


def test_qwen_omits_image_strength(params):
    """qwen-image-edit doesn't accept --image-strength — passing it would
    error mflux. This is the footgun architect #7 specifically named."""
    params["model"] = BACKENDS["qwen"]
    cmd = build_mflux_cmd(**params)
    assert "--image-strength" not in cmd


def test_qwen_omits_negative_prompt_even_when_set(params):
    """qwen-image-edit doesn't accept --negative-prompt either, even if
    the caller passes a non-empty negative string."""
    params["model"] = BACKENDS["qwen"]
    params["negative"] = "should be ignored"
    cmd = build_mflux_cmd(**params)
    assert "--negative-prompt" not in cmd


def test_flux2_klein_edit_9b_omits_negative_prompt_even_when_set(params):
    """v0.7.11 (gap 7 fix): mflux-generate-flux2-edit errors out with
    ``--negative-prompt is not supported for FLUX.2`` if argv carries
    the flag. FLUX.2 family removed CFG/negative entirely. Even though
    the caller passes a non-empty negative (typically leaked from the
    default ``pixar`` style preset's ``negative_prompt`` field), argv
    must not contain ``--negative-prompt``. Mirrors the qwen guard."""
    params["model"] = BACKENDS["flux2-klein-edit-9b"]
    params["negative"] = "low quality, blurry"
    cmd = build_mflux_cmd(**params)
    assert "--negative-prompt" not in cmd


def test_qwen_appends_model_qwen(params):
    params["model"] = BACKENDS["qwen"]
    cmd = build_mflux_cmd(**params)
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "qwen"


# ── Backend semantics, parametrized ─────────────────────────────────

@pytest.mark.parametrize("backend_name,expected_image_flag", [
    ("flux", "--image-path"),
    ("qwen", "--image-paths"),
])
def test_image_flag_per_backend(params, backend_name, expected_image_flag):
    params["model"] = BACKENDS[backend_name]
    cmd = build_mflux_cmd(**params)
    assert expected_image_flag in cmd


def test_hypothetical_backend_without_strength_omits_it(params):
    """Locks the supports_strength=False contract — adding a future
    backend with no strength support shouldn't get --image-strength."""
    no_strength = Backend(
        binary="hypothetical",
        needs_token=False,
        image_flag="--image-path",
        supports_strength=False,
        supports_negative=True,
        extra_args=("--model", "hypo"),
    )
    params["model"] = no_strength
    cmd = build_mflux_cmd(**params)
    assert "--image-strength" not in cmd
    # Negative is supported AND non-empty → must appear
    assert "--negative-prompt" in cmd

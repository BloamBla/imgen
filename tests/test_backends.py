"""Backend dataclass + BACKENDS registry invariants.

These lock down the v0.2 schema: any future change to a Backend field or
to BACKENDS["flux"]/["qwen"] must update the corresponding assertion,
which forces an explicit decision rather than a silent drift.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from imgen.backends import BACKENDS, Backend


def test_BACKENDS_contains_flux_qwen_flux_dev_flux2():
    """v0.7.0: third built-in backend ``flux-dev`` lands for
    ``imgen draw`` (t2i). v0.7.5: fourth backend ``flux2-klein-edit-9b``
    lands for ``imgen refine`` (Hires-Fix i2i via FLUX.2-klein-9B).
    Built-in set is exactly these four; user TOMLs extend via
    backends.d/ but BUILTIN_BACKENDS stays tight."""
    assert set(BACKENDS.keys()) == {
        "flux", "qwen", "flux-dev", "flux2-klein-edit-9b",
    }


def test_BACKENDS_values_are_Backend_instances():
    for be in BACKENDS.values():
        assert isinstance(be, Backend)


def test_Backend_is_frozen():
    be = BACKENDS["flux"]
    with pytest.raises(FrozenInstanceError):
        be.binary = "hacked"


def test_Backend_rejects_unknown_attribute():
    """Typo'd attribute name (e.g. be.support_strength vs supports_strength)
    must raise instead of silently creating a new field on the instance.
    Exception class varies by Python version + dataclass machinery
    (FrozenInstanceError on existing fields, TypeError via super() on
    slots+frozen for new ones) — just assert it fails loudly."""
    be = BACKENDS["flux"]
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        be.totally_new_field = "x"


def test_flux_backend_config_locked():
    flux = BACKENDS["flux"]
    assert flux.binary == "mflux-generate-kontext"
    assert flux.needs_token is True
    assert flux.image_flag == "--image-path"
    assert flux.supports_strength is True
    assert flux.supports_negative is True
    assert flux.extra_args == ("--model", "dev")


def test_qwen_backend_config_locked():
    qwen = BACKENDS["qwen"]
    assert qwen.binary == "mflux-generate-qwen-edit"
    assert qwen.needs_token is False
    assert qwen.image_flag == "--image-paths"
    assert qwen.supports_strength is False
    assert qwen.supports_negative is False
    assert qwen.extra_args == ("--model", "qwen")


# ── v0.4: secret_env_var / secret_required (custom-backend support) ─────


def test_builtin_backends_have_no_custom_secret():
    """Built-in flux/qwen don't use the custom-backend secret slot.
    FLUX uses the legacy needs_token + ~/.imgen/hf_token path; qwen
    needs no secret at all. (v0.4 design decision 2 — schema migration
    trap deliberately left for v0.5.)"""
    for be in BACKENDS.values():
        assert be.secret_env_var is None
        assert be.secret_required is True  # default; meaningless w/ env_var=None


def test_backend_accepts_secret_env_var():
    """Custom backends can declare a single env var to forward into the
    subprocess. Tuple type ensures immutability mirrors the rest of the
    dataclass."""
    be = Backend(
        binary="mflux-generate-sdxl",
        needs_token=False,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "sdxl"),
        secret_env_var="REPLICATE_API_TOKEN",
        secret_required=True,
    )
    assert be.secret_env_var == "REPLICATE_API_TOKEN"
    assert be.secret_required is True


def test_backend_secret_required_defaults_to_true():
    """Default for the required flag — backends with a declared secret
    want it set or fail loud. False is the explicit opt-in for
    "best-effort forward, no error if missing"."""
    be = Backend(
        binary="thing",
        needs_token=False,
        image_flag="--image-path",
        supports_strength=False,
        supports_negative=False,
        extra_args=(),
        secret_env_var="OPTIONAL_KEY",
    )
    assert be.secret_required is True

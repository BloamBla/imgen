"""Backend dataclass + BACKENDS registry invariants.

These lock down the v0.2 schema: any future change to a Backend field or
to BACKENDS["flux"]/["qwen"] must update the corresponding assertion,
which forces an explicit decision rather than a silent drift.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from imgen.backends import BACKENDS, Backend


def test_BACKENDS_contains_flux_and_qwen():
    assert set(BACKENDS.keys()) == {"flux", "qwen"}


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

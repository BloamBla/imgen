"""Structural invariants for the STYLES registry — every preset must have
a usable shape so cmd_generate can't blow up on a missing key."""
from __future__ import annotations

import pytest

from imgen.styles import STYLES, get_style, list_styles


ALL_STYLES = list(STYLES.keys())


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_has_required_keys(name):
    preset = STYLES[name]
    assert "prompt" in preset, f"{name}: missing 'prompt'"
    assert "negative" in preset, f"{name}: missing 'negative'"
    assert isinstance(preset["prompt"], str)
    assert isinstance(preset["negative"], str)
    assert preset["prompt"].strip(), f"{name}: empty prompt"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_guidance_in_argparse_range(name):
    """Argparse validator allows 0.5..15.0 — preset overrides must be in range."""
    g = STYLES[name].get("guidance")
    if g is not None:
        assert 0.5 <= g <= 15.0, f"{name}: guidance {g} out of [0.5, 15.0]"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_strength_in_argparse_range(name):
    """Argparse validator allows 0.0..1.0 — preset overrides must be in range."""
    s = STYLES[name].get("strength")
    if s is not None:
        assert 0.0 <= s <= 1.0, f"{name}: strength {s} out of [0.0, 1.0]"


def test_list_styles_sorted_and_complete():
    assert list_styles() == sorted(STYLES.keys())


def test_get_style_known_returns_same_object():
    assert get_style("anime") is STYLES["anime"]


def test_get_style_unknown_raises_keyerror_with_hint():
    with pytest.raises(KeyError) as exc_info:
        get_style("nonexistent_style_xyz")
    # Error message should include available styles for cmd_generate hint
    msg = str(exc_info.value)
    assert "nonexistent_style_xyz" in msg
    assert "anime" in msg  # at least one known style listed

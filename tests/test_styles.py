"""Structural invariants for the STYLES registry — every preset must have
a usable shape so cmd_generate can't blow up on a missing key.

Also covers `parse_style_list` (v0.2.3) — the comma-list parser that
backs the future multi-style CLI surface.
"""
from __future__ import annotations

import pytest

from imgen.styles import STYLES, get_style, list_styles, parse_style_list


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


# ── parse_style_list (v0.2.3 plumbing for multi-style CLI) ──────────────

def test_parse_style_list_single_style_returns_one_element_list():
    """`--style anime` keeps v0.2.x single-style behaviour as a 1-element list."""
    assert parse_style_list("anime") == ["anime"]


def test_parse_style_list_multi_style_preserves_order():
    """Order of listed styles is the order of generation in v0.2.3+ multi-style.

    No alphabetical sort — `--style ghibli,anime` runs ghibli first.
    """
    assert parse_style_list("ghibli,anime,pixar") == ["ghibli", "anime", "pixar"]


def test_parse_style_list_strips_whitespace_around_items():
    """`--style 'anime , ghibli'` is forgiving — common copy-paste case."""
    assert parse_style_list("anime , ghibli") == ["anime", "ghibli"]


def test_parse_style_list_dedupes_with_stable_order_and_warn(capsys):
    """Duplicate names dropped; first occurrence wins; user is warned once."""
    result = parse_style_list("anime,ghibli,anime,pixar,ghibli")
    assert result == ["anime", "ghibli", "pixar"]
    out = capsys.readouterr().out + capsys.readouterr().err  # may be on either
    # Re-run to capture (capsys consumed on first read).
    parse_style_list("anime,anime")
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "duplicate" in out.lower()


def test_parse_style_list_empty_string_raises():
    with pytest.raises(ValueError) as exc_info:
        parse_style_list("")
    assert "empty" in str(exc_info.value).lower()


def test_parse_style_list_just_commas_raises():
    """`--style ,,` is unambiguously a typo."""
    with pytest.raises(ValueError) as exc_info:
        parse_style_list(",,")
    assert "empty" in str(exc_info.value).lower()


def test_parse_style_list_trailing_comma_raises():
    """`--style anime,` rejected — comma without a name is a typo."""
    with pytest.raises(ValueError) as exc_info:
        parse_style_list("anime,")
    assert "empty" in str(exc_info.value).lower()


def test_parse_style_list_unknown_name_raises_with_known_list():
    with pytest.raises(ValueError) as exc_info:
        parse_style_list("anime,nonexistent_xyz")
    msg = str(exc_info.value)
    assert "nonexistent_xyz" in msg
    # Known styles are surfaced so the user can fix the typo from the
    # error alone, without running `imgen --list-styles`.
    assert "anime" in msg


def test_parse_style_list_multiple_unknown_names_listed():
    with pytest.raises(ValueError) as exc_info:
        parse_style_list("anime,bogus1,bogus2")
    msg = str(exc_info.value)
    assert "bogus1" in msg
    assert "bogus2" in msg


def test_parse_style_list_known_name_after_dedupe_still_passes():
    """Edge case: `anime,anime,anime` → single 'anime', no spurious unknown error."""
    assert parse_style_list("anime,anime,anime") == ["anime"]


def test_parse_style_list_single_whitespace_item_raises():
    """`--style ' '` is empty after strip, treat as empty."""
    with pytest.raises(ValueError):
        parse_style_list("   ")

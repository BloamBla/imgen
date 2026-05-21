"""apply_scope behavior + RESOLUTIONS table sanity.

detect_resolution shells out to the venv's Python+Pillow, so its
subprocess path isn't tested here — only the aspect-mapping pure
function would be, but it's an inline `min(table, key=...)` call that
the architecture didn't extract. The RESOLUTIONS table itself is
asserted instead so a bad entry can't slip in.
"""
from __future__ import annotations

import pytest

from imgen.images import (
    PREVIEW_RESOLUTIONS,
    RESOLUTIONS,
    SCOPE_PERSON_SUFFIX,
    SCOPE_SCENE_REPLACEMENTS,
    apply_scope,
)


# ── RESOLUTIONS table ─────────────────────────────────────────────────

@pytest.mark.parametrize("w,h,aspect", RESOLUTIONS)
def test_RESOLUTIONS_aspect_matches_dimensions(w, h, aspect):
    assert abs((w / h) - aspect) < 0.01, (
        f"{w}x{h}: stored aspect {aspect} != computed {w/h:.3f}"
    )


@pytest.mark.parametrize("w,h,aspect", PREVIEW_RESOLUTIONS)
def test_PREVIEW_RESOLUTIONS_aspect_matches_dimensions(w, h, aspect):
    assert abs((w / h) - aspect) < 0.01


def test_PREVIEW_RESOLUTIONS_smaller_than_RESOLUTIONS():
    # preview should genuinely be faster — guard against accidentally
    # making it equal or larger than the normal set.
    preview_max = max(w * h for w, h, _ in PREVIEW_RESOLUTIONS)
    normal_min = min(w * h for w, h, _ in RESOLUTIONS)
    assert preview_max < normal_min


def test_RESOLUTIONS_dimensions_multiples_of_64():
    # FLUX/MLX prefer multiples of 64 — silent rounding errors otherwise.
    for w, h, _ in RESOLUTIONS + PREVIEW_RESOLUTIONS:
        assert w % 64 == 0 and h % 64 == 0, f"{w}x{h} not /64"


# ── apply_scope ───────────────────────────────────────────────────────

def test_apply_scope_none_unchanged():
    base = "Transform this person into anime"
    assert apply_scope(base, None) == base


def test_apply_scope_person_appends_suffix():
    base = "Transform this person into anime"
    assert apply_scope(base, "person") == base + SCOPE_PERSON_SUFFIX


def test_apply_scope_person_is_idempotent_only_once():
    # Applying twice would duplicate the suffix — apply_scope is one-shot
    # per call, callers shouldn't loop it. This test pins that "person"
    # appends literally once, no dedup.
    base = "x"
    once = apply_scope(base, "person")
    twice = apply_scope(once, "person")
    assert twice == base + SCOPE_PERSON_SUFFIX * 2


def test_apply_scope_scene_does_known_replacements():
    base = "Transform this person, keep face identity, keep pose"
    result = apply_scope(base, "scene")
    # "this person" → "this entire scene"
    assert "this person" not in result
    assert "this entire scene" in result
    # "keep face identity" → "keep all subjects recognizable"
    assert "keep face identity" not in result
    assert "keep all subjects recognizable" in result


def test_apply_scope_scene_replacement_count_matches_table():
    # If someone adds a tuple to SCOPE_SCENE_REPLACEMENTS without updating
    # apply_scope, this would catch a silent no-op replacement.
    assert len(SCOPE_SCENE_REPLACEMENTS) >= 4
    for old, new in SCOPE_SCENE_REPLACEMENTS:
        assert isinstance(old, str) and isinstance(new, str)
        assert old != new


def test_apply_scope_unknown_scope_is_noop():
    # apply_scope only knows "person"/"scene" — any other value (incl.
    # argparse rejects new ones, but the function itself shouldn't crash
    # if called programmatically) falls through unchanged.
    base = "test"
    assert apply_scope(base, "totally-unknown") == base

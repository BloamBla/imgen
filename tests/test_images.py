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
    SCOPE_SCENE_SUFFIX,
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
    # v0.3.4: table grew to 8 entries (3 new preservation-clause variants
    # for v0.3.4 prompts + 5 legacy v0.1.x triggers kept for back-compat).
    assert len(SCOPE_SCENE_REPLACEMENTS) >= 8
    for old, new in SCOPE_SCENE_REPLACEMENTS:
        assert isinstance(old, str) and isinstance(new, str)
        assert old != new


# ── scene fallback (v0.3.3 — hybrid apply_scope) ──────────────────────


def test_apply_scope_scene_appends_suffix_when_no_trigger_matches():
    """User-supplied styles in ~/.imgen/styles.d/ won't follow the
    v0.1.x built-in wording convention ("this person" / "keep face
    identity" / etc.). v0.3.2 made scope=scene the default, which made
    silent no-ops on those prompts a real bug — user asks for scene
    framing, gets a person-focused prompt unchanged. v0.3.3 fix: when
    no SCOPE_SCENE_REPLACEMENTS trigger matched, append
    SCOPE_SCENE_SUFFIX as a fallback directive."""
    base = "Render the subject as a watercolor painting"
    result = apply_scope(base, "scene")
    assert result == base + SCOPE_SCENE_SUFFIX


def test_apply_scope_scene_does_not_double_apply_suffix_when_triggers_present():
    """Built-in presets (which trigger the substring rewrites) must
    NOT also get the fallback suffix appended — that would be a
    redundant scene-directive on top of the already-rewritten wording."""
    base = "Transform this person, keep face identity, keep pose"
    result = apply_scope(base, "scene")
    # Triggers rewrote in-place; no trailing suffix.
    assert not result.endswith(SCOPE_SCENE_SUFFIX)
    # And the rewrites still fired (sanity).
    assert "this entire scene" in result
    assert "keep all subjects recognizable" in result


def test_apply_scope_scene_fallback_on_empty_prompt():
    """Edge case: empty prompt has no triggers → fallback appends.
    Defensive — shouldn't happen in practice (resolve_prompt rejects
    empty), but apply_scope itself stays defined."""
    assert apply_scope("", "scene") == SCOPE_SCENE_SUFFIX


def test_apply_scope_scene_partial_trigger_match_no_fallback():
    """If even ONE trigger fires, the fallback does NOT — the targeted
    rewrite is sufficient. Lock the boundary against double-treatment."""
    # Prompt has only one of the five triggers ("this person") — others
    # absent. The single rewrite fires; no suffix gets appended.
    base = "Transform this person into watercolor"
    result = apply_scope(base, "scene")
    assert "this entire scene" in result
    assert not result.endswith(SCOPE_SCENE_SUFFIX)


def test_apply_scope_scene_suffix_constant_shape():
    """SCOPE_SCENE_SUFFIX must lead with a comma (so it grafts onto an
    existing prompt cleanly) and be non-trivial."""
    assert SCOPE_SCENE_SUFFIX.startswith(",")
    assert "scene" in SCOPE_SCENE_SUFFIX.lower()
    # Symmetric in shape with SCOPE_PERSON_SUFFIX — both lead with
    # ", " for clean concatenation.
    assert SCOPE_PERSON_SUFFIX.startswith(",")


def test_apply_scope_person_unaffected_by_scene_changes():
    """Hybrid scene logic must not touch the person path — it stays a
    pure suffix append."""
    base = "Render the subject as watercolor"  # no triggers
    result = apply_scope(base, "person")
    assert result == base + SCOPE_PERSON_SUFFIX
    # And no scene-suffix leaked in.
    assert SCOPE_SCENE_SUFFIX not in result


# ── v0.3.4: HIGH-2 — no double-rewrite on v0.3.4 prompts ──────────────


def test_apply_scope_scene_does_not_double_rewrite_v034_subject_opener():
    """v0.3.4 review HIGH-2: pre-fix, the legacy 'this person' →
    'this entire scene' trigger fired on the v0.3.4 opener
    ('Restyle this person as <X>') simultaneously with the new
    preservation-clause trigger, producing grammatically-broken
    output like 'Restyle this entire scene as a Pixar 3D character'.
    Fix: the legacy trigger was anchored to 'Transform this person'
    (v0.1.x form only). v0.3.4 openers stay verbatim under scope=scene;
    only the preservation clause gets relaxed."""
    base = (
        "Restyle this person as a polished Pixar 3D animated character, "
        "while preserving the facial identity, hairstyle, body "
        "proportions, and pose, with cartoon styling"
    )
    result = apply_scope(base, "scene")
    # Opener stays — "Restyle this person as" not corrupted by the
    # legacy subject rewrite.
    assert "Restyle this person as" in result
    assert "this entire scene as" not in result
    # But the preservation clause IS scene-rewritten.
    assert "facial identity" not in result
    assert "overall composition" in result


def test_apply_scope_scene_legacy_transform_pattern_still_works():
    """Back-compat: user styles in ~/.imgen/styles.d/ that still use
    the v0.1.x 'Transform this person ...' wording must continue to
    get the legacy subject rewrite under scope=scene. Anchoring the
    trigger to 'Transform this person' (v0.3.4 HIGH-2 fix) keeps that
    code path live for legacy callers."""
    legacy = "Transform this person into watercolor painting"
    result = apply_scope(legacy, "scene")
    assert "Transform this entire scene" in result
    assert "this person" not in result


def test_apply_scope_scene_facial_identity_variant_rewrites():
    """v0.3.4 pixar/anime/ghibli use 'facial identity' instead of
    'exact facial features' because their styles restructure facial
    geometry (HIGH-1 fix). Scene mode must recognize that variant."""
    base = (
        "Restyle this person as a Pixar character, while preserving "
        "the facial identity, hairstyle, body proportions, and pose, "
        "with cartoon styling"
    )
    result = apply_scope(base, "scene")
    assert "facial identity" not in result
    assert "overall composition" in result


# ── v0.3.4: built-in presets must fire scene-mode rewrites ─────────────


def test_apply_scope_scene_rewrites_v034_preservation_clause():
    """v0.3.4 built-in prompts have a uniform preservation clause:
    "while preserving the exact facial features, hairstyle, body
    proportions, and pose". Scene mode must relax this into a scene-
    wide preservation directive — otherwise person-anchored
    preservation overrides the user's scene-wide intent."""
    base = (
        "Restyle this person as a Pixar 3D character, while preserving "
        "the exact facial features, hairstyle, body proportions, and "
        "pose, with cartoon styling"
    )
    result = apply_scope(base, "scene")
    # Person-anchored preservation gone, scene-wide preservation in.
    assert "exact facial features" not in result
    assert "overall composition" in result
    assert "relative position of all subjects" in result


def test_apply_scope_scene_rewrites_simpsons_variant_preservation():
    """The Simpsons preset uses a slightly different preservation
    phrasing ("recognizable expression" instead of "facial features"
    because the style restructures the face). That variant must also
    be recognized by scene mode."""
    base = (
        "Restyle this person as a Simpsons character, while preserving "
        "the recognizable expression, hairstyle, body proportions, "
        "and pose, with yellow skin"
    )
    result = apply_scope(base, "scene")
    assert "recognizable expression" not in result
    assert "overall composition" in result


@pytest.mark.parametrize("name", [
    "pixar", "anime", "simpsons", "ghibli", "vangogh", "pencil",
])
def test_apply_scope_scene_actually_rewrites_every_built_in(name):
    """End-to-end lock: scope=scene applied to every built-in preset
    prompt must produce a visibly different string (and NOT trigger
    the v0.3.3 SCOPE_SCENE_SUFFIX fallback — built-ins should rewrite,
    not append). Catches future drift where a preset's wording slips
    past every trigger and silently no-ops.

    v0.3.2 made scope=scene the default — this guarantee is critical
    because every default invocation depends on it."""
    from imgen.images import SCOPE_SCENE_SUFFIX
    from imgen.styles import STYLES

    original = STYLES[name]["prompt"]
    rewritten = apply_scope(original, "scene")

    assert rewritten != original, (
        f"{name}: scope=scene was a no-op on built-in prompt — no "
        f"trigger matched and fallback didn't fire"
    )
    # Built-ins must REWRITE (their "this person" or v0.3.4 preservation
    # clause triggers), not fall back to the suffix append meant for
    # truly-trigger-free prompts.
    assert not rewritten.endswith(SCOPE_SCENE_SUFFIX), (
        f"{name}: scope=scene used the fallback suffix instead of the "
        f"targeted substring rewrite — a trigger should have fired"
    )


@pytest.mark.parametrize("name", [
    "pixar", "anime", "simpsons", "ghibli", "vangogh", "pencil",
])
def test_apply_scope_person_appends_suffix_to_every_built_in(name):
    """Symmetric to the scene check — person mode must append the
    background-preservation suffix to every built-in preset."""
    from imgen.styles import STYLES

    original = STYLES[name]["prompt"]
    result = apply_scope(original, "person")
    assert result == original + SCOPE_PERSON_SUFFIX


def test_apply_scope_unknown_scope_is_noop():
    # apply_scope only knows "person"/"scene" — any other value (incl.
    # argparse rejects new ones, but the function itself shouldn't crash
    # if called programmatically) falls through unchanged.
    base = "test"
    assert apply_scope(base, "totally-unknown") == base

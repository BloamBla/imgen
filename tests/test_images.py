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
    SCOPE_SCENE_SUFFIX_GENERIC,
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


# ── apply_scope (v0.5 semantics) ─────────────────────────────────────
#
# v0.5 redesign: both scopes preserve the prompt's identity-anchor
# language verbatim. Scope only governs the BACKGROUND directive
# appended. Per-style ``scene_suffix`` tuned for visual school
# (Pixar 3D-painterly, Simpsons flat, Ghibli watercolor, etc.).
#
# This replaces the v0.3.x ``SCOPE_SCENE_REPLACEMENTS`` substring
# table that actively stripped identity language from scope=scene
# prompts — silently losing face preservation on every default-scope
# invocation (scope=scene is the imgen default since v0.3.2).


def test_apply_scope_none_unchanged():
    base = "Transform this person into anime"
    assert apply_scope(base, None) == base


def test_apply_scope_unknown_value_unchanged():
    """Defensive: anything that's not 'person'/'scene' is a no-op.
    argparse enforces the choices at CLI boundary; this guards
    against programmatic callers passing junk."""
    base = "x"
    assert apply_scope(base, "subject") == base
    assert apply_scope(base, "") == base


def test_apply_scope_person_appends_suffix():
    base = "Restyle this person preserving facial identity, anime"
    assert apply_scope(base, "person") == base + SCOPE_PERSON_SUFFIX


def test_apply_scope_person_ignores_scene_suffix_arg():
    """``scene_suffix`` is scope=scene-specific. Person mode must
    ignore it — passing a per-style scene_suffix to a person-scope
    call shouldn't leak the wrong directive into the prompt."""
    base = "x"
    result = apply_scope(base, "person", scene_suffix=", and paint sky")
    assert result == base + SCOPE_PERSON_SUFFIX
    assert "paint sky" not in result


def test_apply_scope_scene_no_suffix_uses_generic():
    """No per-style suffix → fall back to the generic background
    directive. This is the path for user-styles in styles.d/*.toml
    that haven't tuned their own scene_suffix yet."""
    base = "Render the subject as a watercolor painting"
    result = apply_scope(base, "scene")
    assert result == base + SCOPE_SCENE_SUFFIX_GENERIC


def test_apply_scope_scene_per_style_suffix_overrides_generic():
    """Per-style ``scene_suffix`` wins over the generic fallback.
    Built-in styles use this to tune background language to each
    style's visual school."""
    base = "Restyle this person preserving facial identity, ghibli"
    ghibli_suffix = (
        ", and transform the background into a Studio Ghibli "
        "watercolor environment"
    )
    result = apply_scope(base, "scene", scene_suffix=ghibli_suffix)
    assert result == base + ghibli_suffix
    # Generic must NOT have leaked in alongside.
    assert SCOPE_SCENE_SUFFIX_GENERIC not in result


def test_apply_scope_scene_empty_per_style_suffix_falls_back_to_generic():
    """Empty-string scene_suffix is "no per-style override" — treat as
    if absent, fall back to generic."""
    base = "x"
    result = apply_scope(base, "scene", scene_suffix="")
    assert result == base + SCOPE_SCENE_SUFFIX_GENERIC


def test_apply_scope_scene_fallback_on_empty_prompt():
    """Edge case: empty prompt + scope=scene → just the suffix.
    Defensive — shouldn't happen in practice (resolve_prompt rejects
    empty input upstream)."""
    assert apply_scope("", "scene") == SCOPE_SCENE_SUFFIX_GENERIC


def test_apply_scope_suffix_constants_shape():
    """Both suffix constants must lead with a comma so they graft
    cleanly onto an existing prompt."""
    assert SCOPE_PERSON_SUFFIX.startswith(",")
    assert SCOPE_SCENE_SUFFIX_GENERIC.startswith(",")


# ── v0.5 regression: identity-anchor language survives BOTH scopes ────


def test_apply_scope_person_keeps_facial_identity():
    """scope=person preserves the identity-anchor language verbatim
    (it lives in the style preset, not in the scope suffix)."""
    base = (
        "Restyle this person as anime, while preserving the facial "
        "identity, hairstyle, body proportions, and pose"
    )
    result = apply_scope(base, "person")
    assert "facial identity" in result
    assert "hairstyle, body proportions, and pose" in result


def test_apply_scope_scene_keeps_facial_identity():
    """v0.5 regression test — scope=scene must NOT strip identity-
    anchor language. v0.3.x SCOPE_SCENE_REPLACEMENTS used to rewrite
    'preserving the facial identity, hairstyle, body proportions, and
    pose' into 'preserving the overall composition...' — that silently
    broke face preservation on every default invocation. v0.5 fix:
    identity language stays untouched, scene-suffix only ADDS a
    background directive."""
    base = (
        "Restyle this person as anime, while preserving the facial "
        "identity, hairstyle, body proportions, and pose"
    )
    result = apply_scope(base, "scene")
    # Identity-anchor language untouched.
    assert "facial identity" in result
    assert "hairstyle, body proportions, and pose" in result
    # AND background directive added on top.
    assert "background" in result.lower()


def test_apply_scope_scene_keeps_exact_facial_features_variant():
    """Same regression for the 'exact facial features' anchor used
    by vangogh + pencil styles."""
    base = (
        "Restyle this person's portrait as a Van Gogh oil painting, "
        "while preserving the exact facial features, hairstyle, body "
        "proportions, and pose"
    )
    result = apply_scope(base, "scene")
    assert "exact facial features" in result


def test_apply_scope_scene_keeps_recognizable_expression_variant():
    """Same regression for the 'recognizable expression' anchor used
    by the simpsons style (face restructures too radically to anchor
    on identity)."""
    base = (
        "Restyle this person as a Simpsons character, while preserving "
        "the recognizable expression, hairstyle, body proportions, and "
        "pose"
    )
    result = apply_scope(base, "scene")
    assert "recognizable expression" in result


# ── Per-style scene_suffix lock-in on every built-in ──────────────────


@pytest.mark.parametrize("name", [
    "pixar", "anime", "simpsons", "ghibli", "vangogh", "pencil",
])
def test_every_built_in_has_scene_suffix(name):
    """All 6 built-in styles carry their own ``scene_suffix`` tuned
    to the style's visual school. Lock-in test so a future refactor
    that drops the field silently doesn't fall the style back to the
    generic — the per-style language is part of the brand promise."""
    from imgen.styles import BUILTIN_STYLES
    suffix = BUILTIN_STYLES[name].get("scene_suffix")
    assert isinstance(suffix, str) and suffix.strip(), (
        f"{name}: missing or empty scene_suffix field"
    )
    # All built-in suffixes start with a comma+space for clean
    # concatenation onto the prompt.
    assert suffix.startswith(", "), f"{name}: scene_suffix shape"


@pytest.mark.parametrize("name,must_contain", [
    ("pixar", "Pixar"),
    ("anime", "anime"),
    ("simpsons", "Simpsons"),
    ("ghibli", "Ghibli"),
    ("vangogh", "Van Gogh"),
    ("pencil", "pencil"),
])
def test_per_style_scene_suffix_mentions_its_style(name, must_contain):
    """Each per-style scene_suffix should name its style/school so
    FLUX has a clear cue. Pixar suffix mentions 'Pixar', Ghibli
    mentions 'Ghibli', etc. Catches accidental copy-paste between
    style suffixes."""
    from imgen.styles import BUILTIN_STYLES
    suffix = BUILTIN_STYLES[name]["scene_suffix"]
    assert must_contain in suffix, (
        f"{name}: scene_suffix doesn't mention '{must_contain}': "
        f"{suffix!r}"
    )


def test_every_built_in_scene_suffix_is_unique():
    """No two built-in styles should have the same scene_suffix —
    each is tuned for its specific visual school. A copy-paste
    accident (e.g. vangogh's scene_suffix landing on pencil) would
    silently produce wrong backgrounds for one of the affected
    styles."""
    from imgen.styles import BUILTIN_STYLES
    suffixes = [
        BUILTIN_STYLES[n]["scene_suffix"]
        for n in ("pixar", "anime", "simpsons", "ghibli",
                  "vangogh", "pencil")
    ]
    assert len(set(suffixes)) == len(suffixes), (
        "duplicate scene_suffix across built-in styles"
    )


@pytest.mark.parametrize("name", [
    "pixar", "anime", "simpsons", "ghibli", "vangogh", "pencil",
])
def test_apply_scope_scene_on_built_in_uses_per_style_suffix(name):
    """End-to-end through apply_scope: feeding a built-in style's
    prompt + its scene_suffix produces output ending in the
    per-style suffix (not the generic). This is the contract the
    cmd_helpers call-site relies on."""
    from imgen.styles import BUILTIN_STYLES

    style = BUILTIN_STYLES[name]
    result = apply_scope(
        style["prompt"], "scene",
        scene_suffix=style["scene_suffix"],
    )
    assert result.endswith(style["scene_suffix"])
    assert SCOPE_SCENE_SUFFIX_GENERIC not in result


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

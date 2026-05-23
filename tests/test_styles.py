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
    assert preset.prompt is not None, f"{name}: missing 'prompt'"
    assert bool(preset.negative), f"{name}: empty or missing 'negative'"
    assert isinstance(preset.prompt, str)
    assert isinstance(preset.negative, str)
    assert preset.prompt.strip(), f"{name}: empty prompt"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_guidance_in_argparse_range(name):
    """Argparse validator allows 0.5..15.0 — preset overrides must be in range."""
    g = STYLES[name].guidance
    if g is not None:
        assert 0.5 <= g <= 15.0, f"{name}: guidance {g} out of [0.5, 15.0]"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_strength_in_argparse_range(name):
    """Argparse validator allows 0.0..1.0 — preset overrides must be in range."""
    s = STYLES[name].strength
    if s is not None:
        assert 0.0 <= s <= 1.0, f"{name}: strength {s} out of [0.0, 1.0]"


# ── v0.3.4: structural lock on the BFL-aligned prompt shape ────────────


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_uses_restyle_verb_not_transform_person(name):
    """v0.3.4: every built-in preset starts with "Restyle this person
    as X" — NOT the v0.1.x-v0.3.3 "Transform this person into X"
    pattern. BFL guidance explicitly flags "Transform [person]" as
    identity-drift risk; "Restyle" / "Convert" target the rendering
    rather than the person object."""
    prompt = STYLES[name].prompt
    assert prompt.startswith("Restyle this person as "), \
        f"{name}: prompt must lead with 'Restyle this person as …' " \
        f"(got: {prompt[:60]!r})"
    assert "Transform this person" not in prompt, \
        f"{name}: legacy 'Transform this person' verb leaked back in"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_has_explicit_preservation_clause(name):
    """v0.3.4: every preset must contain an explicit "while preserving"
    clause that anchors identity/figure preservation in the middle of
    the prompt (BFL-recommended position). Without this anchor the
    style descriptors at the tail can drift the model away from the
    source person."""
    prompt = STYLES[name].prompt
    assert "while preserving" in prompt, \
        f"{name}: missing 'while preserving …' preservation clause"
    # Must cover hairstyle + body proportions + pose at minimum
    # (the four-anchor pattern face/hair/body/pose).
    for anchor in ("hairstyle", "body proportions", "pose"):
        assert anchor in prompt, f"{name}: preservation missing '{anchor}'"


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_drops_legacy_keep_face_identity_phrasing(name):
    """The terse legacy "keep face identity" / "keep pose" phrasing
    from early built-in prompts is replaced by the explicit
    "while preserving the facial identity / exact facial features /
    recognizable expression, hairstyle, body proportions, and pose"
    block. Locks against accidental drift back to the old wording —
    that wording lost the identity-anchor entirely in scope=scene.

    The v0.3.x ``SCOPE_SCENE_REPLACEMENTS`` substring-rewrite table
    that translated this legacy phrasing to scene-anchored language
    was deleted in the v0.5 ``apply_scope`` rewrite — built-ins now
    ship with the explicit preservation clause directly, no rewrite
    needed."""
    prompt = STYLES[name].prompt
    assert "keep face identity" not in prompt
    assert "keep pose" not in prompt


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


def test_get_style_unknown_raises_stylenotfound_subclass():
    """v0.3.6: raise StyleNotFound (KeyError subclass) so existing
    `except KeyError` handlers still match, but the friendly __str__
    avoids KeyError's repr-style quoting around the message.
    (python #21 from v0.1.x review.)"""
    from imgen.styles import StyleNotFound
    with pytest.raises(StyleNotFound) as exc_info:
        get_style("nonexistent_style_xyz")
    # Subclass relationship: legacy `except KeyError` callers still catch.
    assert isinstance(exc_info.value, KeyError)
    # Clean __str__ — no surrounding quotes like KeyError would add.
    msg = str(exc_info.value)
    assert msg.startswith("Unknown style 'nonexistent_style_xyz'.")
    assert not msg.startswith('"'), "StyleNotFound must not inherit KeyError repr-quoting"


def test_stylenotfound_str_with_non_string_arg_does_not_raise():
    """CPython requires __str__ to return str — `return self.args[0]`
    without wrapping in str() would raise TypeError on a non-string
    first arg. The wrap guards future callers from that footgun.
    (v0.3.6 python-reviewer CRITICAL.)"""
    from imgen.styles import StyleNotFound

    # Pathological construction — not how get_style raises, but the
    # public class surface must be safe under any positional type.
    exc = StyleNotFound(42)
    # Must not raise TypeError under str().
    msg = str(exc)
    assert msg == "42"


def test_stylenotfound_str_with_no_args_returns_empty():
    """`raise StyleNotFound()` with no args → empty __str__, never crash.
    Symmetric with KeyError() which __str__'s to ''."""
    from imgen.styles import StyleNotFound
    assert str(StyleNotFound()) == ""


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


# ── v0.6: built-in LoRA mappings (anime / pixar / ghibli) ──────────────

# Locked picks per project-v050-v060-design memo (2026-05-22). The exact
# HF repo IDs + weights + triggers are lock-in tests so a typo or a
# silent "let me try this other LoRA" never reaches a tagged release
# without surfacing in CI first.
#
# simpsons / vangogh / pencil INTENTIONALLY stay text-only — design
# memo concluded no quality LoRA exists on HF for those three (Simpsons
# IP-blocked, pencil well-trained in FLUX base, vangogh only loose
# impressionism matches that didn't beat text-only in research).


def test_ghibli_ships_openfree_ghibli_lora_at_0p8():
    """openfree/flux-chatgpt-ghibli-lora — the only v0.6.0 built-in
    LoRA pick that survived the v0.6.1 Kontext post-ship A/B (anime
    + pixar reverted, ghibli kept). v0.6.3 LoRA round-2 re-confirmed
    it remains the right ghibli pick; new picks landed for anime /
    pixar / vangogh / pencil. See lock-in tests below for those.
    """
    from imgen.styles import LoraRef
    loras = STYLES["ghibli"].loras
    assert len(loras) == 1
    assert loras[0] == LoraRef(
        ref="openfree/flux-chatgpt-ghibli-lora",
        weight=0.8,
        compatible_with=("flux-1",),
        trigger="Ghibli style",
    )


def test_v063_lora_picks_per_style():
    """v0.6.3 BUILTIN_STYLES LoRA assignments after Phase 1 + Phase 2.

    Six of seven built-in styles ship a LoRA after v0.6.3:

    * anime → Shakker-Labs/...-Flat-Cartoon-Style @ 0.8 (primary user pick)
    * anime_alt → Kontext-Style/Irasutoya_lora @ 0.8 (alternative aesthetic)
    * pixar → Kontext-Style/Poly_lora @ 0.8 (primary user pick)
    * pixar_alt → Kontext-Style/3D_Chibi_lora @ 0.8 (alternative aesthetic)
    * vangogh → Kontext-Style/Oil_Painting_lora @ 0.8
    * pencil → Shakker-Labs/...-Sketch-Style @ 0.8 (replaces the v0.6.x
      pencil text-only fallback; Monochrome-Pencil crashed Phase 1)
    * ghibli → unchanged from v0.6.0 (see dedicated test above)

    simpsons stays text-only — Phase 1 found no Kontext-trained Simpsons
    LoRA that the user wanted to ship.

    Lock-in via this parametrised test so any future repo-id / weight /
    trigger drift surfaces immediately.
    """
    from imgen.styles import LoraRef
    expected = {
        "anime": LoraRef(
            ref="Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style",
            weight=0.8, compatible_with=("flux-1",),
            trigger="flat cartoon style",
        ),
        "anime_alt": LoraRef(
            ref="Kontext-Style/Irasutoya_lora",
            weight=0.8, compatible_with=("flux-1",),
            trigger="Irasutoya style",
        ),
        "pixar": LoraRef(
            ref="Kontext-Style/Poly_lora",
            weight=0.8, compatible_with=("flux-1",),
            trigger="Poly style",
        ),
        "pixar_alt": LoraRef(
            ref="Kontext-Style/3D_Chibi_lora",
            weight=0.8, compatible_with=("flux-1",),
            trigger="3D Chibi",
        ),
        "vangogh": LoraRef(
            ref="Kontext-Style/Oil_Painting_lora",
            weight=0.8, compatible_with=("flux-1",),
            trigger="Oil Painting",
        ),
        "pencil": LoraRef(
            ref="Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style",
            weight=0.8, compatible_with=("flux-1",),
            trigger="sketch",
        ),
    }
    for name, expected_lora in expected.items():
        loras = STYLES[name].loras
        assert len(loras) == 1, f"{name} should ship exactly 1 LoRA"
        assert loras[0] == expected_lora, (
            f"{name}: BUILTIN_STYLES drift — expected {expected_lora}, "
            f"got {loras[0]}"
        )


def test_alt_styles_share_non_lora_fields_with_primary():
    """v0.6.3 (architect IMP-1): ``anime_alt`` and ``pixar_alt`` share
    every non-LoRA field (prompt, negative, guidance, strength,
    scene_suffix) with their primary counterparts. The whole POINT of
    the ``_alt`` convention is "same style, different LoRA" — drift
    between primary and alt would silently break that promise.

    Lock-in here so a future prompt tune on ``anime`` is forced to
    either update ``anime_alt`` in lockstep OR explicitly disable the
    parity assertion (which requires editing this test, which requires
    review).
    """
    from imgen.styles import BUILTIN_STYLES
    pairs = [("anime", "anime_alt"), ("pixar", "pixar_alt")]
    parity_fields = ("prompt", "negative", "guidance", "strength", "scene_suffix")
    for primary, alt in pairs:
        primary_style = BUILTIN_STYLES[primary]
        alt_style = BUILTIN_STYLES[alt]
        for field in parity_fields:
            primary_val = getattr(primary_style, field)
            alt_val = getattr(alt_style, field)
            assert primary_val == alt_val, (
                f"{alt} drifted from {primary} on `{field}`:\n"
                f"  {primary}.{field} = {primary_val!r}\n"
                f"  {alt}.{field}     = {alt_val!r}\n"
                f"`_alt` styles must share every non-LoRA field with "
                f"their primary — only `loras` may differ. If the "
                f"divergence is intentional, drop `_alt` and give the "
                f"new style a distinct name."
            )
        # And the loras MUST differ — same LoRA on primary + alt would
        # make the alt redundant.
        assert primary_style.loras != alt_style.loras, (
            f"{alt} ships the same LoRA stack as {primary} — "
            f"`_alt` only exists to expose an alternative LoRA"
        )


def test_builtin_lora_triggers_pass_is_safe_stem():
    """v0.6.3 security NIT-3 follow-up: built-in LoRA triggers are
    Python string literals in ``BUILTIN_STYLES`` and never pass through
    ``_is_safe_stem`` via the user-TOML loader path. They're trusted
    by code-review at commit time, but a defence-in-depth test
    asserts the invariant explicitly.

    Catches a future built-in trigger that accidentally contains a
    control byte (C0 / DEL / C1) — those bytes would otherwise ride
    into the prompt → mflux argv → log file → user's terminal on
    ``cat <log>``.
    """
    from imgen.styles import BUILTIN_STYLES, _is_safe_stem
    for name, style in BUILTIN_STYLES.items():
        for lora in style.loras:
            if lora.trigger is None:
                continue
            assert _is_safe_stem(lora.trigger), (
                f"BUILTIN_STYLES[{name!r}].loras trigger "
                f"{lora.trigger!r} contains control bytes — would "
                f"propagate into prompt → mflux argv → log file"
            )


def test_simpsons_stays_text_only():
    """simpsons remains text-only after v0.6.3 — Phase 1 found no
    Kontext-trained Simpsons LoRA worth shipping. Shakker-Labs/
    FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style was the closest
    proxy and works on the `anime` slot, but user Phase 2 verdict
    was that Flat-Cartoon does NOT fit Simpsons' specific yellow-
    skin / round-eyes / bold-outlines aesthetic. Lock-in so a
    future drive-by add doesn't silently regress."""
    assert STYLES["simpsons"].loras == ()


def test_built_in_loras_target_flux_1_only():
    """All v0.6.3 built-in LoRAs target the flux-1 compat group.
    Qwen backend gets text-only for all built-ins per design memo
    (HF ecosystem for Qwen style LoRAs is sparse; user can attach
    Qwen LoRAs ad-hoc via CLI --lora)."""
    from imgen.styles import BUILTIN_STYLES
    for name, style in BUILTIN_STYLES.items():
        for lora in style.loras:
            assert "flux-1" in lora.compatible_with, (
                f"{name}: LoRA {lora.ref!r} compat={lora.compatible_with} "
                f"missing flux-1"
            )


def test_style_dict_compat_api_removed_in_v0_7_9():
    """v0.7.9: the dict-compat shims (__getitem__, __contains__, get)
    were deleted after the v0.7.8 architect review's IMP-1 noted that
    3 callers pinned the surface. All production callers + tests now
    use attribute access (`style.prompt`, `style.guidance is not None`,
    `bool(style.loras)`). Misuse fails LOUDLY at runtime — no silent
    None landing in mflux argv via a stale `.get` call. Lock-in:
    each removed shim raises the expected exception type.
    """
    from imgen.styles import Style
    style = Style(prompt="x")

    # __getitem__ removed → subscript raises TypeError ("not
    # subscriptable") rather than returning None or KeyError.
    with pytest.raises(TypeError):
        _ = style["prompt"]

    # __contains__ removed → `in` raises TypeError ("not iterable")
    # rather than False/True from the v0.6.2-v0.7.8 shim.
    with pytest.raises(TypeError):
        _ = "prompt" in style

    # .get(...) removed → AttributeError, not the v0.6.2 shim's
    # default-on-missing return.
    with pytest.raises(AttributeError):
        _ = style.get("prompt")

    # Attribute access remains the canonical surface.
    assert style.prompt == "x"
    assert style.negative == ""
    assert style.guidance is None
    assert style.loras == ()


def test_repo_from_cache_dir_asserts_on_missing_prefix():
    """v0.6.2 python NIT-2: passing a non-``models--`` name to
    repo_from_cache_dir is a caller bug that historically corrupted
    repo ids with embedded ``--``. Assertion fails fast."""
    from imgen.hf_cache import repo_from_cache_dir
    assert repo_from_cache_dir("models--openfree--flux-chatgpt-ghibli-lora") == \
        "openfree/flux-chatgpt-ghibli-lora"
    with pytest.raises(AssertionError, match="repo_from_cache_dir expects"):
        repo_from_cache_dir("not-a-cache-dir")


def test_builtin_loras_roundtrip_through_parse_lora_refs():
    """v0.6.x backlog python NIT-7: every BUILTIN_STYLES.loras entry
    must round-trip through :func:`parse_lora_refs` (the same validator
    user TOMLs go through). Catches a future hand-written pick that
    bypasses the schema — e.g. a typo'd compat group, a weight out of
    bounds, control bytes in a trigger.

    Moved here from import-time invariant in styles.py (architect
    v0.6.2 NIT-1): pytest-level surfaces this as a normal test failure,
    not an ImportError raised during ``import imgen.styles``.
    """
    from imgen.styles import BUILTIN_STYLES, UserStyleError, parse_lora_refs
    for name, style in BUILTIN_STYLES.items():
        loras = style.loras
        if not loras:
            continue
        as_dicts = [
            {
                "ref": lora.ref,
                "weight": lora.weight,
                "compatible_with": list(lora.compatible_with),
                **(
                    {"trigger": lora.trigger}
                    if lora.trigger is not None else {}
                ),
            }
            for lora in loras
        ]
        try:
            roundtrip = parse_lora_refs(as_dicts, f"BUILTIN_STYLES[{name!r}]")
        except UserStyleError as e:
            pytest.fail(
                f"BUILTIN_STYLES[{name!r}].loras failed parse_lora_refs "
                f"validation: {e}"
            )
        assert roundtrip == loras, (
            f"BUILTIN_STYLES[{name!r}].loras did not round-trip through "
            f"parse_lora_refs (schema drift?): "
            f"input={loras} roundtrip={roundtrip}"
        )

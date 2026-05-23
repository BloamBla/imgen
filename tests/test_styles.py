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


# ── v0.3.4: structural lock on the BFL-aligned prompt shape ────────────


@pytest.mark.parametrize("name", ALL_STYLES)
def test_preset_uses_restyle_verb_not_transform_person(name):
    """v0.3.4: every built-in preset starts with "Restyle this person
    as X" — NOT the v0.1.x-v0.3.3 "Transform this person into X"
    pattern. BFL guidance explicitly flags "Transform [person]" as
    identity-drift risk; "Restyle" / "Convert" target the rendering
    rather than the person object."""
    prompt = STYLES[name]["prompt"]
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
    prompt = STYLES[name]["prompt"]
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
    prompt = STYLES[name]["prompt"]
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
    """openfree/flux-chatgpt-ghibli-lora is the design-memo FALLBACK
    pick — the primary alvarobartt/ghibli-characters-flux-lora carried
    a "TBD pending license check" marker so we ship the well-
    established openfree LoRA with clearer licensing for v0.6 cut.

    v0.6.1 lesson: this is the ONLY built-in LoRA that survived the
    post-ship A/B on FLUX.1-Kontext-dev. anime (Flux-Animeo-v1-LoRA)
    and pixar (Canopus-Pixar-3D-FluxDev-LoRA) crashed mflux Kontext
    at the first denoise step with a (1,4992,16)×(64,3072) attention
    shape mismatch — they were trained on FLUX.1-dev base T2I whose
    attention layer is shaped differently than Kontext's. All 912
    LoRA keys "matched" at load time, but the runtime attention math
    exploded. openfree-ghibli happens to use a LoRA rank that maps
    onto Kontext's attention shapes correctly. Per-LoRA Kontext-
    compat must be verified by ACTUAL inference, not by name."""
    from imgen.styles import LoraRef
    loras = STYLES["ghibli"].get("loras", ())
    assert len(loras) == 1
    assert loras[0] == LoraRef(
        ref="openfree/flux-chatgpt-ghibli-lora",
        weight=0.8,
        compatible_with=("flux-1",),
        trigger="Ghibli style",
    )


@pytest.mark.parametrize(
    "name", ["anime", "pixar", "simpsons", "vangogh", "pencil"],
)
def test_text_only_built_ins_have_no_loras(name):
    """Five of six built-in styles ship text-only in v0.6.1:

    * anime / pixar — REGRESSION-LOCK from v0.6.0. The two FLUX.1-dev
      LoRAs (Flux-Animeo-v1-LoRA + Canopus-Pixar-3D-FluxDev-LoRA)
      crashed mflux Kontext at runtime; reverted to text-only in
      v0.6.1. If a future commit re-adds a LoRA to either style,
      the regression resurfaces silently — this test catches it.
      Per-LoRA Kontext-compat MUST be verified by real inference
      before adding back. (See ``test_ghibli_ships_openfree_ghibli_
      lora_at_0p8`` for the full lesson.)
    * simpsons — IP-blocked on HF, no quality LoRA exists.
    * vangogh — available impressionism LoRAs didn't beat text-only.
    * pencil — already strong in FLUX base training, no quality boost
      from available LoRAs.
    """
    assert STYLES[name].get("loras", ()) == ()


def test_built_in_loras_target_flux_1_only():
    """Built-in LoRAs in v0.6.1 (only ghibli now) target the flux-1
    compat group. Qwen backend gets text-only for all built-ins per
    design memo (HF ecosystem for Qwen style LoRAs is sparse; user
    can attach Qwen LoRAs ad-hoc via CLI --lora)."""
    for lora in STYLES["ghibli"].get("loras", ()):
        assert "flux-1" in lora.compatible_with


def test_style_contains_returns_false_for_none_fields():
    """v0.6.2 python IMP-1 regression: a Style with ``guidance=None``
    (unset Optional field) must report ``"guidance" not in style`` so
    cmd_helpers.build_iterations's ``"guidance" in preset → preset
    ["guidance"]`` gate cleanly falls through to merged_defaults.

    Without this, the slotted dataclass would always report every
    declared field as "in" — landing ``None`` in mflux's --guidance
    argv after the fallback chain incorrectly stopped early.
    """
    from imgen.styles import Style
    style = Style(prompt="x")
    assert "prompt" in style
    # None-valued Optional fields look "absent" — mirrors the v0.5 dict
    # behaviour where a TOML omitting `guidance` produced a dict without
    # that key at all.
    assert "guidance" not in style
    assert "strength" not in style
    assert "scene_suffix" not in style
    # Fields with non-None empty defaults stay "in" (callers read them
    # unconditionally).
    assert "negative" in style
    assert "loras" in style
    # And a fully populated style has every field "in".
    full = Style(
        prompt="x", negative="y", guidance=3.5, strength=0.5,
        scene_suffix="z",
    )
    for f in ("prompt", "negative", "guidance", "strength", "scene_suffix"):
        assert f in full
    # Private / non-string keys are never "in".
    assert "__class__" not in full
    assert 42 not in full  # type: ignore[operator]


def test_style_get_filters_private_attributes():
    """v0.6.2 python NIT-1: ``Style.get("__class__")`` must return the
    default, not the class object — match ``__contains__`` semantics.
    """
    from imgen.styles import Style
    style = Style(prompt="x")
    assert style.get("__class__", "sentinel") == "sentinel"
    assert style.get("__init__", "sentinel") == "sentinel"
    # Real fields still work.
    assert style.get("prompt") == "x"
    assert style.get("negative") == ""


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

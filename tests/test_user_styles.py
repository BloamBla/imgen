"""User-style loader: read ~/.imgen/styles.d/*.toml, validate, merge with
built-ins. Conflicts get `_0001` etc. suffix, never shadow built-in.
"""
from __future__ import annotations

import pytest

from imgen.styles import (
    BUILTIN_STYLES,
    UserStyleError,
    load_user_style_file,
    load_user_styles_dir,
    merge_user_styles,
    reset_styles_cache,
)
import imgen.styles as styles_mod


# ── load_user_style_file — single TOML → preset dict ─────────────────────

def test_load_user_style_file_minimal_prompt_only(tmp_path):
    p = tmp_path / "noir.toml"
    p.write_text('prompt = "film noir style, black and white"')
    preset = load_user_style_file(p)
    assert preset == {"prompt": "film noir style, black and white"}


def test_load_user_style_file_all_fields(tmp_path):
    p = tmp_path / "cyberpunk.toml"
    p.write_text(
        'prompt = "cyberpunk neon city"\n'
        'negative = "rural, daylight"\n'
        "guidance = 4.5\n"
        "strength = 0.7\n"
    )
    preset = load_user_style_file(p)
    assert preset == {
        "prompt": "cyberpunk neon city",
        "negative": "rural, daylight",
        "guidance": 4.5,
        "strength": 0.7,
    }


def test_load_user_style_file_no_required_fields(tmp_path):
    """A param-only TOML (no prompt) is valid at load time. cmd_generate
    will check at use time whether --custom-prompt is supplied to fill
    the gap. This is the "flexibility" the user asked for — TOML files
    can be pure param presets."""
    p = tmp_path / "loose.toml"
    p.write_text("guidance = 4.0\n")
    preset = load_user_style_file(p)
    assert preset == {"guidance": 4.0}


def test_load_user_style_file_rejects_invalid_toml(tmp_path):
    p = tmp_path / "broken.toml"
    p.write_text("prompt = unclosed\n")
    with pytest.raises(UserStyleError) as exc_info:
        load_user_style_file(p)
    assert "broken.toml" in str(exc_info.value)


@pytest.mark.parametrize("bad_g", [0.4, 15.1, -1.0])
def test_load_user_style_file_rejects_bad_guidance(tmp_path, bad_g):
    p = tmp_path / "bad.toml"
    p.write_text(f"prompt = \"x\"\nguidance = {bad_g}\n")
    with pytest.raises(UserStyleError):
        load_user_style_file(p)


@pytest.mark.parametrize("bad_s", [-0.1, 1.1, 2.0])
def test_load_user_style_file_rejects_bad_strength(tmp_path, bad_s):
    p = tmp_path / "bad.toml"
    p.write_text(f"prompt = \"x\"\nstrength = {bad_s}\n")
    with pytest.raises(UserStyleError):
        load_user_style_file(p)


def test_load_user_style_file_rejects_bool_as_numeric(tmp_path):
    """TOML `guidance = true` would silently pass isinstance(int) — pinned."""
    p = tmp_path / "bool.toml"
    p.write_text('prompt = "x"\nguidance = true\n')
    with pytest.raises(UserStyleError):
        load_user_style_file(p)


def test_load_user_style_file_warns_on_unknown_field(tmp_path, capsys):
    p = tmp_path / "extra.toml"
    p.write_text('prompt = "x"\nmade_up_key = "ignored"\n')
    preset = load_user_style_file(p)
    assert "made_up_key" not in preset
    captured = capsys.readouterr()
    assert "made_up_key" in (captured.out + captured.err)


def test_load_user_style_file_rejects_non_string_prompt(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("prompt = 42\n")
    with pytest.raises(UserStyleError):
        load_user_style_file(p)


def test_load_user_style_file_rejects_oversized(tmp_path):
    """A 100 MB rogue TOML in styles.d/ shouldn't OOM tomllib. (security I2)"""
    from imgen.styles import USER_STYLE_MAX_BYTES
    p = tmp_path / "huge.toml"
    p.write_bytes(b"prompt = \"x\"\n" + b"# pad " * (USER_STYLE_MAX_BYTES // 6 + 100))
    with pytest.raises(UserStyleError) as exc_info:
        load_user_style_file(p)
    assert "too large" in str(exc_info.value).lower()


# ── reset_styles_cache — public API for tests / future imgen serve ──────

def test_reset_styles_cache_clears_module_cache():
    """The merged-styles cache is process-scoped. A public reset API lets
    tests + future `imgen serve` invalidate it without poking the private
    `_cached_merged` attribute. (architect #5)"""
    from imgen.styles import _load_merged_styles
    # Populate the cache
    _load_merged_styles()
    assert styles_mod._cached_merged is not None
    # Reset clears it
    reset_styles_cache()
    assert styles_mod._cached_merged is None
    # Next access repopulates
    _load_merged_styles()
    assert styles_mod._cached_merged is not None


# ── load_user_styles_dir — directory scan ────────────────────────────────

def test_load_user_styles_dir_missing_returns_empty(tmp_path):
    assert load_user_styles_dir(tmp_path / "nonexistent") == {}


def test_load_user_styles_dir_empty_dir_returns_empty(tmp_path):
    (tmp_path / "styles.d").mkdir()
    assert load_user_styles_dir(tmp_path / "styles.d") == {}


def test_load_user_styles_dir_filename_as_style_name(tmp_path):
    d = tmp_path / "styles.d"
    d.mkdir()
    (d / "noir.toml").write_text('prompt = "noir"')
    (d / "vapor.toml").write_text('prompt = "vapor"')
    result = load_user_styles_dir(d)
    assert set(result.keys()) == {"noir", "vapor"}
    assert result["noir"]["prompt"] == "noir"


def test_load_user_styles_dir_skips_non_toml(tmp_path):
    d = tmp_path / "styles.d"
    d.mkdir()
    (d / "good.toml").write_text('prompt = "good"')
    (d / "notes.md").write_text("ignore me")
    (d / "bad.json").write_text("{}")
    result = load_user_styles_dir(d)
    assert set(result.keys()) == {"good"}


def test_load_user_styles_dir_alphabetical_order(tmp_path):
    """Sort order determines which user-style gets the lower suffix on
    conflict — pin it to filename alphabetical so behavior is predictable."""
    d = tmp_path / "styles.d"
    d.mkdir()
    for name in ["zebra", "apple", "mango"]:
        (d / f"{name}.toml").write_text(f'prompt = "{name}"')
    result = load_user_styles_dir(d)
    # Python 3.7+ dicts preserve insertion order — alphabetical input
    # means alphabetical iteration.
    assert list(result.keys()) == ["apple", "mango", "zebra"]


def test_load_user_styles_dir_one_bad_file_doesnt_kill_others(tmp_path, capsys):
    """A single broken .toml should warn and skip — other styles still load."""
    d = tmp_path / "styles.d"
    d.mkdir()
    (d / "good.toml").write_text('prompt = "good"')
    (d / "broken.toml").write_text("prompt = unclosed")
    result = load_user_styles_dir(d)
    assert "good" in result
    assert "broken" not in result
    captured = capsys.readouterr()
    assert "broken.toml" in (captured.out + captured.err)


# ── merge_user_styles — built-in + user merge with suffix ────────────────

def test_merge_user_styles_no_conflict():
    builtins = {"pixar": {"prompt": "p"}, "anime": {"prompt": "a"}}
    user = {"noir": {"prompt": "n"}, "vapor": {"prompt": "v"}}
    merged = merge_user_styles(builtins, user)
    assert set(merged.keys()) == {"pixar", "anime", "noir", "vapor"}


def test_merge_user_styles_doesnt_mutate_inputs():
    builtins = {"pixar": {"prompt": "p"}}
    user = {"noir": {"prompt": "n"}}
    merge_user_styles(builtins, user)
    assert builtins == {"pixar": {"prompt": "p"}}
    assert user == {"noir": {"prompt": "n"}}


def test_merge_user_styles_conflict_with_builtin_gets_0001_suffix(capsys):
    """User-defined 'anime' must not shadow built-in 'anime'. Gets renamed
    to 'anime_0001' with a warning. Built-in stays accessible as 'anime'."""
    builtins = {"anime": {"prompt": "builtin anime"}}
    user = {"anime": {"prompt": "user anime"}}
    merged = merge_user_styles(builtins, user)
    assert merged["anime"]["prompt"] == "builtin anime"  # built-in wins
    assert merged["anime_0001"]["prompt"] == "user anime"
    captured = capsys.readouterr()
    assert "anime_0001" in (captured.out + captured.err)


def test_merge_user_styles_conflict_with_existing_user_increments():
    """Hypothetical: if 'anime_0001' is ALSO taken (e.g. user has both
    anime.toml and anime_0001.toml), incrementing the suffix until free."""
    builtins = {"anime": {"prompt": "builtin"}}
    user = {
        "anime": {"prompt": "u1"},
        "anime_0001": {"prompt": "u2"},  # explicit user-named clash
    }
    merged = merge_user_styles(builtins, user)
    assert merged["anime"]["prompt"] == "builtin"
    # anime_0001 was already taken by user's explicit name, so the
    # *unnamed* (filename-derived) anime → anime_0002
    assert "anime_0001" in merged
    assert "anime_0002" in merged
    # Which assignment maps to which depends on input dict order; just
    # assert both user prompts are accessible somewhere.
    user_prompts = {merged["anime_0001"]["prompt"], merged["anime_0002"]["prompt"]}
    assert user_prompts == {"u1", "u2"}


def test_merge_user_styles_builtin_unchanged_after_conflict():
    """Built-in dict must not be modified during the merge — pure function."""
    builtins = {"anime": {"prompt": "builtin", "guidance": 4.0}}
    user = {"anime": {"prompt": "user"}}
    merge_user_styles(builtins, user)
    assert builtins["anime"] == {"prompt": "builtin", "guidance": 4.0}

"""Tests for src/imgen/_i2v_resolve.py — v0.9.3 i2v pure decision layer.

Two surfaces, both strictly pure (no subprocess, no diffusers import):

  * ``resolve_i2v_motion_defaults`` — when ``--image`` is present, bump
    LTX guidance from t2v default (3) to i2v default (5) and inject a
    motion-aware negative prompt unless the user explicitly overrode
    either flag. Without ``--image`` (t2v path) the function is a
    pass-through that returns the CLI values unchanged.

  * ``validate_image_path_or_die`` — user-supplied conditioning-image
    path must exist, be a real file, carry no C0/DEL/C1 control bytes,
    use one of the LTX-VAE-safe extensions {.png, .jpg, .jpeg}, and not
    be a symlink with an absolute / traversal target (mirroring the
    v0.9.1 B-14 venv-binary guard pattern for defense-in-depth).

Strict TDD per project CLAUDE.md matrix — pure functions get failing
test → minimal impl → green. RED step is checked by running the suite
against this file before any production module exists.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from imgen._i2v_resolve import (
    _I2V_DEFAULT_CFG,
    _I2V_DEFAULT_NEGATIVE,
    _I2V_INPUT_EXTS,
    resolve_i2v_motion_defaults,
    validate_image_path_or_die,
)


# ── resolve_i2v_motion_defaults ─────────────────────────────────────────


def test_resolve_t2v_path_returns_inputs_unchanged():
    """When image is None (t2v mode), the resolver is a pass-through.
    Caller applies its own t2v defaults further downstream."""
    g, n = resolve_i2v_motion_defaults(
        image=None, cli_guidance=None, cli_negative=None,
    )
    assert g is None
    assert n is None


def test_resolve_t2v_path_preserves_user_cli_values():
    """User-set --guidance / --negative-prompt in t2v mode pass through
    untouched — i2v resolver doesn't shadow t2v overrides."""
    g, n = resolve_i2v_motion_defaults(
        image=None, cli_guidance=4.0, cli_negative="ugly, blurry",
    )
    assert g == 4.0
    assert n == "ugly, blurry"


def test_resolve_i2v_defaults_when_image_present_and_no_overrides(tmp_path):
    """Bare ``imgen video --image PATH --prompt …`` → cfg becomes the
    i2v default (5.0) and the motion-aware negative prompt is injected.
    Research V2 (cfg=3, no negative) produced static-shimmer; cfg=5 +
    motion-negative fixes it."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    g, n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=None, cli_negative=None,
    )
    assert g == _I2V_DEFAULT_CFG
    assert g == 5.0
    assert n == _I2V_DEFAULT_NEGATIVE


def test_resolve_i2v_user_cfg_override_wins(tmp_path):
    """``imgen video --image PATH --guidance 7`` — explicit user value
    wins over the i2v default. Negative still gets the motion sentinel
    because the user didn't override it."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    g, n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=7.0, cli_negative=None,
    )
    assert g == 7.0
    assert n == _I2V_DEFAULT_NEGATIVE


def test_resolve_i2v_user_negative_override_wins(tmp_path):
    """User-supplied ``--negative-prompt "foo"`` replaces the motion
    sentinel verbatim — no merging, no appending. If the user wants
    motion-aware-plus-foo they pass both clauses in one string."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    g, n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=None, cli_negative="foo",
    )
    assert g == _I2V_DEFAULT_CFG
    assert n == "foo"


def test_resolve_i2v_both_overrides_pass_through(tmp_path):
    """Both flags set → both pass through. The resolver doesn't
    second-guess user intent; user owns both knobs."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    g, n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=6.5, cli_negative="ugly",
    )
    assert g == 6.5
    assert n == "ugly"


def test_resolve_i2v_user_cfg_zero_is_preserved(tmp_path):
    """Edge case: --guidance 0 is a legitimate user choice (disables
    classifier-free guidance). Must NOT collapse to the i2v default
    because 0.0 is falsy. ``is not None`` semantics, not truthiness."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    g, _n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=0.0, cli_negative=None,
    )
    assert g == 0.0


def test_resolve_i2v_user_negative_empty_string_is_preserved(tmp_path):
    """Edge case: --negative-prompt "" is a legitimate "disable my
    negative" intent. Must NOT collapse to the motion sentinel — empty
    string is falsy but not None."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    _g, n = resolve_i2v_motion_defaults(
        image=img, cli_guidance=None, cli_negative="",
    )
    assert n == ""


# ── validate_image_path_or_die ──────────────────────────────────────────


def test_validate_image_path_returns_resolved_absolute(tmp_path):
    """Happy path: existing .png file → resolved absolute Path."""
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    result = validate_image_path_or_die(str(img))
    assert result == img.resolve()
    assert result.is_absolute()


def test_validate_image_path_accepts_jpg_and_jpeg(tmp_path):
    """LTX VAE accepts the standard photographic encodings; the i2v
    allowlist mirrors {.png, .jpg, .jpeg}. Both .jpg and .jpeg variants
    must pass."""
    for name in ("a.jpg", "b.jpeg", "c.JPG"):
        img = tmp_path / name
        img.write_bytes(b"x")
        result = validate_image_path_or_die(str(img))
        assert result.name == name


def test_validate_image_path_expanduser(tmp_path, monkeypatch):
    """``~`` in user-typed paths gets expanded — colleagues paste
    ``~/Pictures/x.png`` and we resolve it like the shell would."""
    monkeypatch.setenv("HOME", str(tmp_path))
    img = tmp_path / "tilde.png"
    img.write_bytes(b"x")
    result = validate_image_path_or_die("~/tilde.png")
    assert result == img.resolve()


def test_validate_image_path_nonexistent_dies_code_2(tmp_path):
    """Non-existent path is a user-error class (exit 2) — exits with
    a recovery hint before any PIL/diffusers import."""
    missing = tmp_path / "does-not-exist.png"
    with pytest.raises(SystemExit) as exc:
        validate_image_path_or_die(str(missing))
    assert exc.value.code == 2


def test_validate_image_path_not_a_file_dies(tmp_path):
    """Pointing at a directory (typo / wrong arg) dies cleanly — never
    silently treat a dir as zero-byte image."""
    d = tmp_path / "subdir"
    d.mkdir()
    with pytest.raises(SystemExit) as exc:
        validate_image_path_or_die(str(d))
    assert exc.value.code == 2


def test_validate_image_path_unsupported_extension_dies(tmp_path):
    """.webp / .gif / .heic / .mp4 / no-extension all reject. LTX VAE's
    behaviour on these hasn't been smoke-verified; refuse rather than
    silently route them to PIL and risk a half-broken inference."""
    for name in ("a.webp", "b.gif", "c.heic", "d.mp4", "e", "f.PNG.bak"):
        bad = tmp_path / name
        bad.write_bytes(b"x")
        with pytest.raises(SystemExit) as exc:
            validate_image_path_or_die(str(bad))
        assert exc.value.code == 2, f"{name} should reject with code=2"


def test_validate_image_path_control_bytes_in_name_dies(tmp_path):
    """C0/DEL/C1 control bytes in the resolved filename inject ANSI
    escapes into terminal + log surfaces. Mirror discovery_inputs +
    resolve_single_input_path hardening."""
    # \x1b is ESC — would clear screen if printed raw.
    bad = tmp_path / "img\x1b[2J.png"
    bad.write_bytes(b"x")
    with pytest.raises(SystemExit) as exc:
        validate_image_path_or_die(str(bad))
    assert exc.value.code == 2


def test_validate_image_path_symlink_relative_peer_allowed(tmp_path):
    """Canonical relative same-dir symlinks (the v0.9.1 B-14 venv-python
    pattern equivalent for user-supplied images) are allowed. ``ln -s
    real.png link.png`` is a normal user filesystem convenience."""
    real = tmp_path / "real.png"
    real.write_bytes(b"x")
    link = tmp_path / "link.png"
    link.symlink_to("real.png")  # relative, peer — should pass
    result = validate_image_path_or_die(str(link))
    # resolve() collapses to real.png; the function may return either
    # the resolved real path. Important: no SystemExit.
    assert result.exists()


def test_validate_image_path_symlink_with_path_target_rejects(tmp_path):
    """Symlink whose readlink contains ``/`` (absolute target or
    traversal) is a plant-attack signal — mirrors v0.9.1 B-14 narrow
    guard. ``ln -s /etc/passwd img.png`` rejects."""
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    real = target_dir / "real.png"
    real.write_bytes(b"x")
    link = tmp_path / "link.png"
    link.symlink_to(real)  # absolute → contains "/"
    with pytest.raises(SystemExit) as exc:
        validate_image_path_or_die(str(link))
    assert exc.value.code == 2


def test_validate_image_path_symlink_toctou_falls_through(tmp_path, monkeypatch):
    """TOCTOU resilience (v0.9.1 B-14 pattern): if readlink() raises
    OSError between the is_symlink() check and our explicit read (e.g.
    symlink vanished, race), we DON'T propagate OSError — we let the
    downstream existence + is_file() guards catch the state with a
    clean user-facing die. Mirror commands/video.py:147-158.

    Test design note: the production code calls our defensive
    ``os.readlink`` first and then ``Path.resolve()`` which internally
    calls os.readlink too. To simulate "TOCTOU on our explicit call
    only", the mock raises on the FIRST call and passes through to the
    real implementation afterwards. That way ``Path.resolve()`` keeps
    working as expected; only our defensive try/except wrapper is
    exercised.
    """
    real = tmp_path / "real.png"
    real.write_bytes(b"x")
    link = tmp_path / "link.png"
    link.symlink_to("real.png")

    real_readlink = os.readlink
    call_count = {"n": 0}

    def boom(p, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated TOCTOU race on first readlink")
        return real_readlink(p, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", boom)
    # Should NOT raise OSError — defensive try/except swallows; the
    # downstream existence check passes (symlink + target intact), so
    # this is a clean accept. Test goal: no unhandled OSError leak.
    result = validate_image_path_or_die(str(link))
    assert result.exists()
    assert call_count["n"] >= 1, "defensive readlink should have been called"


# ── module-level constants ──────────────────────────────────────────────


def test_constants_have_expected_values():
    """Lock the i2v defaults so downstream tests / docs can refer to
    these without re-derivation. Changes here are intentional and
    review-worthy (research V2 burned cfg=3 for ambient prompts)."""
    assert _I2V_DEFAULT_CFG == 5.0
    assert _I2V_DEFAULT_NEGATIVE == "static, still, frozen, no motion"
    assert _I2V_INPUT_EXTS == frozenset({".png", ".jpg", ".jpeg"})

"""Pure decision layer for LTX-Video image-to-video mode (v0.9.3).

Two surfaces, both strictly pure (no subprocess, no diffusers import):

* :func:`resolve_i2v_motion_defaults` — when ``--image`` is present,
  LTX t2v defaults (cfg=3, no negative_prompt) produce static-shimmer
  results in i2v mode (research V2 at cfg=3 was nearly static — see
  [[project-ltx-i2v-research-2026-05-27]]). When the conditioning image
  is set, bump guidance → :data:`_I2V_DEFAULT_CFG` and inject the motion-
  aware :data:`_I2V_DEFAULT_NEGATIVE` UNLESS the user explicitly
  overrode either flag. Override semantics are ``is not None`` (not
  truthiness): ``--guidance 0`` and ``--negative-prompt ""`` are
  legitimate user choices that must survive.

* :func:`validate_image_path_or_die` — user-supplied conditioning-image
  path must exist, be a real file, carry no C0/DEL/C1 control bytes,
  use one of the LTX-VAE-safe extensions in :data:`_I2V_INPUT_EXTS`,
  and not be a symlink with an absolute / traversal target (mirrors
  v0.9.1 B-14 narrow guard on `.venv-diffusers/bin/{python,pip}` —
  same plant-attack signal). Exits with code=2 + recovery hint on
  failure; returns the resolved absolute Path on success.

Why a dedicated module instead of extending ``inputs.py``: the t2i/i2i
input-side resolver (``resolve_single_input_path``) does not validate
extension because mflux's PIL accepts a broader set, while LTX VAE's
behaviour on .webp / .gif / .heic hasn't been smoke-verified — we
refuse rather than risk a half-broken inference. Keeping i2v's
allowlist + i2v-specific error wording in a separate module avoids
broadening the t2i validator's contract.
"""
from __future__ import annotations

import os
from pathlib import Path

from .colors import die

__all__ = [
    "resolve_i2v_motion_defaults",
    "validate_image_path_or_die",
]


# ── i2v motion defaults ─────────────────────────────────────────────────

# LTX i2v needs stronger prompt adherence than t2v to escape stasis.
# Research V2 (cfg=3, ambient mountain prompt) was nearly static; V1
# (cfg=3 too but with a "samurai in fog" subject mid-motion) animated
# fine. cfg=5 is the conservative bump that helps ambient prompts
# without over-cooking dynamic ones. User can override via --guidance.
_I2V_DEFAULT_CFG: float = 5.0

# Motion-aware negative prompt pushes LTX out of "shimmer-only" outputs
# even when the user-supplied --prompt lacks explicit motion verbs.
# Research mitigation #3 from [[project-ltx-i2v-research-2026-05-27]].
# User can override (including with "" to disable) via --negative-prompt.
_I2V_DEFAULT_NEGATIVE: str = "static, still, frozen, no motion"

# Extension allowlist for the --image arg. LTX VAE has been verified
# on PNG + JPEG in research; .webp / .gif / .heic / .mp4 are rejected
# because we haven't smoke-verified the VAE encode path. Users with
# HEIC inputs can route through `imgen draw` or `sips` to convert.
_I2V_INPUT_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


def resolve_i2v_motion_defaults(
    *,
    image: Path | None,
    cli_guidance: float | None,
    cli_negative: str | None,
) -> tuple[float | None, str | None]:
    """Compute the (guidance, negative_prompt) pair for the video pipeline.

    When ``image is None`` (t2v mode), pass-through: returns the CLI
    values unchanged so the t2v default-resolver downstream applies its
    own (cfg=3) defaults.

    When ``image`` is set (i2v mode), use the i2v defaults
    (:data:`_I2V_DEFAULT_CFG`, :data:`_I2V_DEFAULT_NEGATIVE`) unless
    the user explicitly overrode either flag. Override detection is
    ``is not None``, NOT truthiness: ``cli_guidance=0.0`` and
    ``cli_negative=""`` are legitimate user choices and must survive.
    """
    if image is None:
        return cli_guidance, cli_negative
    guidance = cli_guidance if cli_guidance is not None else _I2V_DEFAULT_CFG
    negative = cli_negative if cli_negative is not None else _I2V_DEFAULT_NEGATIVE
    return guidance, negative


# ── --image path validator ──────────────────────────────────────────────


def _has_unsafe_controls(name: str) -> bool:
    """True if ``name`` contains C0, DEL, or C1 control bytes.

    Mirrors :func:`imgen.inputs._has_unsafe_controls` and
    :func:`imgen.styles._is_safe_stem` — the same predicate is
    intentionally repeated rather than shared via underscore-import so
    each surface owns its own filter and a future tightening of one
    doesn't silently affect the others.
    """
    return any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in name
    )


def validate_image_path_or_die(image_arg: str | os.PathLike[str]) -> Path:
    """Resolve, expand ``~``, and verify the user-supplied conditioning image.

    Failure modes (all exit code=2 with a recovery hint, before any
    PIL/diffusers import):

    * Symlink with absolute target or path traversal — plant-attack
      signal, mirrors v0.9.1 B-14 narrow guard. Same-dir relative
      symlinks (``ln -s real.png link.png``) are allowed; the narrow
      check is ``"/" in readlink()``. TOCTOU-resilient: ``OSError``
      from readlink() falls through to the downstream existence check
      rather than propagating an unhandled exception.
    * File not found / not a file — typo or wrong arg.
    * Filename contains C0/DEL/C1 control bytes — would inject ANSI
      escapes into terminal + log surfaces (same hardening as the
      v0.3.0 batch input filter).
    * Extension not in :data:`_I2V_INPUT_EXTS` — refuse rather than
      route unverified formats to LTX VAE.

    Returns the resolved absolute Path on success.
    """
    raw = Path(image_arg).expanduser()

    # v0.9.1 B-14 mirror: narrow symlink guard BEFORE resolve(). Only
    # reject when readlink() target contains "/" (absolute or
    # traversal); same-dir relative symlinks are allowed because
    # photo-management workflows produce them legitimately.
    if raw.is_symlink():
        try:
            target = os.readlink(raw)
            if "/" in target:
                die(
                    f"video --image: symlink target contains path component: "
                    f"{str(raw)!r} → {target!r}",
                    code=2,
                    hint="Refusing to follow non-peer symlinks (plant-attack "
                         "signal — same-dir relative symlinks like `ln -s "
                         "real.png link.png` are fine). Pass the resolved "
                         "file path instead.",
                )
        except OSError:
            # TOCTOU race (symlink vanished between is_symlink() and
            # readlink()) — fall through, the existence check below
            # catches the missing file with a clean user-facing die.
            pass

    path = raw.resolve()

    if not path.exists():
        die(
            f"video --image: file not found: {path}",
            code=2,
            hint="Check the path. Use absolute path if unsure.",
        )
    if not path.is_file():
        die(
            f"video --image: not a file: {path}",
            code=2,
            hint="`--image` takes a single image file; pass a .png or "
                 ".jpg path, not a directory.",
        )
    if _has_unsafe_controls(path.name):
        # repr() escapes the unsafe bytes so the diagnostic itself
        # doesn't re-emit the escapes (mirror the styles + inputs
        # pattern — otherwise rejecting an ANSI-injection payload
        # would itself perform the injection in the warn output).
        die(
            f"video --image: filename contains unsafe control bytes: "
            f"{path.name!r}",
            code=2,
            hint="Rename the file — C0/DEL/C1 bytes can inject "
                 "terminal escape sequences into logs and prompts.",
        )
    ext = path.suffix.lower()
    if ext not in _I2V_INPUT_EXTS:
        die(
            f"video --image: unsupported extension {ext!r} for {path.name}",
            code=2,
            hint=f"Allowed: {', '.join(sorted(_I2V_INPUT_EXTS))}. Use "
                 "`imgen draw` to generate a compatible still, or `sips "
                 "-s format png in.heic --out out.png` to convert.",
        )
    return path

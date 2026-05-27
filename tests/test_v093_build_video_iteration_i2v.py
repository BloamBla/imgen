"""v0.9.3 C3 — build_video_iteration accepts image_path; validate sig
refactored to take GenParams (closes B-3).

Two interleaved refactors land together in C3 because B-3's trigger
("3rd validate-time param beyond quantize, guidance, num_frames, fps")
fires when ``image_path`` (or any future i2v-specific field) needs to
be validated at the parent-side gate. Continuing to pile on kwargs
into the validator would compound the two-step value-threading
debt (caller → kwargs → placeholder GenParams) that B-3 was filed for.

Tests in this file lock the v0.9.3 contract:

* :func:`build_video_iteration` accepts an optional ``image_path: Path
  | None = None``. When None (default), the resulting Iteration has
  ``params.input_path = None`` (v0.9.0 t2v behaviour). When set, the
  path threads through ``_assemble_iteration_no_style(input_path=...)``
  → ``GenParams.input_path``. No CLI surface yet (C5 wires it in).
* :func:`validate_engine_params_or_die` now takes a single keyword-only
  ``params: GenParams`` argument instead of per-field kwargs. Callers
  build GenParams BEFORE the validate call. This eliminates the
  placeholder-GenParams construction and lets the engine see the
  actual per-iteration shape (including ``input_path`` for future i2v
  validation hooks).
* The 2 existing production call sites
  (``build_iteration._assemble_iteration_no_style`` +
  ``build_iteration.build_iterations``) reorder to "resolve params →
  resolve loras → build GenParams → validate" — the prior ordering
  validated BEFORE LoRA resolution, but LoRA resolution is pure-string
  filtering (no SystemExit), so the reorder is observable only by the
  surfacing order of validate errors vs LoRA-incompat warnings.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ── build_video_iteration accepts + threads image_path ──────────────────


def _video_args(tmp_path, **overrides):
    """Argparse-Namespace stand-in for build_video_iteration. Defaults
    align with v0.9.0 ``imgen video`` parser output."""
    defaults = dict(
        model="ltx-video",
        prompt="a samurai walks through fog",
        steps=50,
        guidance=3.0,
        seed=42,
        width=512,
        height=512,
        num_frames=None,
        duration=None,
        fps=None,
        negative_prompt=None,
        output=None,
        loras=(),
        lora=None,
        no_lora=False,
        custom_prompt=None,
        scope=None,
        style=None,
        quantize=None,
        strength=None,
        preview=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _video_merged_defaults():
    """The full merged_defaults shape that build_video_iteration reads
    from. v0.7+ DEFAULTS owns the canonical set; copying its fields
    here keeps the test fixture decoupled from build_iteration's
    internal reads."""
    from imgen.defaults import DEFAULTS
    return dict(DEFAULTS)


def _ltx_backend():
    """Reuse the v0.9.0-derived LTX backend wrapper. BUILTIN_BACKENDS
    derives from BUILTIN_MODELS so the LTX row is registered after
    v0.9.0 commit 9 landed."""
    from imgen.backends import BUILTIN_BACKENDS
    return BUILTIN_BACKENDS["ltx-video"]


class TestBuildVideoIterationImagePath:
    """C3 acceptance — the new optional kwarg lands on the function
    signature and threads through to GenParams.input_path."""

    def test_build_video_iteration_t2v_default_input_path_is_none(
        self, tmp_path,
    ):
        """v0.9.0 backward compat: omitting image_path produces a t2v
        Iteration with ``params.input_path is None``."""
        from imgen.build_iteration import build_video_iteration

        args = _video_args(tmp_path)
        iters = build_video_iteration(
            args=args,
            prompt="a samurai walks through fog",
            merged_defaults=_video_merged_defaults(),
            be=_ltx_backend(),
            width=512, height=512,
            explicit_output=tmp_path / "out.mp4",
            run_dir=None,
            base_seed=42,
        )
        assert len(iters) == 1
        it = iters[0]
        assert it.params is not None
        assert it.params.input_path is None

    def test_build_video_iteration_i2v_threads_image_path(self, tmp_path):
        """v0.9.3 i2v: ``image_path=PATH`` lands in
        ``iteration.params.input_path`` verbatim. The Engine then
        observes ``params.input_path is not None`` and routes to the
        i2v pipeline (C4)."""
        from imgen.build_iteration import build_video_iteration

        cond_image = tmp_path / "still.png"
        cond_image.write_bytes(b"fake-png")

        args = _video_args(tmp_path)
        iters = build_video_iteration(
            args=args,
            prompt="wind blows, slow push-in",
            merged_defaults=_video_merged_defaults(),
            be=_ltx_backend(),
            width=512, height=512,
            explicit_output=tmp_path / "out.mp4",
            run_dir=None,
            base_seed=42,
            image_path=cond_image,
        )
        assert len(iters) == 1
        it = iters[0]
        assert it.params is not None
        assert it.params.input_path == cond_image


# ── validate_engine_params_or_die new signature ─────────────────────────


def _make_genparams_for_model(model, **overrides):
    """Build a validate-ready GenParams from a Model row. Reads sane
    defaults from the Model so the resulting params pass validate
    out-of-the-box; tests override specific axes."""
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="x", negative="", width=64, height=64,
        steps=model.default_steps,
        guidance=model.default_guidance,
        seed=0,
        quantize=model.supported_quants[0] if model.supported_quants else 0,
        strength=0.0,
        input_path=None,
        output_path=Path("/tmp/_validate_placeholder.png"),
        loras=(),
        mlx_cache_gb=12, battery_stop=20,
        num_frames=1, fps=24,
    )
    defaults.update(overrides)
    return GenParams(**defaults)


class TestValidateEngineParamsNewSignature:
    """The v0.9.3 signature is ``(model, *, params: GenParams)``. The
    function reads ``params.quantize`` / ``params.guidance`` /
    ``params.num_frames`` / ``params.fps`` / ``params.input_path``
    instead of accepting them as individual kwargs."""

    def test_validate_passes_when_params_in_range(self):
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["flux-kontext"]
        params = _make_genparams_for_model(model, quantize=8, guidance=3.5)
        validate_engine_params_or_die(model, params=params)
        # No SystemExit reached.

    def test_validate_dies_on_guidance_violation(self, capsys):
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["flux2-klein-edit-9b"]
        params = _make_genparams_for_model(model, quantize=4, guidance=3.5)
        with pytest.raises(SystemExit) as exc:
            validate_engine_params_or_die(model, params=params)
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "guidance" in err
        assert "3.5" in err

    def test_validate_noop_for_user_toml_none_model(self):
        """v0.7-shape Backend lookups still go through ``model=None``;
        the helper accepts it and no-ops."""
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS
        params = _make_genparams_for_model(BUILTIN_MODELS["flux-kontext"])
        validate_engine_params_or_die(None, params=params)
        # No SystemExit reached.

    def test_validate_video_reads_num_frames_from_params(self):
        """Video Model validation reads ``params.num_frames`` for
        alignment checks. C3 confirms the value flows from caller
        GenParams (no more placeholder)."""
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["ltx-video"]
        # LTX requires 8k+1 frames. 26 is invalid (must be 25, 33, ...).
        params = _make_genparams_for_model(
            model, quantize=0, guidance=3.0, num_frames=26, fps=24,
        )
        with pytest.raises(SystemExit) as exc:
            validate_engine_params_or_die(model, params=params)
        assert exc.value.code == 2

    def test_validate_video_passes_valid_num_frames(self):
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["ltx-video"]
        params = _make_genparams_for_model(
            model, quantize=0, guidance=3.0, num_frames=25, fps=24,
        )
        validate_engine_params_or_die(model, params=params)


# ── B-3 backlog closure marker ──────────────────────────────────────────


class TestB3SignatureScales:
    """Add an axis to validate without touching the helper signature —
    the whole point of B-3. Any future per-iteration field that
    affects validation rides on GenParams instead of becoming a new
    kwarg."""

    def test_input_path_field_reaches_validate_via_params(self):
        """``params.input_path`` is now visible to ``Engine.validate``.
        v0.9.3 doesn't add an i2v-specific rejection yet (Engine.run
        owns the i2v dispatch in C4), but the surface is in place so
        a future "reject i2v on a Model without i2v pipeline_class"
        rule can be added without changing the helper signature."""
        from imgen.engine_dispatch import validate_engine_params_or_die
        from imgen.models import BUILTIN_MODELS

        model = BUILTIN_MODELS["ltx-video"]
        params = _make_genparams_for_model(
            model, quantize=0, guidance=3.0, num_frames=25, fps=24,
            input_path=Path("/tmp/cond.png"),
        )
        # No SystemExit — i2v allowed at the validate gate; pipeline_
        # dispatch picks up input_path downstream in the Engine.
        validate_engine_params_or_die(model, params=params)

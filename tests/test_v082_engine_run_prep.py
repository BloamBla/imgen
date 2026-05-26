"""v0.8.2 M-1A — Iteration carries Model + GenParams for Engine.run wire-up.

Per [[project-v080-design]] §A-§E + the v0.8.2 architect pre-vet
CRITICAL-3 fix. Sub-commit M-1A is the data-shape change: each
Iteration carries its resolved Model + GenParams alongside the legacy
pre-built ``cmd`` argv. The dispatch flip (Engine.run actually called)
lands in sub-commit M-1C. These tests lock in the data invariants the
flip depends on.

Three lock-ins:

* Every Iteration produced by the 4 production build_* helpers
  (build_iterations, build_draw_iterations, build_refine_iteration,
  build_bare_i2i_iteration) has non-None ``model`` + ``params``.
  Architect MEDIUM-2 — catches construction-site drift before
  sub-commit M-1C's dispatch fence ships.
* ``apply_enhance_results_to_iterations`` dual-updates ``cmd`` AND
  ``params.prompt`` when an enhance succeeds. Architect CRITICAL-3 —
  pre-fix would silently drop enhanced prompts on Engine-dispatched
  iterations because MfluxEngine.run reads ``params.prompt``, not
  the spliced ``cmd``.
* Legacy callers constructing Iteration without ``params`` (e.g.
  test fixtures) still work — the dual-update path preserves None
  rather than crashing.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.cmd_helpers import apply_enhance_results_to_iterations
from imgen.enhance import EnhanceResult
from imgen.runs import Iteration


# ── apply_enhance dual-update (CRITICAL-3) ───────────────────────────


def _make_iter_with_params(prompt: str = "samurai") -> Iteration:
    """Construct an Iteration with ``params`` populated — matches the
    post-M-1A production shape from build_iterations /
    build_draw_iterations. v0.8.4 M-NEW-D dropped the legacy ``cmd``
    field; apply_enhance now updates ``params.prompt`` only."""
    from imgen.engines.base import GenParams
    return Iteration(
        style_name="draw",
        prompt=prompt,
        negative="",
        final_steps=20,
        final_quantize=4,
        final_guidance=3.5,
        final_strength=0.0,
        output_path=Path("/tmp/out.png"),
        params=GenParams(
            prompt=prompt,
            negative="",
            width=1024, height=1024,
            steps=20, guidance=3.5, seed=42, quantize=4, strength=0.0,
            input_path=None,
            output_path=Path("/tmp/out.png"),
            loras=(),
        ),
    )


def _make_enhance_result(original: str, enhanced: str) -> EnhanceResult:
    return EnhanceResult(
        final_prompt=enhanced,
        original_prompt=original,
        was_enhanced=True,
        fallback_reason=None,
        was_truncated=False,
        raw_llm_output=enhanced,
    )


def test_apply_enhance_updates_params_prompt():
    """v0.8.4 M-NEW-D: single-update — ``Iteration.params.prompt``
    carries the enhanced text (this is what MfluxEngine.run dispatches
    on AND what ``iteration_dryrun_display`` derives argv from for
    ``--dry-run``). The legacy ``Iteration.cmd`` field is gone; no
    dual-update needed.
    """
    it = _make_iter_with_params(prompt="samurai")
    enhanced = "a fierce samurai standing on a misty mountain at dawn"
    result = _make_enhance_result("samurai", enhanced)

    out = apply_enhance_results_to_iterations([it], [result])
    assert len(out) == 1
    new_it = out[0]

    # Iteration.prompt updated (existing v0.5 contract)
    assert new_it.prompt == enhanced
    # params.prompt carries the enhanced text — Engine.run reads this,
    # iteration_dryrun_display derives argv from it.
    assert new_it.params is not None
    assert new_it.params.prompt == enhanced


def test_apply_enhance_preserves_other_params_fields_on_single_update():
    """The single-update mutates ONLY ``params.prompt``. Every other
    GenParams field must round-trip unchanged — defence against a
    future careless rebuild that clobbers seed / steps / etc."""
    it = _make_iter_with_params(prompt="samurai")
    enhanced = "fierce samurai etc"
    result = _make_enhance_result("samurai", enhanced)

    out = apply_enhance_results_to_iterations([it], [result])
    new_params = out[0].params
    assert new_params is not None
    # Mutated:
    assert new_params.prompt == enhanced
    # All other GenParams fields unchanged:
    assert new_params.negative == ""
    assert new_params.width == 1024
    assert new_params.height == 1024
    assert new_params.steps == 20
    assert new_params.guidance == 3.5
    assert new_params.seed == 42
    assert new_params.quantize == 4
    assert new_params.strength == 0.0
    assert new_params.input_path is None
    assert new_params.output_path == Path("/tmp/out.png")


def test_apply_enhance_legacy_iteration_without_params_still_works():
    """Iterations constructed without ``params`` (legacy callers; some
    test fixtures) keep working through ``apply_enhance_results_to_
    iterations`` — params stays None, Iteration.prompt is updated, no
    crash. ``Iteration.params`` has a None default for exactly this
    transition window per architect MEDIUM-2.

    v0.8.3 M-NEW-C: such Iterations can no longer round-trip through
    ``run_one_iteration`` (which hard-asserts both fields), but the
    data-shape path through apply_enhance stays defensive.
    """
    legacy_it = Iteration(
        style_name="legacy",
        prompt="original",
        negative="",
        final_steps=20,
        final_quantize=4,
        final_guidance=3.5,
        final_strength=0.0,
        output_path=Path("/tmp/out.png"),
        # NO model, NO params — legacy shape
    )
    result = _make_enhance_result("original", "enhanced version")

    out = apply_enhance_results_to_iterations([legacy_it], [result])
    new_it = out[0]
    assert new_it.prompt == "enhanced version"
    # No params to update — stays None, no crash
    assert new_it.params is None


def test_apply_enhance_skipped_path_doesnt_touch_params():
    """When an enhance result is a fallback (was_enhanced=False),
    Iteration is returned unchanged — including ``params``. Locks in
    that the no-op branch doesn't accidentally reach into params."""
    it = _make_iter_with_params(prompt="samurai")
    original_params = it.params  # capture for identity check
    fallback = EnhanceResult(
        final_prompt="samurai",
        original_prompt="samurai",
        was_enhanced=False,
        fallback_reason="invariant_violated",
        was_truncated=False,
        raw_llm_output=None,
    )

    out = apply_enhance_results_to_iterations([it], [fallback])
    # Unchanged iteration — same Iteration object (no replace fired)
    assert out[0] is it
    assert out[0].params is original_params


# ── build_* helpers populate Model + GenParams (MEDIUM-2) ────────────


def test_build_iterations_populates_model_and_params(monkeypatch):
    """v0.8.2 architect MEDIUM-2 lock-in: every Iteration produced by
    ``build_iterations`` (the i2i preset path) carries non-None
    ``model`` + ``params``. After sub-commit M-1C's dispatch flip,
    ``run_one_iteration`` will dispatch via Engine.run for any
    iteration where ``it.model`` is non-None — if a build_* helper
    silently leaves them None, that iteration falls into the legacy
    fallback branch instead of the new Engine path. This test
    catches construction-site drift before production hits the
    fallback.
    """
    from pathlib import Path

    import imgen.cmd_helpers as ch
    from imgen.backends import BACKENDS

    # _build_args mirrors the test_generate_helpers.py shape; minimal
    # Namespace satisfying build_iterations's getattr chain.
    args = SimpleNamespace(
        style=None,
        custom_prompt=None,
        prompt_file=None,
        scope=None,
        preview=False,
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        seed=42,
        model="flux",  # v0.7 alias; resolver maps to flux-kontext
        lora=None,
        no_lora=False,
        negative_prompt=None,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    merged_defaults = dict(
        style="pixar", model="flux-kontext", backend_draw="flux-dev",
        quantize=4, steps=20, guidance=3.5, strength=0.55,
        mlx_cache_gb=12, battery_stop=20,
    )
    iters = ch.build_iterations(
        styles_list=["anime"],
        args=args,
        effective_custom_prompt=None,
        merged_defaults=merged_defaults,
        be=BACKENDS["flux"],
        input_path=Path("/tmp/photo.jpg"),
        width=1024, height=1024,
        explicit_output=None,
        run_dir=Path("/tmp/run"),
        seed=42,
    )
    assert len(iters) == 1
    it = iters[0]
    # M-1A guarantee:
    assert it.model is not None, (
        "build_iterations must populate Iteration.model "
        "for Engine.run dispatch (architect MEDIUM-2)"
    )
    assert it.params is not None, (
        "build_iterations must populate Iteration.params "
        "for Engine.run dispatch"
    )
    # Sanity: the model is the resolved v0.8 canonical Model
    assert it.model.engine == "mflux"
    # GenParams carries the same numeric quartet as Iteration
    assert it.params.steps == it.final_steps
    assert it.params.quantize == it.final_quantize
    assert it.params.guidance == it.final_guidance
    assert it.params.strength == it.final_strength
    assert it.params.seed == it.seed


def test_build_draw_iterations_populates_model_and_params():
    """Same MEDIUM-2 lock-in for the draw (t2i) path. cmd_draw's
    iterations must dispatch through Engine.run too — and
    build_draw_iterations is the sole construction site for that
    path. Architect M-1A scope: all 4 build_* helpers carry the
    payload."""
    from pathlib import Path

    import imgen.cmd_helpers as ch
    from imgen.backends import BACKENDS

    args = SimpleNamespace(
        prompt="a samurai",
        steps=None, quantize=None, guidance=None,
        seed=42,
        model="flux-dev",
        width=1024, height=1024,
        lora=None, no_lora=False,
        negative_prompt=None,
        preview=False,
        num_iterations=1,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    merged_defaults = dict(
        style="pixar", model="flux-kontext", backend_draw="flux-dev",
        quantize=4, steps=20, guidance=3.5, strength=0.55,
        mlx_cache_gb=12, battery_stop=20,
    )
    iters = ch.build_draw_iterations(
        args=args,
        prompt="a samurai",
        merged_defaults=merged_defaults,
        be=BACKENDS["flux-dev"],
        width=1024, height=1024,
        explicit_output=None,
        run_dir=Path("/tmp/run"),
        base_seed=42,
        num_iterations=1,
    )
    assert len(iters) == 1
    it = iters[0]
    assert it.model is not None
    assert it.params is not None
    assert it.model.engine == "mflux"
    # Draw is t2i → input_path is None
    assert it.params.input_path is None


# ── MfluxEngine.run argv-identity matrix (CRITICAL-2) ────────────────


def _gen_params_with(**overrides):
    """Build a GenParams with sensible defaults, overriding only what
    the matrix axis being tested cares about."""
    from imgen.engines.base import GenParams
    base = dict(
        prompt="a samurai on a misty mountain",
        negative="",
        width=1024, height=1024,
        steps=20, guidance=3.5, seed=42, quantize=4, strength=0.55,
        input_path=Path("/fake/in.png"),
        output_path=Path("/fake/out.png"),
        loras=(),
        mlx_cache_gb=12, battery_stop=20,
    )
    base.update(overrides)
    return GenParams(**base)


def _legacy_argv(backend, **kwargs):
    """Build the legacy build_mflux_cmd argv with matching defaults."""
    from imgen.backends import build_mflux_cmd
    base = dict(
        binary=Path("/fake/mflux-bin"),
        model=backend,
        input_path=Path("/fake/in.png"),
        output_path=Path("/fake/out.png"),
        prompt="a samurai on a misty mountain",
        negative="",
        quantize=4, steps=20, guidance=3.5, strength=0.55,
        seed=42, width=1024, height=1024,
        mlx_cache_gb=12, battery_stop=20,
        loras=(),
    )
    base.update(kwargs)
    return build_mflux_cmd(**base)


def test_mflux_engine_build_cmd_matches_legacy_with_negative_prompt():
    """v0.8.2 architect CRITICAL-2 lock-in (negative axis): when
    ``params.negative`` is set + model.supports_negative=True, the
    Engine path's argv MUST be byte-identical to the legacy
    build_mflux_cmd's. Pre-M-1 the legacy path was production; post-
    M-1 Engine.run owns argv via self.build_cmd. Drift = silent
    behaviour change."""
    from imgen.backends import BACKENDS
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS

    # flux-dev supports_negative=True per the registry — verified by
    # the build_mflux_cmd branch at backends.py
    model = BUILTIN_MODELS["flux-dev"]
    backend = BACKENDS["flux-dev"]

    params = _gen_params_with(
        negative="blurry, low quality, jpeg artifacts",
        input_path=None,  # flux-dev is t2i
    )
    legacy = _legacy_argv(
        backend, input_path=None,
        negative="blurry, low quality, jpeg artifacts",
    )
    new = MfluxEngine().build_cmd(
        model, params, binary=Path("/fake/mflux-bin"),
    )
    assert new == legacy, (
        f"argv drift with negative_prompt:\n"
        f"  legacy: {legacy}\n  new:    {new}"
    )
    # Sanity: negative IS in the argv (not silently dropped)
    assert "--negative-prompt" in new
    assert "blurry, low quality, jpeg artifacts" in new


def test_mflux_engine_build_cmd_matches_legacy_with_loras():
    """CRITICAL-2 lock-in (LoRA axis): a non-empty ``params.loras``
    flows through filter_compatible_loras → --lora-paths /
    --lora-scales argv slots, byte-identical to legacy."""
    from imgen.backends import BACKENDS
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS
    from imgen.styles import LoraRef

    model = BUILTIN_MODELS["flux-kontext"]
    backend = BACKENDS["flux"]

    loras = (
        LoraRef(
            ref="strangerzonehf/Flux-Cute-3D-Kawaii-LoRA",
            weight=0.85,
            compatible_with=("flux-1",),
            trigger="Pixar 3D",
        ),
    )
    params = _gen_params_with(loras=loras)
    legacy = _legacy_argv(backend, loras=loras)
    new = MfluxEngine().build_cmd(
        model, params, binary=Path("/fake/mflux-bin"),
    )
    assert new == legacy, (
        f"argv drift with LoRAs:\n  legacy: {legacy}\n  new:    {new}"
    )
    assert "--lora-paths" in new
    assert "--lora-scales" in new


def test_mflux_engine_build_cmd_matches_legacy_no_input_path():
    """CRITICAL-2 lock-in (input_path=None / t2i axis): when
    ``params.input_path is None``, the ``--image-path / --image-paths``
    slot must be ABSENT from argv. Drift would mean an empty-string
    image path leaks into mflux's argv and crashes the subprocess at
    parse time."""
    from imgen.backends import BACKENDS
    from imgen.engines.mflux_engine import MfluxEngine
    from imgen.models import BUILTIN_MODELS

    model = BUILTIN_MODELS["flux-dev"]
    backend = BACKENDS["flux-dev"]

    params = _gen_params_with(input_path=None)
    legacy = _legacy_argv(backend, input_path=None)
    new = MfluxEngine().build_cmd(
        model, params, binary=Path("/fake/mflux-bin"),
    )
    assert new == legacy
    assert "--image-path" not in new
    assert "--image-paths" not in new


# ── MfluxEngine.run KeyboardInterrupt re-raise (HIGH-2) ──────────────


def test_mflux_engine_run_propagates_keyboard_interrupt_unwrapped(
    monkeypatch,
):
    """v0.8.2 architect HIGH-2 lock-in: when ``run_with_stderr_redaction``
    raises KeyboardInterrupt (user hit Ctrl-C mid-generation), the
    Engine.run path must let it propagate UNWRAPPED. The cancel-
    history-marker side effect lives in
    ``cmd_helpers.run_one_iteration``; if Engine.run wrapped the
    exception (as RuntimeError, SystemExit, etc.) the marker would
    silently not fire and the cancelled run would be missing from
    history.jsonl."""
    from imgen.engines.base import GenParams
    from imgen.engines import mflux_engine as me
    from imgen.models import BUILTIN_MODELS

    def fake_run_with_stderr_redaction(*args, **kwargs):
        raise KeyboardInterrupt("user pressed Ctrl-C")

    monkeypatch.setattr(
        me, "run_with_stderr_redaction",
        # Bind via module-level name so the late `from ..subprocess_helpers
        # import` inside MfluxEngine.run still sees the patched fn — we
        # need to monkeypatch the import source.
        fake_run_with_stderr_redaction,
        raising=False,
    )
    # The MfluxEngine.run uses a late `from ..subprocess_helpers import`
    # so patch THAT module's binding too.
    from imgen import subprocess_helpers
    monkeypatch.setattr(
        subprocess_helpers, "run_with_stderr_redaction",
        fake_run_with_stderr_redaction,
    )

    model = BUILTIN_MODELS["flux-dev"]
    params = _gen_params_with(input_path=None)

    with pytest.raises(KeyboardInterrupt):
        me.MfluxEngine().run(model, params, env={"PATH": "/usr/bin"})


def test_mflux_engine_run_delegates_to_subprocess_helpers_with_correct_args(
    monkeypatch,
):
    """v0.8.2 architect MEDIUM-1 lock-in: MfluxEngine.run is a thin
    delegation shim over ``subprocess_helpers.run_with_stderr_redaction``.
    This test asserts the delegation contract:
      * cmd argv comes from self.build_cmd(model, params)
      * env passes through verbatim
      * log_file passes through verbatim

    Transitive HF_TOKEN redaction coverage: by proving MfluxEngine.run
    routes through run_with_stderr_redaction, all existing redaction
    tests in tests/test_subprocess_helpers.py apply to the engine
    path too. No need to duplicate the spawn-real-subprocess E2E
    here — the trust boundary is the SAME function.
    """
    from imgen.engines import mflux_engine as me
    from imgen.models import BUILTIN_MODELS

    captured = {}

    def fake_run_with_stderr_redaction(cmd, env, log_file=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["log_file"] = log_file
        return 0

    monkeypatch.setattr(
        "imgen.subprocess_helpers.run_with_stderr_redaction",
        fake_run_with_stderr_redaction,
    )

    model = BUILTIN_MODELS["flux-dev"]
    params = _gen_params_with(input_path=None)
    env = {"PATH": "/usr/bin", "HF_TOKEN": "hf_test_token_should_be_passed_through"}
    log_sentinel = object()  # any non-None marker

    rc = me.MfluxEngine().run(
        model, params, env=env, log_file=log_sentinel,
    )
    assert rc == 0

    # Delegation contract
    assert captured["cmd"][0] is not None  # build_cmd was called
    assert "--prompt" in captured["cmd"]   # mflux argv shape preserved
    # env passes through (dict() conversion preserves contents)
    assert captured["env"]["HF_TOKEN"] == "hf_test_token_should_be_passed_through"
    assert captured["env"]["PATH"] == "/usr/bin"
    # log_file passes through unchanged
    assert captured["log_file"] is log_sentinel

"""Engine.run dispatch layer — extracted from cmd_helpers in v0.8.3 (M-NEW-B).

This module owns the path from a built :class:`Iteration` through the
v0.8 Engine layer to a finished generation subprocess. Pre-v0.8.3 it
lived inside ``cmd_helpers.py``, which had grown past the project's
800-line ceiling (~2380 LoC after the v0.8.2 M-1A additions). The
extraction is pure refactor — same functions, same signatures,
same behaviour — re-exported from ``cmd_helpers`` so existing
imports (production code in ``commands/*`` + ~15 test modules) keep
working untouched.

What lives here:

* :func:`_engine_for_model` — Model.engine → Engine implementation lookup.
* :func:`validate_engine_params_or_die` — pre-iteration ``Engine.validate``
  gate; die(code=2) on any rejection.
* :func:`_genparams_from_iteration_inputs` — pack per-iteration values
  into the :class:`GenParams` payload that Engines consume.
* :func:`apply_enhance_results_to_iterations` — splice LLM-enhanced
  prompts back into the iteration plan (dual-updates both ``cmd``
  and ``params.prompt``).
* :func:`apply_enhance_results_to_groups` — :class:`PerInputBatch`
  wrapper around the per-iteration applier (cmd_batch's N×M flow).
* :func:`safe_append_history` — degrade-don't-die wrapper around
  ``history.append_history``.
* :func:`run_one_iteration` — the orchestrator: print banner, write
  log markers, dispatch through ``engine.run``, update history,
  catch KeyboardInterrupt + InsufficientRAMError uniformly.
* :func:`emit_gated_repo_hint_if_failed` — friendly HF
  license-grant hint when mflux failed on a gated repo.

What stays in cmd_helpers (and why):

* ``build_iterations`` + siblings — they construct Iterations and
  remain part of the cmd_* orchestration layer.
* ``_model_for_validate`` — Model lookup from ``args.model``, used
  by the build_* helpers BEFORE the engine layer takes over.
* ``preflight_resources``, ``print_batch_summary``, ``open_results``,
  ``exit_code`` — non-engine orchestration concerns.

No cyclic imports: ``engine_dispatch`` depends on the lower-level
``backends``, ``engines``, ``history``, ``runs``, ``subprocess_helpers``,
``colors``, ``enhance``; it does NOT depend on ``cmd_helpers``.
``cmd_helpers`` re-exports from ``engine_dispatch`` at the bottom of
its module body so the back-compat surface stays clean.
"""
from __future__ import annotations

import datetime
from dataclasses import replace as _dataclass_replace
from pathlib import Path

from .backends import Backend
from .colors import C, die, err, info, ok, step, warn
from .enhance import EnhanceResult
from .history import append_history
from .runs import BatchContext, BatchLogger, Iteration, PerInputBatch
from .subprocess_helpers import InsufficientRAMError, format_cmd

__all__ = [
    "_engine_for_model",
    "_genparams_from_iteration_inputs",
    "apply_enhance_results_to_groups",
    "apply_enhance_results_to_iterations",
    "emit_gated_repo_hint_if_failed",
    "iteration_dryrun_display",
    "run_one_iteration",
    "safe_append_history",
    "validate_engine_params_or_die",
]


# ── Engine lookup + validation gate ─────────────────────────────────────


def iteration_dryrun_display(it: Iteration) -> str:
    """Return the ``--dry-run`` display string for one Iteration.

    Engine-aware. v0.8.4 M-NEW-D replacement for the dropped
    ``Iteration.cmd`` field: pre-v0.8.4 dry-run printed
    ``format_cmd(it.cmd)`` which was the build-time argv snapshot;
    post-v0.8.4 we derive the dispatch shape from (model, params) so
    the displayed text matches whatever Engine.run will actually do.

    * ``mflux`` — return ``format_cmd(MfluxEngine.build_cmd(model,
      params))``. Byte-identical with what Engine.run dispatches; the
      v0.8.2 CRITICAL-2 lock-in
      (``test_mflux_engine_build_cmd_matches_legacy_build_mflux_cmd``)
      guarantees this stays byte-identical with the legacy
      ``backends.build_mflux_cmd`` shape that pre-v0.8.4 dry-run
      showed.
    * ``diffusers_mps`` — multi-line structured display of the
      stdin-JSON payload that DiffusersMpsEngine.run sends to
      ``_diffusers_runner``. Pre-v0.8.4 dry-run on a diffusers user-
      TOML printed the legacy mflux argv (a latent bug — no diffusers
      built-in shipped pre-v0.8.4, so dry-run on diffusers Models
      never actually fired in production). This is the corrected
      shape.

    Returns ``"(legacy Iteration — no model/params)"`` when fields
    are None — defensive for any caller bypassing the post-M-NEW-C
    invariant.
    """
    if it.model is None or it.params is None:
        return "(legacy Iteration — no model/params)"
    if it.model.engine == "mflux":
        from .engines.mflux_engine import MfluxEngine
        return format_cmd(MfluxEngine().build_cmd(it.model, it.params))
    if it.model.engine == "diffusers_mps":
        # v0.9 commit 8 (§H): branch by output_type so video-specific
        # payload fields (num_frames, fps, force_cpu_offload,
        # pipeline_class) surface in dry-run. Image path unchanged.
        if it.model.output_type == "video":
            return _format_diffusers_video_dryrun(it)
        return _format_diffusers_dryrun(it)
    raise ValueError(
        f"unknown engine={it.model.engine!r} for dry-run display"
    )


def _format_diffusers_dryrun(it: Iteration) -> str:
    """Multi-line dry-run display for diffusers_mps Iterations.

    Pretty-print the JSON payload that DiffusersMpsEngine.run streams
    to ``_diffusers_runner`` on stdin. Matches the python-style invocation
    the engine actually uses so the user sees the real dispatch shape
    (``.venv-diffusers/bin/python -m imgen.engines._diffusers_runner``)
    + the key payload fields. ``$HOME`` rewriting matches format_cmd's
    privacy-vs-discoverability trade-off.
    """
    from .paths import IMGEN_INSTALL_ROOT
    home = str(Path.home())
    runner = IMGEN_INSTALL_ROOT / ".venv-diffusers" / "bin" / "python"
    runner_str = str(runner).replace(home, "~", 1) if home else str(runner)

    model = it.model
    params = it.params
    assert model is not None and params is not None  # narrowed by caller

    def _scrub(p) -> str:
        s = str(p)
        if home and s.startswith(home):
            return "~" + s[len(home):]
        return s

    # v0.9 commit 8 (security §R.2 MEDIUM-1): safe_display() wraps
    # the prompt + negative via repr() so any C0/DEL/C1 byte in the
    # input (hand-crafted via --prompt-file PATH or stdin) renders
    # as a visible escape literal instead of triggering terminal
    # control sequences. f-string's !r conversion is structurally
    # identical to repr() — using the helper keeps the discipline
    # explicit and matches the design memo §H.
    from ._safe import safe_display
    lines = [
        f"{runner_str} -m imgen.engines._diffusers_runner",
        "  (stdin-JSON payload)",
        f"  repo:            {model.repo}",
        f"  prompt:          {safe_display(params.prompt)}",
        f"  negative:        {safe_display(params.negative)}",
        f"  steps: {params.steps}  guidance: {params.guidance}  "
        f"seed: {params.seed}  width: {params.width}  height: {params.height}",
        f"  cpu_offload_threshold_mp: {model.cpu_offload_threshold_mp}",
        f"  output_path:     {_scrub(params.output_path)}",
    ]
    if params.input_path is not None:
        lines.insert(-1, f"  input_path:      {_scrub(params.input_path)}")
    if model.param_overrides:
        overrides = dict(model.param_overrides)
        lines.append(f"  param_overrides: {overrides}")
    return "\n".join(lines)


def _format_diffusers_video_dryrun(it: Iteration) -> str:
    """v0.9 commit 8 (§H): video-shaped dry-run for diffusers_mps
    Iterations with ``model.video is not None``.

    Mirrors :func:`_format_diffusers_dryrun` shape but surfaces the
    video-specific payload fields (num_frames, fps,
    force_cpu_offload, pipeline_class, computed duration_sec).
    Same ``$HOME`` rewriting + ``safe_display()`` prompt escaping
    (security §R.2 MEDIUM-1) as the image path.

    v0.9.3 C2 (B-1 closure): ``pipeline_class`` is read from the
    Model's VideoConfig rather than hardcoded. v0.9.0 t2v Model rows
    keep showing ``"LTXPipeline"`` (the VideoConfig default); v0.9.3
    i2v shows ``"LTXImageToVideoPipeline"``. The dry-run truth must
    match the payload truth — diverging the two would let a user
    inspecting ``--dry-run`` see the wrong pipeline name.
    """
    from ._safe import safe_display
    from .paths import IMGEN_INSTALL_ROOT
    home = str(Path.home())
    runner = IMGEN_INSTALL_ROOT / ".venv-diffusers" / "bin" / "python"
    runner_str = str(runner).replace(home, "~", 1) if home else str(runner)

    model = it.model
    params = it.params
    assert model is not None and params is not None  # narrowed by caller

    def _scrub(p) -> str:
        s = str(p)
        if home and s.startswith(home):
            return "~" + s[len(home):]
        return s

    duration_sec = params.num_frames / params.fps if params.fps > 0 else 0.0
    vc = model.video
    force_offload = vc.force_cpu_offload if vc is not None else False

    pipeline_class = vc.pipeline_class if vc is not None else "LTXPipeline"
    lines = [
        f"{runner_str} -m imgen.engines._diffusers_runner",
        "  (stdin-JSON payload)",
        f"  repo:            {model.repo}",
        f"  output_type:     video",
        f"  pipeline_class:  {pipeline_class}",
        f"  prompt:          {safe_display(params.prompt)}",
        f"  negative:        {safe_display(params.negative)}",
        f"  steps: {params.steps}  guidance: {params.guidance}  "
        f"seed: {params.seed}  width: {params.width}  height: {params.height}",
        f"  num_frames: {params.num_frames}  fps: {params.fps}  "
        f"duration: {duration_sec:.2f}s",
        f"  force_cpu_offload: {force_offload}",
        f"  output_path:     {_scrub(params.output_path)}",
    ]
    if model.param_overrides:
        overrides = dict(model.param_overrides)
        lines.append(f"  param_overrides: {overrides}")
    return "\n".join(lines)


def _engine_for_model(model):
    """Return the Engine implementation matching ``model.engine``.

    v0.8.1 N-3 closure: unknown-engine path now dies with exit 2 (user
    input class) rather than raising bare ValueError. The unknown
    branch is theoretically unreachable today — ``Model.__post_init__``
    enforces ``engine in {'mflux', 'diffusers_mps'}`` at construction
    — but the v0.8.1 user-TOML schema extension widened the surface
    area enough that hardening this gate is cheap defence-in-depth.
    """
    from .engines import DiffusersMpsEngine, MfluxEngine
    if model.engine == "mflux":
        return MfluxEngine()
    if model.engine == "diffusers_mps":
        return DiffusersMpsEngine()
    die(
        f"Model {getattr(model, 'binary', None) or getattr(model, 'repo', None)!r}: "
        f"engine={model.engine!r} not recognised. "
        "Expected one of {'mflux', 'diffusers_mps'}.",
        code=2,
    )


def validate_engine_params_or_die(model, *, params: GenParams) -> None:
    """Call ``Engine.validate(model, params)`` on the resolved per-
    iteration GenParams; die with the joined error list on rejection.

    Centralised gate hit by every cmd_* (generate / batch / draw /
    refine / video). Replaces the pre-v0.8.0 hardcoded special-cases
    scattered across cmd_* (e.g. ``refine.py:238`` flux2-klein-edit-9b
    guidance pin) with a per-Model contract that scales without
    per-binary cmd_* edits.

    v0.9.3 C3 (B-3 closure): signature is now ``(model, *, params:
    GenParams)`` instead of the v0.8.0-v0.9.2 kwargs shape ``(model,
    *, quantize, guidance, num_frames=1, fps=24)``. The kwargs version
    built a placeholder GenParams internally — fine for two fields
    but didn't scale (every new validate-time field needed a new
    kwarg AND a new placeholder slot). v0.9.3 i2v adds ``input_path``
    as a validate-visible field; rather than add a 5th kwarg, the
    helper now takes the actual GenParams the caller is about to
    hand to ``Engine.run``, so validate sees the exact per-iteration
    shape including ``input_path``, ``num_frames``, ``fps``, etc.

    Callers must build the GenParams BEFORE the validate gate. In
    practice that means resolving IterationParams + loras + prompt
    first, then assembling GenParams via
    :func:`_genparams_from_iteration_inputs`, then validating. The
    pre-v0.9.3 ordering ran validate immediately after IterationParams
    resolution; the v0.9.3 reorder is observable only by surfacing
    order (validate errors land after LoRA-incompat warnings,
    instead of before them) — neither path is a hot loop.

    Still a no-op when ``model`` is None — user TOML lookups go
    through Backend, which doesn't carry v0.8 validation surface.
    """
    if model is None:
        return
    engine = _engine_for_model(model)
    errors = engine.validate(model, params)
    if errors:
        die("\n".join(errors), code=2)


# ── GenParams construction (build-time → dispatch payload) ──────────────


def _genparams_from_iteration_inputs(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    params,  # IterationParams (final_steps / final_quantize / etc.)
    seed: int,
    input_path,  # Path | None — i2i input, None for draw
    output_path,
    loras,  # tuple[LoraRef, ...]
    merged_defaults: dict,
    num_frames: int = 1,
    fps: int = 24,
):
    """v0.8.2 M-1A: pack the per-iteration inputs into a GenParams
    payload suitable for the Engine.run dispatch path.

    Pure function. Both Iteration construction sites
    (``_assemble_iteration_no_style`` + ``build_iterations``) call this
    in parallel with their existing ``build_mflux_cmd`` invocation so
    the Iteration carries BOTH shapes — the legacy ``cmd`` (consumed by
    the pre-M-1 ``run_with_stderr_redaction(cmd, ...)`` path) and the
    new ``params`` (consumed by the post-M-1 ``engine.run(model,
    params, ...)`` path). Once the legacy fallback retires (post-
    v0.8.x bake), the ``cmd`` field on Iteration can go.

    ``loras`` should be the PRE-compat-filter stack (``effective_loras``
    from ``LoraResolution``) — matches the input shape that
    ``build_mflux_cmd`` accepts AND that ``MfluxEngine.build_cmd``
    re-filters internally via ``filter_compatible_loras``. Symmetric
    construction guarantees argv bit-identity between the legacy and
    Engine paths (architect CRITICAL-2 lock-in: see
    ``test_mflux_engine_build_cmd_matches_legacy_build_mflux_cmd``).
    """
    from .engines.base import GenParams
    return GenParams(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        steps=params.final_steps,
        guidance=params.final_guidance,
        seed=seed,
        quantize=params.final_quantize,
        strength=params.final_strength,
        input_path=input_path,
        output_path=output_path,
        loras=loras,
        mlx_cache_gb=merged_defaults["mlx_cache_gb"],
        battery_stop=merged_defaults["battery_stop"],
        # v0.9 commit 7: video extensions appended. Defaults match
        # GenParams' image defaults (num_frames=1 / fps=24) — image
        # callers don't need to pass these.
        num_frames=num_frames,
        fps=fps,
    )


# ── Enhance result application ──────────────────────────────────────────


def apply_enhance_results_to_iterations(
    iterations: list[Iteration],
    enhance_results: list[EnhanceResult],
) -> list[Iteration]:
    """Splice enhanced prompts back into the iteration plan.

    Iteration is frozen+slots — we build NEW instances via
    :func:`dataclasses.replace` rather than mutating. For each
    iteration whose result was successfully enhanced, the new
    iteration carries the LLM-expanded prompt on both
    ``Iteration.prompt`` (display) and ``Iteration.params.prompt``
    (Engine.run's dispatch payload). Skipped / fallback / invariant-
    violated iterations are returned unchanged.

    v0.8.4 M-NEW-D: single-update — pre-v0.8.4 the rebuild also
    spliced the enhanced text into ``it.cmd`` via
    ``replace_prompt_in_cmd``. ``cmd`` field was retired alongside,
    so the splice is dead. Dry-run-with-enhance still shows enhanced
    text because :func:`iteration_dryrun_display` derives argv from
    ``it.params`` (which we DO update), not from the legacy snapshot.

    Asserts aligned lengths because misalignment would silently
    write the wrong prompt to the wrong iteration — louder failure
    is better.
    """
    if len(iterations) != len(enhance_results):
        raise ValueError(
            f"iteration / enhance-result count mismatch: "
            f"{len(iterations)} iterations vs {len(enhance_results)} results"
        )
    out: list[Iteration] = []
    for it, r in zip(iterations, enhance_results):
        if r.was_enhanced and r.final_prompt != it.prompt:
            new_params = (
                _dataclass_replace(it.params, prompt=r.final_prompt)
                if it.params is not None else None
            )
            out.append(_dataclass_replace(
                it, prompt=r.final_prompt, params=new_params,
            ))
        else:
            out.append(it)
    return out


def apply_enhance_results_to_groups(
    groups: list[PerInputBatch],
    enhance_results: list[EnhanceResult],
) -> list[PerInputBatch]:
    """Wrapper around :func:`apply_enhance_results_to_iterations` for
    cmd_batch's per-input shape. Eliminates the sliding-cursor block
    that used to live inline in batch.py (v0.5 architect IMP #2,
    extracted in v0.6.4 as ``apply_enhance_results_to_per_input``;
    signature promoted to :class:`PerInputBatch` in v0.6.5; renamed
    to ``_to_groups`` in v0.7.0).

    ``groups`` is a list of :class:`PerInputBatch` (one per input
    photo in cmd_batch's N×M flow). ``enhance_results`` is the FLAT
    list returned by :func:`enhance_iteration_prompts` aligned to
    ``[it for g in groups for it in g.iters]``.

    v0.7.4: the v0.7.0 ``IterationGroup`` Protocol +
    ``DrawIterationGroup`` sibling were retired — neither earned a
    real consumer in two releases (v0.7.0 wrapped a single iter; v0.7.3
    cmd_draw refactor moved to enhance-prompt-first, never building a
    group). Signature tightened to ``list[PerInputBatch]`` since
    PerInputBatch is the sole concrete shape in production. The name
    ``_to_groups`` stays as a slight readability win over
    ``_to_per_input`` (group is a meaningful noun: "the M iterations
    of one input"). If a future video / multi-shot path needs a
    Protocol-typed generalisation, resurrecting it is straightforward.

    Returns a new list of :class:`PerInputBatch` with enhanced
    prompts spliced in. Per-group lengths preserved (helper doesn't
    assume uniform group sizes; ragged groups stay intact). Uses
    :func:`dataclasses.replace` so future field additions to
    PerInputBatch propagate automatically (v0.6.5 architect FL-6).

    Pure: no I/O. Asserts the flat-shape count matches sum of group
    lengths — misalignment would silently miswire prompts.
    """
    expected_flat = sum(len(g.iters) for g in groups)
    if expected_flat != len(enhance_results):
        raise ValueError(
            f"enhance-result count mismatch: groups sum to "
            f"{expected_flat} iterations vs {len(enhance_results)} results"
        )
    out: list[PerInputBatch] = []
    cursor = 0
    for g in groups:
        group_len = len(g.iters)
        group_results = enhance_results[cursor:cursor + group_len]
        cursor += group_len
        new_iters = apply_enhance_results_to_iterations(
            list(g.iters), group_results,
        )
        out.append(_dataclass_replace(g, iters=tuple(new_iters)))
    return out


# ── History append (degrade-don't-die) ──────────────────────────────────


def safe_append_history(entry: dict) -> None:
    """Append to history, warn on unexpected failure.

    history.append_history already swallows OSError and returns 0 on
    disk-level problems (lock contention, ENOSPC). This wrapper exists
    so any *other* exception class — JSON encoding error on a weird
    value, unicode mistake in a path — degrades to a warn() instead of
    aborting :func:`run_one_iteration` between the subprocess success
    and the log end-marker. Without it, a raise here would skip the
    iteration_end marker, leaving the next iteration's start marker
    flush against this one (looks like a hung iteration in the log).
    (v0.2.4 review IMP-2 — wrap landed in v0.2.5)
    """
    try:
        append_history(entry)
    except Exception as e:  # noqa: BLE001 — degrade-don't-die is the point
        warn(f"history entry not recorded: {type(e).__name__}: {e}")


# ── One iteration: the subprocess workhorse ─────────────────────────────


def run_one_iteration(
    *,
    it: Iteration,
    idx: int,
    total: int,
    is_batch: bool,
    ctx: BatchContext,
    logger: BatchLogger | None,
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
    enhance_result: EnhanceResult | None = None,
    enhance_model: str | None = None,
) -> bool:
    """Execute one mflux iteration end-to-end.

    Steps: print banner → write log start-marker → run subprocess →
    update history → write log end-marker → append to succeeded or
    failed. Mutates the two lists (caller owns the storage; the helper
    is the producer of entries).

    `ctx` is the batch-wide BatchContext (backend, seed, dimensions,
    input path, custom prompt, args namespace, batch_id, env) — built
    once in the caller, shared across every iteration.

    Returns ``True`` to keep the batch loop going, ``False`` if the user
    pressed Ctrl-C (caller should early-exit with 130). The KeyboardInterrupt
    handler writes a `cancelled` history entry and the matching log
    marker before returning so a re-run via `imgen history --replay`
    can pick up where the interrupted batch left off.
    """
    style_name = it.style_name
    output_path = it.output_path

    if is_batch:
        step(f"Generating [{idx}/{total}] {style_name} → {output_path.name}")
    else:
        step(f"Generating {style_name} → {output_path.name}")
    print(f"   {C.DIM}model: {ctx.model} q{it.final_quantize}  "
          f"steps: {it.final_steps}  guidance: {it.final_guidance}  "
          f"strength: {it.final_strength}  seed: {it.seed}{C.END}")
    # v0.7.0: ctx.input_path is None for t2i (`imgen draw`). The display
    # line swaps in a t2i marker; the history JSONL serialises None as
    # JSON null so future replay readers see absence-as-null cleanly.
    input_display = ctx.input_path.name if ctx.input_path else "(text-to-image)"
    print(f"   {C.DIM}size: {ctx.width}x{ctx.height}  "
          f"input: {input_display} → output: {output_path}{C.END}")
    print()

    started = datetime.datetime.now()
    history_entry: dict = {
        "ts": started.isoformat(timespec="seconds"),
        "input": str(ctx.input_path) if ctx.input_path else None,
        "output": str(output_path),
        # v0.7.0: which subcommand produced this entry — drives replay
        # routing back through the right orchestrator. v=3 read-compat
        # additive (older entries fall through `entry.get("command",
        # "generate")` at the reader).
        "command": ctx.command,
        # `style` stored as the per-iteration style name when there's
        # no custom prompt — replay uses it to reload the same preset.
        "style": style_name if not ctx.effective_custom_prompt else None,
        "custom_prompt": ctx.effective_custom_prompt,
        # v0.6.5 architect IMP-A: complete the FL-3 defence. `scope` is
        # i2i-parser-specific; the future imgen draw will not declare
        # it on its Namespace. History readers already use `.get` so
        # None lands cleanly. Without this getattr run_one_iteration
        # would AttributeError mid-batch AFTER mflux had already
        # produced the image — partial run + traceback. Pre-empt here.
        # `preview` stays a direct attribute access — it's declared on
        # both i2i and t2i parsers (image-input dimension shorthand /
        # initial size), so no getattr needed.
        "scope": getattr(ctx.args, "scope", None),
        "preview": ctx.args.preview,
        "prompt": it.prompt,
        "negative": it.negative,
        # v0.7.3 fix: per-Iteration seed, NOT ctx.seed (which is the
        # base of the cmd_draw ladder; writing it to every row would
        # collapse N draw iterations onto the same recorded seed and
        # break replay reproducibility for rows 2..N).
        "seed": it.seed,
        "steps": it.final_steps,
        "guidance": it.final_guidance,
        "strength": it.final_strength,
        # v0.8.0 commit 9 (§K + §Q): history schema v=3 → v=4 KEY
        # RENAME — ``backend`` → ``model``. Value is the v0.8 canonical
        # name (translated by the parser resolver at commit 4a/4b
        # before reaching ctx). Dual-shape READ dispatch lives in
        # ``history.entry_model_name(entry)`` — old v=3 rows on disk
        # are still readable; new v=4 rows write the renamed key only.
        "model": ctx.model,
        "quantize": it.final_quantize,
        "width": ctx.width,
        "height": ctx.height,
        # v0.2.3: ties multi-style entries together. Null for single-
        # style invocations (preserves v0.2.x shape).
        "batch_id": ctx.batch_id,
        "batch_index": f"{idx}/{total}" if is_batch else None,
        # v0.6 schema v=3: COMPAT-FILTERED LoRA stack that mflux actually
        # saw on this iteration. Architect-CRITICAL #1 from the v0.6
        # pre-tag review — without this, ``imgen replay <id>`` silently
        # diverges on LoRA selection (style's current built-ins get
        # re-injected and the original --lora / --no-lora opt-outs lost).
        # Stored as a list of dicts (LoraRef is frozen+slots; replay
        # reconstructs via LoraRef(**dict)). Empty list = text-only run
        # (either no style LoRAs + no CLI LoRAs, or --no-lora dropped
        # the whole stack).
        "loras": [
            {
                "ref": lora.ref,
                "weight": lora.weight,
                "compatible_with": list(lora.compatible_with),
                "trigger": lora.trigger,
            }
            for lora in it.loras
        ],
    }

    # v0.9 commit 11.3 (§R.3 architect HIGH-1 closure / §J keystone):
    # video Models write num_frames + fps + video_codec into the
    # history entry so ``imgen replay <id>`` reproduces the temporal
    # structure exactly. The replay reader at
    # commands/history.py:_replay_video_entry uses
    # entry.get("num_frames", 25) / entry.get("fps", 24) — without
    # the write side, every video replay would silently fall back to
    # defaults regardless of the original --num-frames / --fps.
    # Schema stays v=4 additive per §J (no v=5 bump).
    if ctx.command == "video" and it.params is not None:
        history_entry["num_frames"] = it.params.num_frames
        history_entry["fps"] = it.params.fps
        # v0.9.0 ships only libx264 via imageio-ffmpeg. Future codec
        # support (WebM via libvpx-vp9) would extend the VideoConfig
        # supports_video_codecs allowlist — stored value reflects
        # what the runner actually muxed.
        history_entry["video_codec"] = "libx264"
        # v0.9.3 C6 — i2v conditioning image. Additive field per §J
        # (no v=5 bump). ABSENCE (key not present) discriminates t2v
        # entries from i2v — matches the v0.5 ``enhanced`` field
        # pattern. ``str(Path)`` so JSON serialisation produces a
        # plain string; the replay-read at
        # ``commands/history.py:_replay_video_entry`` reconstructs
        # the Path via ``Path(entry["image_path"])``.
        if it.params.input_path is not None:
            history_entry["image_path"] = str(it.params.input_path)

    # v0.5: optional LLM enhancer recording. Fields land only when the
    # enhancer was actually engaged this run (either ran or was opted-
    # out at CLI level — both signal "user knew enhance is a thing").
    # When enhance_result is None (legacy callers that haven't been
    # updated to pass it), the history entry stays in v0.4.x shape
    # except for the always-v=2 stamp added by append_history.
    #
    # The pre-enhance prompt is read directly from
    # enhance_result.original_prompt — the orchestrator captures it
    # at every fallback path, eliminating the v0.5 Phase C-1
    # parallel-list dance (which was fragile against reordering).
    if enhance_result is not None:
        history_entry["prompt_original"] = enhance_result.original_prompt
        history_entry["enhanced"] = enhance_result.was_enhanced
        # ``enhance_model`` is recorded ONLY when the LLM actually
        # produced an enhanced prompt that made it through invariants.
        # On opt-out / fallback / runner error we leave it null —
        # claiming a model "was used" when its output was discarded
        # would be misleading.
        history_entry["enhance_model"] = (
            enhance_model if enhance_result.was_enhanced else None
        )
        history_entry["enhance_fallback_reason"] = enhance_result.fallback_reason
        # v0.5 python I-4 (shipped v0.6.4): verbose diagnostic string
        # for fallback paths whose coarse token loses detail (currently
        # only "invariant_violated" — names which clause(s) the LLM
        # dropped). None for paths where the coarse token IS the full
        # story. Read-compatible additive field; v=2 readers using
        # ``entry.get`` won't see it on older entries.
        history_entry["enhance_fallback_detail"] = enhance_result.fallback_detail

    if logger is not None:
        logger.iteration_start(idx, total, style_name, started)

    try:
        # v0.8.2 M-1C dispatch flip: every production iteration post-
        # M-1A carries a resolved v0.8 Model + GenParams, and routes
        # through Engine.run. v0.8.3 (M-NEW-C) retired the legacy
        # ``run_with_stderr_redaction(it.cmd, ...)`` fallback after
        # one tag cycle per architect HIGH-1 — direct-construct test
        # fixtures (e.g. test_generate_helpers._full_iter) were
        # migrated to populate model + params so this fence stays
        # tight.
        #
        # Argv byte-identity between Engine.build_cmd and the legacy
        # ``backends.build_mflux_cmd`` is locked by
        # tests/test_v082_engine_run_prep.py
        # ``test_mflux_engine_build_cmd_matches_legacy_*`` (CRITICAL-2
        # property tests across negative-prompt / LoRA / no-input
        # axes).
        #
        # Diffusers_mps Models route through DiffusersMpsEngine.run
        # (Stati-runner subprocess via stdin-JSON) — reachable end-to-
        # end since v0.8.1 HIGH-2 closure + the M-1C dispatch flip.
        if it.model is None or it.params is None:
            # Defensive — v0.8.3 invariant. Reached only if a build_*
            # helper or test fixture silently leaves either None. The
            # MEDIUM-2 cross-build lock-in catches construction-site
            # drift before this fence ever sees it.
            raise AssertionError(
                "run_one_iteration: Iteration.model and "
                ".params must both be populated (v0.8.3 M-NEW-C "
                "invariant)"
            )
        engine = _engine_for_model(it.model)
        returncode = engine.run(
            it.model, it.params,
            env=ctx.env,
            log_file=logger.borrow_fd() if logger else None,
        )
    except KeyboardInterrupt:
        warn("Cancelled by user")
        cancel_duration = int(
            (datetime.datetime.now() - started).total_seconds())
        history_entry["status"] = "cancelled"
        history_entry["duration_sec"] = cancel_duration
        safe_append_history(history_entry)
        if logger is not None:
            logger.iteration_cancelled(idx, total, style_name, cancel_duration)
        return False
    except InsufficientRAMError as e:
        # v0.8.2 safety net hit BEFORE any mflux Popen. Defence-in-depth
        # against preflight bypass — see subprocess_helpers
        # ``_assert_safe_ram_or_raise`` docstring for the 6 scenarios
        # this catches.
        #
        # Continue the batch loop (return True) so the user sees ALL
        # affected iterations in the summary; an abrupt early-exit on
        # the first per-iteration RAM-safety failure would hide the
        # scope of the issue. Status="failed" + duration=0 records the
        # refusal in history.jsonl for replay diagnostics.
        err(f"RAM safety: {e}")
        fail_duration = int(
            (datetime.datetime.now() - started).total_seconds())
        history_entry["status"] = "failed"
        history_entry["duration_sec"] = fail_duration
        safe_append_history(history_entry)
        if logger is not None:
            logger.iteration_end(idx, total, style_name, -1, fail_duration)
        failed.append((style_name, -1, output_path))
        print()
        return True

    duration = int((datetime.datetime.now() - started).total_seconds())
    history_entry["duration_sec"] = duration
    history_entry["status"] = "success" if returncode == 0 else "failed"
    safe_append_history(history_entry)

    if logger is not None:
        logger.iteration_end(idx, total, style_name, returncode, duration)

    if returncode != 0:
        err(f"mflux exited with code {returncode} after {duration}s "
            f"— {style_name}")
        failed.append((style_name, returncode, output_path))
        # Continue with next style — don't waste already-done work.
        print()
        return True

    succeeded.append((style_name, output_path, duration))
    print()
    ok(f"Done in {duration // 60}m {duration % 60}s — {output_path}")
    print()
    return True


# ── Gated-repo failure hint ─────────────────────────────────────────────


def emit_gated_repo_hint_if_failed(
    *,
    failed: list[tuple[str, Path, int]],
    backend_obj: Backend,
) -> None:
    """Surface a friendly HF license-grant hint when mflux failed AND
    the backend declares a gated repo.

    Common failure for cold-install colleagues: their HF token IS
    valid (it authenticates fine) but they haven't accepted the
    specific model's license on HuggingFace — FLUX.1-dev and
    FLUX.1-Kontext-dev are SEPARATE gated repos with SEPARATE
    per-model license-grants. The mflux trace already says
    "Cannot access gated repo for url ..." but it's buried 30 lines
    into a stack trace; this helper surfaces the URL at the bottom
    where the user is looking after the failure summary.

    Pure side-effect (prints to stdout) on the failure path; no-op
    on success or when the backend doesn't declare ``hf_gated_repo``
    (qwen — open repo; user TOMLs that don't set the field).

    v0.7.0 originally inlined this in cmd_draw; v0.7.1 extracts so
    cmd_generate + cmd_batch get the same hint on Kontext UX gaps.
    """
    if not failed or not getattr(backend_obj, "hf_gated_repo", None):
        return
    print()
    info(
        "If mflux failed with HTTP 401 / GatedRepoError above, "
        "accept the license for this model on HuggingFace:"
    )
    print(f"   {C.DIM}https://huggingface.co/{backend_obj.hf_gated_repo}{C.END}")
    print(f"   {C.DIM}(per-repo grant — your token's access to one "
          f"gated model doesn't auto-share to siblings){C.END}")

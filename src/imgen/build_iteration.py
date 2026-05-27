"""Iteration-build helpers — extracted from cmd_helpers in v0.8.4 (M-NEW-E).

This module owns everything that takes user CLI inputs (parsed args
namespace + style preset + backend) and produces a list of fully-
resolved :class:`Iteration` objects ready for ``Engine.run`` dispatch.
Pre-v0.8.4 it lived inside ``cmd_helpers.py``, which had grown to
~1890 LoC even after the v0.8.3 M-NEW-B engine_dispatch extraction.
M-NEW-E pulls the iteration-build family out so cmd_helpers can fit
under the 800-line ceiling per ``~/.claude/rules/common/coding-style.md``.

What lives here:

* :func:`_flatten_cli_lora`, :func:`resolve_effective_loras`,
  :func:`prepend_trigger_words` — LoRA stack resolution + trigger-
  word prepending.
* :func:`check_prompt_style_compat` — pre-build invariant check.
* :func:`_model_for_validate` — args.model → v0.8 Model lookup
  (two-tier: BUILTIN_MODELS + user-TOML round-trip via
  model_from_backend).
* :class:`IterationParams`, :class:`LoraResolution` — per-iteration
  intermediate-shape dataclasses.
* :func:`_resolve_iteration_params`, :func:`_resolve_iteration_prompt`,
  :func:`_resolve_iteration_loras` — per-axis resolvers.
* :func:`prompt_slug`, :func:`_draw_output_path_for_index` —
  output-path naming.
* :func:`_assemble_iteration_no_style` — shared core for
  ``build_draw_iterations`` + ``build_refine_iteration`` (no style
  preset, no scope substitution).
* :func:`build_iterations` — i2i N×M plan builder (cmd_generate /
  cmd_batch).
* :func:`build_draw_iterations` + :func:`build_draw_iteration` — t2i
  N×1 plan builder (cmd_draw).
* :func:`build_refine_iteration` — single-iteration Hires-Fix
  (cmd_refine).
* :func:`build_bare_i2i_iteration` — single i2i with no style preset.

Re-exported from :mod:`imgen.cmd_helpers` for back-compat so existing
production callers in ``commands/*`` and ~10 test modules importing
from cmd_helpers keep working untouched.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import styles as _styles
from .backends import Backend, filter_compatible_loras, get_backend
from .colors import die, warn
from .defaults import PREVIEW_OVERRIDES
from .engine_dispatch import (
    _genparams_from_iteration_inputs,
    validate_engine_params_or_die,
)
from .images import apply_scope
from .runs import Iteration, next_available_path
from .styles import LoraRef, Style

__all__ = [
    "IterationParams",
    "LoraResolution",
    "_assemble_iteration_no_style",
    "_draw_output_path_for_index",
    "_flatten_cli_lora",
    "_model_for_validate",
    "_resolve_iteration_loras",
    "_resolve_iteration_params",
    "_resolve_iteration_prompt",
    "build_bare_i2i_iteration",
    "build_draw_iteration",
    "build_draw_iterations",
    "build_iterations",
    "build_refine_iteration",
    "check_prompt_style_compat",
    "prepend_trigger_words",
    "prompt_slug",
    "resolve_effective_loras",
]


# ── v0.6: LoRA stack resolution + trigger-word prepending ─────────────


def _flatten_cli_lora(
    cli_lora: list | None,
) -> tuple[LoraRef, ...]:
    """Normalise the ``cli_lora`` shape to a flat tuple of LoraRefs.

    Pre-v0.7.0 the CLI ``--lora`` produced ``list[LoraRef]`` (one ref
    per repeated flag). v0.7.0 added comma-split per element, so the
    argparse-collected shape became ``list[list[LoraRef]]`` (each
    repeated flag yields a list of refs from
    :func:`parser._lora_refs_arg`). Programmatic callers
    (``replay_entry`` rehydrating from history) still pass flat
    ``list[LoraRef]``. This helper accepts either shape and returns
    a flat tuple, so the precedence logic in
    :func:`resolve_effective_loras` doesn't need to care.

    Detection is element-by-element rather than depth-by-depth: each
    item is either a ``LoraRef`` (legacy flat shape) or a
    ``list[LoraRef]`` (v0.7.0 comma-split shape). Mixed inputs are
    handled gracefully — defence-in-depth against future callers.
    """
    if not cli_lora:
        return ()
    out: list[LoraRef] = []
    for item in cli_lora:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return tuple(out)


def resolve_effective_loras(
    preset: Style,
    cli_lora: list | None,
    no_lora: bool,
) -> tuple[LoraRef, ...]:
    """Combine style-declared LoRAs + CLI-supplied LoRAs into the final
    tuple that flows into ``MfluxEngine.build_cmd``.

    Precedence:

    * ``no_lora=True`` → DROP style LoRAs but KEEP ``cli_lora`` if any.
      The CLI argparse layer enforces ``--lora`` and ``--no-lora`` mutex,
      so the user can never get here with both set from the command
      line. The non-empty-cli case is reached via two programmatic
      callers: (a) ``replay_entry`` reconstructs the exact LoRA stack
      from a v=3 history entry by passing ``no_lora=True``
      + ``cli_lora=[stored_loras]`` so the style's CURRENT built-in
      LoRAs don't sneak in if the user upgraded imgen between original
      run and replay; (b) future user-style with ``loras=[]`` declared
      explicitly to override a built-in. Without this carve-out
      ``no_lora=True + cli_lora=[X]`` would return empty and silently
      drop the replay reconstruction — a Architect-CRITICAL #1 hazard.
    * Otherwise the style's ``preset.loras`` (always a tuple, default
      ``()``) provides the base stack; ``cli_lora`` (if non-None) is
      APPENDED. Order in the final tuple = style LoRAs first, CLI
      LoRAs after. mflux applies LoRAs in argv order, so the user's
      CLI additions layer ON TOP of the style's curated stack.

    ``cli_lora`` accepts both ``list[LoraRef]`` (legacy / replay) and
    ``list[list[LoraRef]]`` (v0.7.0 CLI shape after comma-split);
    normalisation happens via :func:`_flatten_cli_lora`.

    Pure: no I/O, no mutation of either input. Returns an empty
    tuple when both sources are empty / disabled.
    """
    cli_flat = _flatten_cli_lora(cli_lora)
    if no_lora:
        return cli_flat
    style_loras = preset.loras
    if not cli_flat:
        return style_loras
    return style_loras + cli_flat


def prepend_trigger_words(
    prompt: str,
    loras: tuple[LoraRef, ...],
) -> str:
    """Ensure each LoRA's ``trigger`` (if set) appears in the prompt.

    Style LoRAs often need a specific trigger word/phrase in the
    prompt to activate (e.g. "Pixar 3D" for the Canopus-Pixar-3D-Flux-
    LoRA — without that token in the prompt, the LoRA's weight delta
    has minimal effect even when loaded). This helper checks each
    LoRA's trigger against the existing prompt (case-insensitive,
    word-boundary anchored); for any missing triggers, prepends them
    comma-separated at the START of the prompt so the LoRA fires.

    Word-boundary anchoring (v0.6 python-reviewer IMP-2): a short
    trigger like ``"ani"`` (hypothetical user LoRA) would have falsely
    matched any prompt containing ``"animation"`` / ``"anime"`` /
    ``"fanatical"`` under the v0.5 unanchored ``substring in`` check.
    Built-in triggers (``"Animeo"`` / ``"Pixar 3D"`` / ``"Ghibli style"``)
    are long enough that the regression was latent, but the surface is
    public-via-user-styles. ``re.search(r"\\b{trigger}\\b", ...)``
    requires the trigger to start/end at a word boundary — handles
    multi-word triggers fine (``"Pixar 3D"`` matches in a prompt only
    when preceded + followed by non-word characters or string edges).

    Triggers already present in the prompt (because the style preset
    or user's ``--custom-prompt`` already mentions them) are left
    alone — no duplication. Caller is expected to pass the COMPATIBLE-
    filtered LoRA tuple; triggers for incompatible LoRAs would
    pollute the prompt for no benefit (the LoRA doesn't fire).

    Pure: no I/O. Returns the (possibly-prepended) prompt string.
    """
    needed: list[str] = []
    seen: set[str] = set()
    for lora in loras:
        if not lora.trigger:
            continue
        trig = lora.trigger.strip()
        if not trig:
            continue
        trig_lower = trig.lower()
        # Word-boundary match — \b in re.IGNORECASE anchors at the
        # transitions between word chars (\w = [a-zA-Z0-9_]) and
        # non-word chars. ``re.escape`` defends against trigger phrases
        # that happen to contain regex meta-characters (``.`` / ``+``
        # / ``(`` / ...). Multi-word triggers like ``"Pixar 3D"`` work
        # because ``\b`` anchors at the outer transitions; internal
        # whitespace inside the trigger matches the same whitespace
        # in the prompt verbatim.
        if re.search(rf"\b{re.escape(trig)}\b", prompt, flags=re.IGNORECASE):
            continue
        if trig_lower in seen:
            continue  # de-dup across multiple LoRAs sharing a trigger
        seen.add(trig_lower)
        needed.append(trig)
    if not needed:
        return prompt
    return ", ".join(needed) + ", " + prompt


# ── Pre-build invariant check ──────────────────────────────────────────


def check_prompt_style_compat(
    styles_list: list[str],
    effective_custom_prompt: str | None,
) -> None:
    """Reject only the genuinely incompatible (prompt, style) combos.

    v0.3.5: `--custom-prompt` now AUGMENTS full-style prompts rather
    than replacing them — the augmentation logic lives in
    :func:`build_iterations`. The only remaining incompatibility is
    "param-only style + no prompt source": a style with no built-in
    `prompt` field and no `--custom-prompt` / `--prompt-file` leaves
    the iteration with nothing to send mflux.

    Raises SystemExit(2) on the remaining incompatibility. Returns
    None on success.
    """
    if effective_custom_prompt:
        # v0.3.5: full-style + custom-prompt now augment — see
        # build_iterations. Nothing to reject here.
        return
    # No custom prompt → every listed style must have its own.
    missing_prompt = [s for s in styles_list if not _styles.get_style(s).prompt]
    if missing_prompt:
        die(f"Style(s) without a prompt: {', '.join(missing_prompt)}. "
            "Pass --custom-prompt (or --prompt-file) to supply one.",
            code=2,
            hint="Param-only styles in ~/.imgen/styles.d/ need a "
                 "CLI-supplied prompt.")


# ── Model lookup from args.model ────────────────────────────────────────


def _model_for_validate(args):
    """Return the v0.8 Model for ``args.model`` if resolvable, else None.

    v0.8.1 HIGH-2 closure: lookup is now two-tier. Built-in Models from
    ``BUILTIN_MODELS`` win; on miss, user TOMLs from the merged backend
    registry are converted to Model via ``model_from_backend`` so their
    declared v0.8 fields (engine, ram_*, default_*, ...) drive
    Engine.validate. v0.8.0 returned None for user TOMLs, leaving their
    declared param defaults effectively dead.

    Returns None only when ``args.model`` is unrecognised in either
    registry (a user passing ``--model bogus`` — error surfaced
    downstream by the get_backend call site).
    """
    from .models import BUILTIN_MODELS
    name = getattr(args, "model", None)
    if name is None:
        return None
    builtin = BUILTIN_MODELS.get(name)
    if builtin is not None:
        return builtin
    # User-TOML fallback. ``get_backend`` returns None when the name is
    # unrecognised (a real "unknown model" — let the downstream error
    # path surface that). When known, ``model_from_backend`` round-
    # trips the v0.8 fields the user declared (or sensible defaults
    # for v0.7-shape TOMLs).
    from .backends import model_from_backend
    backend = get_backend(name)
    if backend is None:
        return None
    try:
        return model_from_backend(name, backend)
    except ValueError:
        # ``Model.__post_init__`` rejected the round-trip (e.g. a
        # hand-crafted Backend with engine="diffusers_mps" but no
        # repo=). Return None so the legacy mflux path keeps working
        # — schema validation at TOML-load time is the primary gate,
        # this is defence-in-depth.
        return None


# ── Intermediate-shape dataclasses ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IterationParams:
    """Resolved numeric quartet (steps / quantize / guidance / strength)
    for one iteration — the output of :func:`_resolve_iteration_params`.

    Each value applies CLI > preview > preset > merged_defaults
    precedence (with per-Model defaults inserted between preset and
    merged_defaults for steps + guidance as of v0.8.0 commit 7). The
    dataclass is an internal hand-off between the resolver and the
    Iteration constructor; not a public surface (only
    ``build_iterations`` consumes it): the named-attribute form
    avoids the 4-tuple unpacking that was fragile against ordering
    drift (architect IMP from v0.6.2 review).
    """
    final_steps: int
    final_quantize: int
    final_guidance: float
    final_strength: float


@dataclass(frozen=True, slots=True)
class LoraResolution:
    """LoRA resolution result — the output of
    :func:`_resolve_iteration_loras`.

    Carries the full effective stack (style + CLI), the compat-
    filtered subset that mflux actually saw (post-
    filter_compatible_loras), the incompat tail (for one-pair warns),
    and the trigger-prepended prompt. Bundled into a frozen dataclass
    so the outer ``build_iterations`` loop reads as named-field access
    rather than 4-tuple unpacking.
    """
    effective_loras: tuple[LoraRef, ...]
    compatible_loras: tuple[LoraRef, ...]
    incompat_loras: tuple[LoraRef, ...]
    prompt_with_triggers: str


def _resolve_iteration_params(
    *,
    args,
    preset: Style,
    merged_defaults: dict,
    model=None,
) -> IterationParams:
    """Apply CLI > preview > preset > model.default_* > merged_defaults
    precedence and return the resolved numeric quartet (steps / quantize /
    guidance / strength).

    Order (per-axis):

      * ``steps``    : CLI > preview > model.default_steps > merged_defaults
                       (preset.steps intentionally NOT honoured —
                       preset is a style preset, not a CLI override;
                       v0.6.2 design lock-in)
      * ``quantize`` : CLI > preview > merged_defaults
      * ``guidance`` : CLI > preset > model.default_guidance > merged_defaults
      * ``strength`` : CLI > preset > merged_defaults

    v0.8.0 commit 7 (§M) added the per-Model default_steps /
    default_guidance layer in the steps/guidance precedence chain.
    When non-None (i.e. a built-in Model from BUILTIN_MODELS),
    ``model.default_steps`` and ``model.default_guidance`` slot in
    between preset and merged_defaults. When None (user-TOML lookup
    or test fixture not setting model), behaves as v0.7.x.

    Extracted v0.6.4 from ``build_iterations`` per the v0.6.2 architect
    IMP-2 split. Pure: no I/O, no mutation.
    """
    if args.steps is not None:
        final_steps = args.steps
    elif args.preview:
        final_steps = PREVIEW_OVERRIDES["steps"]
    elif model is not None:
        # Per-Model default (commit 7). Built-in models declare
        # default_steps explicitly per §G.1; the dataclass default
        # (20) matches DEFAULTS["steps"] so the fallback through
        # this branch is a no-op for FLUX-family but lifts e.g.
        # Qwen-Image-Edit to its 30-step recommendation.
        final_steps = model.default_steps
    else:
        final_steps = merged_defaults["steps"]

    if args.quantize is not None:
        final_quantize = args.quantize
    elif args.preview:
        final_quantize = PREVIEW_OVERRIDES["quantize"]
    elif model is not None and not model.supported_quants:
        # v0.9 commit 8: bf16-only Models (supported_quants=()) default
        # to quantize=0 — no MLX-style quantization. Without this, the
        # merged_defaults["quantize"] (typically 8) fallback would
        # trip the §R.2 quantize gate in
        # DiffusersMpsEngine._validate_video. LTX-Video at v0.9.0 is
        # the only such Model; future bf16-only Models inherit.
        final_quantize = 0
    else:
        final_quantize = merged_defaults["quantize"]

    if args.guidance is not None:
        final_guidance = args.guidance
    elif preset.guidance is not None:
        final_guidance = preset.guidance
    elif model is not None:
        # Per-Model default (commit 7). flux2-klein-edit-9b ships
        # default_guidance=1.0 (the mflux-pinned value); FLUX.1
        # family ships 3.5. Replaces the pre-commit-7 refine.py:238
        # hardcoded `args.guidance = 1.0` override per §M.
        final_guidance = model.default_guidance
    else:
        final_guidance = merged_defaults["guidance"]

    # v0.7.0: ``args.strength`` is i2i-only (no source photo to
    # interpolate against for t2i). The `imgen draw` parser omits the
    # flag entirely; mflux ignores the recorded value on the t2i
    # backend (Backend.supports_strength=False gates argv emission
    # in MfluxEngine.build_cmd). Same getattr pattern as v0.6.5's
    # args.scope FL-3 defence.
    cli_strength = getattr(args, "strength", None)
    if cli_strength is not None:
        final_strength = cli_strength
    elif preset.strength is not None:
        final_strength = preset.strength
    else:
        final_strength = merged_defaults["strength"]

    return IterationParams(
        final_steps=final_steps,
        final_quantize=final_quantize,
        final_guidance=final_guidance,
        final_strength=final_strength,
    )


def _resolve_iteration_prompt(
    *,
    preset: Style,
    args,
    effective_custom_prompt: str | None,
    style_was_explicit: bool,
) -> str | None:
    """Resolve the prompt text for one iteration. 3-way dispatch:

      * explicit full-style + ``--custom-prompt``     → AUGMENTATION
        (preset prompt with scope applied + ``", " + custom``)
      * any ``--custom-prompt`` else                   → custom verbatim
        (covers param-only styles + the v0.3.5 bare-custom-prompt UX
        fix where the default style's prompt is bypassed)
      * no ``--custom-prompt``                         → preset.prompt
        with optional scope substitution

    Extracted v0.6.4 from ``build_iterations`` per the v0.6.2 architect
    IMP-2 split. Pure: no I/O, no mutation. Returns ``None`` for the
    param-only-style-without-custom-prompt case (caller passes through
    to mflux; mflux requires a prompt so an empty value will fail
    cleanly there).

    v0.6.5 (architect FL-3): ``args.scope`` is read via ``getattr`` —
    ``--scope`` is photo-input-specific (i2i-only) and the
    ``imgen draw`` subparser omits it. Pre-emptive defence so this
    helper drops cleanly into the t2i path without a ``--scope=None``
    workaround on the draw parser.
    """
    scope = getattr(args, "scope", None)
    scene_suffix = preset.scene_suffix
    preset_prompt = preset.prompt
    if effective_custom_prompt and preset_prompt and style_was_explicit:
        # v0.3.5 augmentation: explicit full-style + custom-prompt → the
        # preset prompt is the BASE (scope applied to it), then the
        # user's --custom-prompt text is appended as a final detail.
        # Lets the user share one common addition ("wearing a red
        # kimono") across multiple styles in the same invocation via
        # `-s anime,ghibli,pixar --custom-prompt "..."`.
        #
        # Scope applies only to the base — the user's added text is
        # passed through verbatim so scope-mode replacements don't
        # accidentally touch user wording (e.g. their literal "this
        # person" stays "this person", not rewritten).
        base = preset_prompt
        if scope:
            base = apply_scope(base, scope, scene_suffix=scene_suffix)
        return base + ", " + effective_custom_prompt
    if effective_custom_prompt:
        # Custom-only path: either a param-only style (no `prompt`
        # field) or the v0.3.5 bare-custom-prompt UX fix (no explicit
        # --style → default style's params apply but its prompt is
        # bypassed so "Pixar 3D + sepia" nonsense doesn't happen).
        return effective_custom_prompt
    # No custom-prompt → preset prompt is the prompt.
    prompt = preset_prompt
    if scope:
        prompt = apply_scope(prompt, scope, scene_suffix=scene_suffix)
    return prompt


def _resolve_iteration_loras(
    *,
    preset: Style,
    args,
    be: Backend,
    prompt: str,
) -> LoraResolution:
    """Resolve the LoRA stack for one iteration end-to-end.

    Stages:

      1. ``resolve_effective_loras`` — combine preset + CLI.
      2. ``filter_compatible_loras`` — drop entries not in the
         backend's compat group; collect the dropped tail for the
         once-per-pair warn.
      3. ``prepend_trigger_words`` — auto-prepend missing triggers to
         the prompt.

    Returns :class:`LoraResolution` bundling the four results. Pure:
    no I/O. The warn for incompat LoRAs is emitted once-per-pair by
    the caller (build_iterations) using a shared
    ``warned_incompat_loras`` set across the N×M loop.

    v0.7.8: previously inline in build_iterations; extracted so
    build_draw_iteration (t2i, no style preset) can reuse the same
    resolution-and-trigger discipline.
    """
    effective_loras = resolve_effective_loras(
        preset=preset,
        cli_lora=getattr(args, "lora", None),
        no_lora=getattr(args, "no_lora", False),
    )
    compatible_loras, incompat_loras = filter_compatible_loras(
        effective_loras, be,
    )
    prompt_with_triggers = prepend_trigger_words(prompt, compatible_loras)
    return LoraResolution(
        effective_loras=effective_loras,
        compatible_loras=compatible_loras,
        incompat_loras=incompat_loras,
        prompt_with_triggers=prompt_with_triggers,
    )


# ── Output-path naming ──────────────────────────────────────────────────


def prompt_slug(prompt: str, max_words: int = 6, max_len: int = 60) -> str:
    """Slugify a t2i prompt into a filesystem-safe stem.

    Algorithm:

      1. Take first ``max_words`` whitespace-tokens.
      2. NFKD-normalize + strip non-ASCII (CJK → empty after this step).
      3. Lowercase + replace any non-[a-z0-9-_] char with '-'.
      4. Collapse runs of '-' to a single '-'; strip leading/trailing.
      5. Truncate to ``max_len`` chars.
      6. Empty result → fall back to ``"draw"`` (always a valid stem).

    Pure: no I/O. Same generic recipe for any text-to-stem helper —
    architect §D from the v0.7.0 design.
    """
    tokens = prompt.split()[:max_words]
    text = " ".join(tokens)
    # NFKD-normalize accented chars to their bare ASCII counterparts;
    # encode/decode round-trip drops anything that doesn't have an
    # ASCII representation (CJK → empty after this, falls through to
    # the "draw" fallback).
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    text = text[:max_len].rstrip("-")
    return text or "draw"


def _draw_output_path_for_index(
    *,
    run_dir: Path,
    slug: str,
    idx: int,
    num_iterations: int,
) -> Path:
    """Compute the per-iteration output path for an N-iteration draw run.

    N==1: ``<run_dir>/<slug>.png`` (with next_available_path
    collision-suffix appended if a file by that name already exists
    in the run-dir).
    N>=2: ``<run_dir>/<slug>-<idx>.png``, idx 1-based.

    Pure: probes the filesystem read-only.
    """
    if num_iterations == 1:
        return next_available_path(run_dir, slug, suffix=".png")
    indexed = f"{slug}-{idx}"
    return next_available_path(run_dir, indexed, suffix=".png")


# ── Shared core for naked-iteration callers (draw / refine) ─────────────


def _assemble_iteration_no_style(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    input_path: Path | None,
    output_path: Path,
    width: int,
    height: int,
    seed: int,
    style_name: str,
    negative: str = "",
    num_frames: int = 1,
    fps: int = 24,
) -> Iteration:
    """Shared core for `build_draw_iterations` + `build_refine_iteration`
    (v0.7.8 refactor — closes python NIT #5 + architect NIT #F from
    the v0.7.5 review trail; the 3rd-instance-becomes-pattern threshold
    was crossed when `imgen refine` shipped).

    Contract: ``style_name`` is a FREE-FORM LABEL recorded on the
    returned :class:`Iteration` for history.jsonl + UI display
    purposes ONLY. The helper does NOT load a :class:`Style` preset
    by this name — preset is always an empty :class:`Style` (no
    prompt, no negative, no scope_suffix, no LoRAs). Pass
    ``style_name="draw"`` / ``"refine"`` / ``"video-frame"`` etc.;
    if you need real style preset loading, route through
    :func:`build_iterations` instead.

    "No-style" = empty :class:`Style` preset, no scope-substitution
    prompt rewrite, no cross-style incompat-LoRA accumulator. These
    are the distinguishing concerns of :func:`build_iterations` (i2i
    with real style presets) — which is intentionally NOT reduced
    through this helper because its per-style loop owns prompt
    augmentation + accumulation logic that doesn't generalise to
    the empty-preset callers.

    Pure: no I/O, no subprocess, no mutation. Caller (cmd_draw /
    cmd_refine) owns the output-path naming choice (slug-with-index
    vs ``<stem>-refined.png``) and the iteration count (N vs 1).

    Trade-off note: collapses two callers' independent
    `_resolve_iteration_params` + `_resolve_iteration_loras` calls
    into the helper, so `build_draw_iterations` with N>=2 now
    resolves LoRAs N times (was 1× pre-refactor — micro-optimisation
    dropped). Both helpers are pure-string-filter pure-function;
    cost is measured-negligible on N up to the 32 cap.
    """
    preset = Style()
    model = _model_for_validate(args)
    params = _resolve_iteration_params(
        args=args, preset=preset, merged_defaults=merged_defaults,
        model=model,
    )
    # v0.7.11 (gap 1): draw now exposes --negative-prompt via CLI, so
    # the caller (`build_draw_iterations`) passes through args.negative_prompt
    # via the `negative` parameter. Refine intentionally passes "" (empty)
    # because style-inherited negatives fight the Hires-Fix goal of
    # preserving input. Default "" keeps refine's pre-v0.7.11 behaviour.
    lora_resolution = _resolve_iteration_loras(
        preset=preset, args=args, be=be, prompt=prompt,
    )
    # Note: ``lora_resolution.incompat_loras`` is intentionally
    # dropped on the floor here — naked callers have no cross-style
    # accumulator like :func:`build_iterations`' per-style loop. A
    # user --lora pointing at a backend-incompat LoRA gets silently
    # filtered (warn is on the build_iterations path only). Matches
    # pre-v0.7.8 behaviour of both draw + refine.
    #
    # v0.8.4 M-NEW-D: build_mflux_cmd no longer called at iteration-
    # build time — MfluxEngine.build_cmd is invoked inside
    # MfluxEngine.run (production) and iteration_dryrun_display
    # (--dry-run preview). Saves one redundant argv build per iteration.
    #
    # v0.8.2 M-1A: attach the resolved Model + GenParams so
    # Engine.run dispatch in ``run_one_iteration`` can route through
    # ``engine.run(it.model, it.params, ...)``. Both fields are non-
    # None for every recognised --model (BUILTIN_MODELS + user TOMLs
    # via _model_for_validate); a None model here only happens if a
    # caller bypasses the upstream resolver die() — caught by
    # run_one_iteration's M-NEW-C invariant.
    gen_params = _genparams_from_iteration_inputs(
        prompt=lora_resolution.prompt_with_triggers,
        negative=negative,
        width=width,
        height=height,
        params=params,
        seed=seed,
        input_path=input_path,
        output_path=output_path,
        loras=lora_resolution.effective_loras,
        merged_defaults=merged_defaults,
        num_frames=num_frames,
        fps=fps,
    )
    # v0.9.3 C3 (B-3 closure): validate the resolved GenParams. Moved
    # AFTER GenParams construction so the helper sees the actual per-
    # iteration shape (input_path, num_frames, fps, etc.) instead of a
    # placeholder built from per-field kwargs.
    validate_engine_params_or_die(model, params=gen_params)
    return Iteration(
        style_name=style_name,
        prompt=lora_resolution.prompt_with_triggers,
        negative=negative,
        final_steps=params.final_steps,
        final_quantize=params.final_quantize,
        final_guidance=params.final_guidance,
        final_strength=params.final_strength,
        output_path=output_path,
        loras=lora_resolution.compatible_loras,
        seed=seed,
        model=model,
        params=gen_params,
    )


# ── t2i builders (cmd_draw) ──────────────────────────────────────────────


def build_draw_iterations(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    base_seed: int,
    num_iterations: int = 1,
) -> list[Iteration]:
    """Build N Iteration objects for a single t2i ``imgen draw`` call.

    Seed ladder: each iteration uses ``base_seed + i`` (mod 2**32 to
    keep within mflux's accepted range), so a single CLI invocation
    with ``-n 3`` produces three deterministically-different images.
    Pre-v0.7.3 wrote ``ctx.seed`` (= base_seed) to every Iteration,
    silently breaking ``imgen replay`` for rows 2..N (each replay
    regenerated row 1's image).

    Output naming:
      * N==1: ``<run_dir>/<prompt_slug>.png`` (collision-suffix
        appended if needed) OR ``explicit_output`` if user passed
        ``--output FILE``.
      * N>=2: ``<run_dir>/<prompt_slug>-<idx>.png``. ``--output FILE``
        is parser-rejected with N>=2 (the file would get overwritten N
        times).

    Pure: no I/O beyond the next_available_path read-only probe.
    Caller (cmd_draw) hands the resulting list to its run loop, which
    iterates :func:`run_one_iteration` over the returned list.
    """
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
    if explicit_output is not None and num_iterations > 1:
        raise ValueError(
            "explicit_output is mutex with num_iterations > 1 "
            "(single --output FILE can't fan out to N files)"
        )

    iterations: list[Iteration] = []
    slug = prompt_slug(prompt)
    for idx in range(1, num_iterations + 1):
        seed = (base_seed + idx - 1) % (2**32)
        if explicit_output is not None:
            # --output FILE only works with N=1 (parser-enforced);
            # the path is used directly.
            output_path = explicit_output
        else:
            if run_dir is None:
                raise ValueError(
                    "build_draw_iterations: either explicit_output or "
                    "run_dir must be provided"
                )
            output_path = _draw_output_path_for_index(
                run_dir=run_dir, slug=slug, idx=idx,
                num_iterations=num_iterations,
            )
        # v0.7.11 (gap 1): draw exposes --negative-prompt; build_draw_iteration
        # passes it through via the ``negative`` keyword. Refine
        # passes "" explicitly (style-inherited negatives fight
        # Hires-Fix).
        negative = getattr(args, "negative_prompt", None) or ""
        iterations.append(_assemble_iteration_no_style(
            args=args,
            prompt=prompt,
            merged_defaults=merged_defaults,
            be=be,
            input_path=None,  # t2i — no source photo
            output_path=output_path,
            width=width,
            height=height,
            seed=seed,
            style_name="draw",
            negative=negative,
        ))
    return iterations


# ── t2v builders (cmd_video) ─────────────────────────────────────────────


def _resolve_video_frames_and_fps(args) -> tuple[int, int]:
    """Resolve (num_frames, fps) from args.{num_frames,duration,fps}
    via the model's VideoConfig (alignment + offset + default_fps).

    Per §I.1 parser stanza — three input paths:
      * --num-frames N (explicit, wins)
      * --duration S (mutex with --num-frames at parser; ceil UP to
        nearest alignment so output is >= requested duration per
        architect §R.1 MED-2)
      * Neither — fall back to model.video.default_num_frames.

    fps defaults to model.video.default_fps unless user passed --fps.

    Raises ValueError if the resolved model is not a video Model
    (model.video is None) — cmd_video gates upstream so this should
    never fire in practice; defensive.
    """
    model = _model_for_validate(args)
    if model is None or model.video is None:
        raise ValueError(
            "build_video_iteration: model does not declare a "
            "VideoConfig — args.model must reference a video Model"
        )
    vc = model.video

    explicit_num_frames = getattr(args, "num_frames", None)
    explicit_duration = getattr(args, "duration", None)

    if explicit_num_frames is not None:
        num_frames = explicit_num_frames
    elif explicit_duration is not None:
        # Ceil to next-valid-alignment so output >= requested duration.
        # n = offset + k * alignment, find smallest n >= target.
        target = int(float(explicit_duration) * vc.default_fps)
        if target < vc.num_frames_offset:
            num_frames = vc.num_frames_offset
        else:
            k = (target - vc.num_frames_offset
                 + vc.num_frames_alignment - 1) // vc.num_frames_alignment
            num_frames = k * vc.num_frames_alignment + vc.num_frames_offset
        # v0.9.2 B-5: surface the ceil so the user knows the output is
        # longer than requested. design memo §I.1: "Warn line if
        # rounding occurred."
        if num_frames != target:
            actual_duration = num_frames / vc.default_fps
            warn(
                f"--duration {float(explicit_duration):.2f}s rounded UP to "
                f"{num_frames} frames at {vc.default_fps} fps "
                f"(≈{actual_duration:.2f}s output) to fit alignment "
                f"{vc.num_frames_alignment}k+{vc.num_frames_offset}"
            )
    else:
        num_frames = vc.default_num_frames

    fps = getattr(args, "fps", None) or vc.default_fps
    return num_frames, fps


def build_video_iteration(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    base_seed: int,
    num_iterations: int = 1,
    image_path: Path | None = None,
) -> list[Iteration]:
    """v0.9 commit 7: build the single Iteration for a ``imgen video``
    call.

    Returns a one-element list so the orchestrator's run loop iterates
    uniformly across draw (N iterations) and video (always 1 in
    v0.9.0). The seed-ladder + ``--num-iterations N`` shape is shared
    with build_draw_iterations but v0.9.0 video is single-shot per
    §I parser stanza ("No --num-iterations N for v0.9.0. Video is
    expensive enough that single-shot per call is the right UX;
    batching is v0.9.x.").

    The signature mirrors build_draw_iterations so the orchestrator
    can pass ``build_iterations_fn=build_video_iteration`` and
    ``build_iterations_fn=build_draw_iterations`` interchangeably.
    Extra ``num_iterations`` arg accepted for symmetry; values >1 are
    rejected at the parser level so this helper always sees 1.

    v0.9.3 C3: ``image_path`` parameter accepts a validated
    conditioning-image path for i2v mode. None (default) → t2v
    behaviour preserved. When set, the path threads to
    ``Iteration.params.input_path``; the engine reads
    ``params.input_path is not None`` to flip from
    ``LTXPipeline`` to ``LTXImageToVideoPipeline`` at dispatch (C4).
    The path itself is pre-validated at the CLI boundary (C5 via
    :func:`imgen._i2v_resolve.validate_image_path_or_die`) so this
    helper assumes the path exists, is a file, and is an LTX-VAE-safe
    extension.

    Output naming:
      * ``<run_dir>/<prompt_slug>.mp4`` with collision suffix if needed.
      * OR ``explicit_output`` if user passed ``--output PATH`` (.mp4
        extension enforced at parser-level for video).

    Pulls ``args.num_frames`` + ``args.fps`` into the resulting
    Iteration's GenParams. The parser-side --duration/--num-frames
    mutex already resolved them into a single num_frames int.
    """
    if num_iterations != 1:
        raise ValueError(
            f"build_video_iteration: v0.9.0 supports single-shot only "
            f"(got num_iterations={num_iterations}); --num-iterations "
            "is deferred to v0.9.x"
        )
    if explicit_output is not None:
        output_path = explicit_output
    else:
        if run_dir is None:
            raise ValueError(
                "build_video_iteration: either explicit_output or "
                "run_dir must be provided"
            )
        slug = prompt_slug(prompt)
        output_path = next_available_path(run_dir, slug, suffix=".mp4")

    # Negative prompt: video parser may expose --negative-prompt
    # (mirroring draw); if absent, default to "".
    negative = getattr(args, "negative_prompt", None) or ""

    # Resolve num_frames + fps from args via model.video config. Three
    # paths per §I parser stanza:
    #   * --num-frames N — explicit; wins.
    #   * --duration S (mutex with --num-frames at parser) — round UP
    #     to nearest valid alignment so output is >= requested seconds.
    #   * Neither — fall back to model.video.default_num_frames.
    # fps defaults to model.video.default_fps unless user supplies --fps.
    num_frames, fps = _resolve_video_frames_and_fps(args)

    iteration = _assemble_iteration_no_style(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        input_path=image_path,  # v0.9.3: None → t2v, set → i2v
        output_path=output_path,
        width=width,
        height=height,
        seed=base_seed,
        style_name="video",
        negative=negative,
        num_frames=num_frames,
        fps=fps,
    )
    return [iteration]


def build_draw_iteration(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
) -> Iteration:
    """v0.7.0 singular helper — kept as a backward-compat wrapper over
    :func:`build_draw_iterations` (N=1). cmd_draw uses the plural
    form directly since v0.7.3; this thin wrapper remains for any
    external programmatic caller (notebook code, tests) that built
    against v0.7.0–v0.7.2.

    See :func:`build_draw_iterations` for the contract.
    """
    return build_draw_iterations(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        width=width,
        height=height,
        explicit_output=explicit_output,
        run_dir=run_dir,
        base_seed=seed,
        num_iterations=1,
    )[0]


# ── Refine builder (cmd_refine) ─────────────────────────────────────────


def build_refine_iteration(
    *,
    args,
    input_path: Path,
    prompt: str,
    merged_defaults: dict,
    be,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
) -> Iteration:
    """Build a single :class:`Iteration` for `imgen refine` (v0.7.5).

    Refine is a Hires-Fix i2i pass — input image at any resolution →
    output at scaled resolution (typically 1.5× / 2× / fixed
    --width/--height). NO style machinery (refine has a fixed prompt
    or user override), NO scope substitution, NO trigger-word
    prepending against built-in style LoRAs.

    Differences from :func:`build_draw_iteration` (t2i):
      * Has an input photo (``input_path``) — flows through
        ``--image-paths`` (or ``--image-path``) argv via
        :func:`MfluxEngine.build_cmd`.
      * Single iteration always (no ladder; --num-iterations is
        a draw-only concept for now).
      * Output naming: ``<run_dir>/<input.stem>-refined.png`` to
        mark the file as the refined variant (vs the bare
        ``<slug>.png`` for draw).

    Reuses :func:`_resolve_iteration_params` and
    :func:`_resolve_iteration_loras` with a stub empty
    :class:`Style` — same pattern as build_draw_iterations. CLI
    ``--lora REF`` flows through; preset LoRAs are intentionally
    not in play.

    Pure: no subprocess, no I/O beyond the next_available_path
    probe.
    """
    if explicit_output is not None:
        output_path = explicit_output
    else:
        if run_dir is None:
            raise ValueError(
                "build_refine_iteration: either explicit_output or "
                "run_dir must be provided"
            )
        # `<input.stem>-refined.png`. next_available_path handles
        # collisions if the user re-refines into the same run-dir.
        output_path = next_available_path(
            run_dir, f"{input_path.stem}-refined", suffix=".png",
        )

    # v0.7.8: shared core with build_draw_iterations. Naked iteration
    # (empty Style preset, no incompat accumulator).
    return _assemble_iteration_no_style(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        seed=seed,
        style_name="refine",
    )


def build_bare_i2i_iteration(
    *,
    args,
    input_path: Path,
    prompt: str,
    merged_defaults: dict,
    be,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
    style_name: str = "generate",
) -> Iteration:
    """Build a single :class:`Iteration` for naked i2i (cmd_generate
    with ``--style none``).

    Reuses :func:`_assemble_iteration_no_style`. The ``style_name``
    label lands on the Iteration (for history.jsonl). Output path
    falls back to ``<run_dir>/<input.stem>.png`` (with collision
    suffix) when ``explicit_output`` is None.

    Pure: no I/O beyond next_available_path probe.
    """
    if explicit_output is not None:
        output_path = explicit_output
    else:
        if run_dir is None:
            raise ValueError(
                "build_bare_i2i_iteration: either explicit_output or "
                "run_dir must be provided"
            )
        output_path = next_available_path(
            run_dir, input_path.stem, suffix=".png",
        )
    return _assemble_iteration_no_style(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        seed=seed,
        style_name=style_name,
    )


# ── i2i N×M builder (cmd_generate / cmd_batch) ─────────────────────────


def build_iterations(
    *,
    styles_list: list[str],
    args,
    effective_custom_prompt: str | None,
    merged_defaults: dict,
    be,
    input_path: Path,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
    warned_incompat_loras: set | None = None,
) -> list[Iteration]:
    """Build M Iteration objects — one per style — for a single i2i
    invocation (``imgen generate`` per-input, or ``imgen batch``
    per-input within the N×M flow).

    For each style:
      1. Resolve :class:`IterationParams` (CLI > preview > preset >
         model.default_* > merged_defaults).
      2. Validate against ``Engine.validate`` (die-with-2 on rejected
         combos — quantize ∉ supported_quants, guidance out of range).
      3. Resolve prompt (effective_custom_prompt augments preset.prompt;
         scope substitution applied; v0.3.5+).
      4. Resolve LoRAs (preset + CLI, compat-filter, trigger-prepend).
      5. Pack GenParams payload + populate Iteration with model+params
         for Engine.run dispatch.

    ``warned_incompat_loras``: caller-provided set accumulating
    (backend_group, ref) pairs that already produced a warn; lets
    cmd_batch's N×M loop emit one warn per pair across all inputs
    (architect IMP-3 from v0.5 review). When None, the warn-set is
    process-local to this call.

    Pure: no I/O beyond next_available_path probe + the warn() side
    effect for newly-seen incompat LoRA pairs.
    """
    iterations: list[Iteration] = []
    incompat_keys: set = set()
    incompat_details: dict = {}
    # v0.3.5 augmentation key: explicit ``--style`` + ``--custom-prompt``
    # combines preset prompt + user text; bare ``--custom-prompt``
    # without an explicit style replaces the default style's prompt.
    style_was_explicit = bool(getattr(args, "style", None))
    for style_name in styles_list:
        preset = _styles.get_style(style_name)
        # Per-style param resolution (CLI > preview > preset >
        # model.default_* > merged_defaults).
        model = _model_for_validate(args)
        params = _resolve_iteration_params(
            args=args, preset=preset, merged_defaults=merged_defaults,
            model=model,
        )
        # v0.9.3 C3 (B-3 closure): validate_engine_params_or_die now
        # takes the resolved GenParams; the call moves to after
        # ``gen_params = _genparams_from_iteration_inputs(...)`` below.
        # See the engine_dispatch helper docstring for the reorder
        # rationale (validate vs LoRA-incompat warn surfacing order).

        # Per-style prompt resolution. v0.3.5: --custom-prompt
        # augments preset.prompt (when style was explicit). Scope
        # substitution applied per args.scope (v0.3.2+).
        prompt = _resolve_iteration_prompt(
            preset=preset,
            args=args,
            effective_custom_prompt=effective_custom_prompt,
            style_was_explicit=style_was_explicit,
        )
        negative = preset.negative

        # 3. Output path: explicit --output FILE (legacy single-file
        # path) wins; otherwise <run_dir>/<input.stem>-<style>.png.
        # ``-<style>`` suffix lands even for single-style runs — keeps
        # the filename schema consistent across single- and multi-
        # style invocations (no surprise rename if a user later adds
        # `-s anime,ghibli`).
        if explicit_output is not None:
            output_path = explicit_output
        else:
            if run_dir is None:
                raise ValueError(
                    "build_iterations: either explicit_output or "
                    "run_dir must be provided"
                )
            output_path = run_dir / f"{input_path.stem}-{style_name}.png"

        # Per-style LoRA resolution + trigger-word prepending +
        # backend-compat filter. Tracks the dropped tail so the cross-
        # input warn-once works in cmd_batch's N×M flow.
        lora_resolution = _resolve_iteration_loras(
            preset=preset, args=args, be=be, prompt=prompt,
        )
        # v0.6: overwrite ``prompt`` with the trigger-prepended version
        # so the Iteration's recorded prompt matches the argv emission
        # (and dry-run / history.jsonl both display the trigger).
        prompt = lora_resolution.prompt_with_triggers
        if lora_resolution.incompat_loras:
            for lora in lora_resolution.incompat_loras:
                incompat_keys.add(
                    (be.lora_compat_group, lora.ref),
                )
            for lora in lora_resolution.incompat_loras:
                incompat_details.setdefault(
                    (be.lora_compat_group, lora.ref),
                    tuple(sorted(lora.compatible_with)),
                )

        # 5. GenParams payload — Engine.run reads this. v0.8.4 M-NEW-D:
        # pre-built ``cmd`` argv no longer stored on Iteration;
        # MfluxEngine.build_cmd is invoked at dispatch time
        # (or by iteration_dryrun_display for --dry-run preview).
        gen_params = _genparams_from_iteration_inputs(
            prompt=prompt,
            negative=negative,
            width=width,
            height=height,
            params=params,
            seed=seed,
            input_path=input_path,
            output_path=output_path,
            loras=lora_resolution.effective_loras,
            merged_defaults=merged_defaults,
        )
        # v0.9.3 C3 (B-3 closure): validate the resolved GenParams.
        validate_engine_params_or_die(model, params=gen_params)
        iterations.append(Iteration(
            style_name=style_name,
            prompt=prompt,
            negative=negative,
            final_steps=params.final_steps,
            final_quantize=params.final_quantize,
            final_guidance=params.final_guidance,
            final_strength=params.final_strength,
            output_path=output_path,
            # The compat-filtered stack — incompatible LoRAs already
            # warn-and-skipped by filter_compatible_loras above. This is
            # exactly what lands on the argv (via MfluxEngine.build_cmd
            # → filter_compatible_loras), and what v=3 history records
            # for replay determinism.
            loras=lora_resolution.compatible_loras,
            # v0.7.3: per-Iteration seed. i2i (cmd_generate/cmd_batch)
            # uses one seed across all M styles of a single input.
            seed=seed,
            model=model,
            params=gen_params,
        ))

    # v0.6.x backlog python IMP-3: emit one warn per (backend_group, ref)
    # pair we haven't already warned about. The caller-provided set (if
    # any) accumulates across multiple build_iterations calls so cmd_batch
    # doesn't re-warn for every input in an N×M run.
    if incompat_keys:
        already_warned = warned_incompat_loras if warned_incompat_loras is not None else set()
        new_keys = incompat_keys - already_warned
        # Stable order: sort by (group, ref) so test assertions and user
        # output don't depend on set iteration order.
        for key in sorted(new_keys):
            group, ref = key
            compat = incompat_details.get(key, ())
            warn(
                f"LoRA {ref!r} (compat: {list(compat)}) is not compatible "
                f"with backend {group!r} — skipped"
            )
            already_warned.add(key)

    return iterations

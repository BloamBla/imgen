"""Shared helpers for ``cmd_generate`` and ``cmd_batch``.

Extracted v0.3.1 from ``commands/generate.py`` after the v0.3.0
architect review flagged the cross-module underscore-prefix import
pattern: ``batch.py`` was reaching into ``generate._helpers`` for 12
functions, which by Python convention says "private to that module"
while actually being shared. With two command modules consuming the
same pipeline pieces, a dedicated helpers module is the cleaner seam.

What lives here (alphabetical):

* :func:`build_iterations` — pre-build the M (or N×M) iteration plan
  before any subprocess work so dry-run / preflight / confirm gate
  can all reason about the full grid.
* :func:`check_prompt_style_compat` — reject incompatible
  (prompt, style) combinations upfront.
* :func:`estimate_one_seconds` — ETA helper backed by recent
  successful history entries matching backend/quant/preview.
* :func:`exit_code` — single-style passthrough vs batch 0/1/5 mapping.
* :func:`format_duration` — short human duration formatter.
* :func:`load_backend_and_token` — resolve backend dataclass + HF
  token (if needed) + mflux binary path; exits 3 on missing tool.
* :func:`open_results` — Finder/Preview launch with extension safety
  re-check; silent no-op on ``--no-open``.
* :func:`preflight_resources` — RAM / disk / battery / parallel-mflux
  gate; ``--force`` skips.
* :func:`print_batch_summary` — end-of-batch ok/fail count block.
* :func:`resolve_output_layout` — single-file ``--output`` vs run-dir
  layout (pure; mutex-with-multi-style check lives in generate.py
  as ``_check_output_style_mutex`` since batch has no ``--output``).
* :func:`resolve_styles_list` — args.style (parser-validated list) or
  fallback to merged-defaults' single name; pure.
* :func:`run_one_iteration` — one mflux invocation end-to-end:
  banner + log markers + subprocess + history append + result list.
* :func:`safe_append_history` — append-history wrapper that degrades
  unexpected exceptions to ``warn()`` instead of aborting the run loop.

What deliberately stays in ``commands/generate.py``:

* ``_confirm_batch`` — generate's 1×M confirm gate UI (batch has
  its own ``_confirm_dir_batch`` with N×M counts).
* ``_validate_input_path`` — generate-only (batch uses
  ``discover_inputs`` for its dir-of-files input).
* ``_check_output_style_mutex`` — generate-only mutex (batch has no
  ``--output FILE`` flag).
* ``cmd_generate`` — the orchestrator.

Naming convention: the moved functions drop the leading underscore.
They were "private to generate.py" by labelling; now they're a
documented shared surface used by both command modules. Functions
that genuinely stay generate-private keep the underscore.
"""
from __future__ import annotations

import datetime
import os
import subprocess
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path

from .backends import (
    Backend,
    build_mflux_cmd,
    filter_compatible_loras,
    get_backend,
)
from .checks import check_mflux, check_resources, check_venv
from .colors import C, die, err, ok, step, warn
from .config import effective_enhance, effective_output_dir
from .defaults import PREVIEW_OVERRIDES
from .enhance import (
    EnhanceResult,
    enhance_iteration_prompts,
    replace_prompt_in_cmd,
)
from .history import append_history
from .images import apply_scope
from .paths import DEFAULT_OUTPUT_DIR, SAFE_OUTPUT_EXTS, VENV_BIN
from .runs import (
    BatchContext,
    BatchLogger,
    Iteration,
    PerInputBatch,
    auto_run_dirname,
    next_available_run_dir,
)
from .styles import LoraRef, Style, StyleNotFound, get_style
from .subprocess_helpers import run_with_stderr_redaction
from .tokens import load_token

__all__ = [
    "apply_enhance_results_to_iterations",
    "apply_enhance_results_to_per_input",
    "build_iterations",
    "check_prompt_style_compat",
    "estimate_one_seconds",
    "exit_code",
    "format_duration",
    "load_backend_and_token",
    "maybe_enhance_for_command",
    "open_results",
    "preflight_resources",
    "prepend_trigger_words",
    "print_batch_summary",
    "resolve_effective_loras",
    "resolve_enhance_config",
    "resolve_output_layout",
    "resolve_styles_list",
    "run_one_iteration",
    "safe_append_history",
]


# ── v0.6: LoRA stack resolution + trigger-word prepending ─────────────


def resolve_effective_loras(
    preset,  # Style | dict — Style in prod since v0.6.2; replay shim may pass dict
    cli_lora: list | None,
    no_lora: bool,
) -> tuple[LoraRef, ...]:
    """Combine style-declared LoRAs + CLI-supplied LoRAs into the final
    tuple that flows into ``build_mflux_cmd``.

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
    * Otherwise the style's ``preset.get("loras", ())`` provides the
      base stack; ``cli_lora`` (if non-None) is APPENDED. Order in
      the final tuple = style LoRAs first, CLI LoRAs after. mflux
      applies LoRAs in argv order, so the user's CLI additions layer
      ON TOP of the style's curated stack.

    Pure: no I/O, no mutation of either input. Returns an empty
    tuple when both sources are empty / disabled.
    """
    if no_lora:
        return tuple(cli_lora) if cli_lora else ()
    style_loras = tuple(preset.get("loras", ()))
    if not cli_lora:
        return style_loras
    return style_loras + tuple(cli_lora)


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
    import re

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


# ── ETA helpers ─────────────────────────────────────────────────────────


def estimate_one_seconds(
    history_entries: list[dict],
    backend: str,
    quantize: int,
    preview: bool,
) -> int | None:
    """Average duration of recent successful generations matching params.

    Returns None when no matching successes — caller suppresses ETA display
    rather than guessing from a coarse fallback table that would be wildly
    off across M1/M2/M3/M4 hardware variance.
    """
    matching = [
        e for e in history_entries
        if e.get("status") == "success"
        and e.get("backend") == backend
        and e.get("quantize") == quantize
        and e.get("preview") == preview
        and isinstance(e.get("duration_sec"), int)
        # > 0 so a freak `duration_sec = 0` entry (cancelled-in-same-
        # second, or weird mflux exit) can't pull the average toward
        # zero. (python I4 from v0.2.3 review)
        and e["duration_sec"] > 0
    ]
    if not matching:
        return None
    recent = matching[-5:]
    return sum(e["duration_sec"] for e in recent) // len(recent)


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"~{seconds // 60} min"


# ── Prompt / style compatibility ────────────────────────────────────────


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

    Pre-v0.3.5 there was a second mutex (full-style + custom-prompt)
    that's now lifted — see :func:`build_iterations` for the
    augmentation semantics. The UX wart where a bare
    ``imgen photo.jpg --custom-prompt "..."`` invocation died because
    the default style "pixar" had a prompt is also fixed by the lift.

    Raises SystemExit(2) on the remaining incompatibility. Returns
    None on success.
    """
    if effective_custom_prompt:
        # v0.3.5: full-style + custom-prompt now augment — see
        # build_iterations. Nothing to reject here.
        return
    # No custom prompt → every listed style must have its own.
    missing_prompt = [s for s in styles_list if not get_style(s).get("prompt")]
    if missing_prompt:
        die(f"Style(s) without a prompt: {', '.join(missing_prompt)}. "
            "Pass --custom-prompt (or --prompt-file) to supply one.",
            code=2,
            hint="Param-only styles in ~/.imgen/styles.d/ need a "
                 "CLI-supplied prompt.")


# ── Output layout ───────────────────────────────────────────────────────


def resolve_output_layout(
    args,
    config_output_dir: str | None,
) -> tuple[Path | None, Path | None]:
    """Pick between single-file output and run-folder layout.

    Two mutually exclusive modes:
      * ``args.output`` (legacy --output FILE) → returns
        (explicit_path, None). Resolution + ~-expansion applied. The
        caller writes the single file to this path.
      * Otherwise the v0.2.3 folder-per-invocation layout → returns
        (None, run_dir). ``run_dir`` is computed from CLI > config >
        module-default precedence (via ``effective_output_dir``) plus a
        timestamp suffix (``auto_run_dirname``) with `_2`/`_3`
        collision handling. The directory is NOT created here — caller
        mkdir's after confirm gates so cancel doesn't orphan an empty
        dir.

    ``imgen batch`` always lands in the run-dir branch — its parser
    stanza omits ``--output FILE`` entirely. ``getattr`` accommodates
    that so the same helper composes between generate and batch.
    """
    if getattr(args, "output", None):
        explicit_output = Path(args.output).expanduser().resolve()
        return explicit_output, None
    parent = effective_output_dir(
        cli_value=getattr(args, "output_dir", None),
        config_value=config_output_dir,
        module_default=DEFAULT_OUTPUT_DIR,
    )
    run_dir = next_available_run_dir(parent, auto_run_dirname())
    return None, run_dir


# ── History append (guarded) ────────────────────────────────────────────


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


# ── v0.5: LLM prompt enhancer integration ──────────────────────────────


def resolve_enhance_config(
    *,
    cli_enable: bool | None,
    cli_model: str | None,
    cli_temperature: float | None,
    config_enhance: dict,
) -> dict:
    """Apply CLI > config > module-default precedence to the enhance
    settings and return the resolved dict (matches the existing
    ``config.effective_enhance`` contract — keys ``enabled`` /
    ``model`` / ``temperature`` / ``max_tokens`` / ``timeout_s``).

    Split out of :func:`maybe_enhance_for_command` in v0.6.4 per the
    v0.5 architect IMP #3. The wrapper used to poke ``args`` directly
    (``getattr(args, "enhance", None)`` etc.) which coupled it to the
    argparse Namespace shape — callers (commands/generate.py +
    commands/batch.py) now do the explicit two-step:

    .. code:: python

        eff = resolve_enhance_config(
            cli_enable=args.enhance,
            cli_model=args.enhance_model,
            cli_temperature=args.enhance_temperature,
            config_enhance=getattr(args, "imgen_config_enhance", {}),
        )
        results, model = maybe_enhance_for_command(
            eff_enhance=eff, backend_obj=be, iterations=iterations,
        )

    Pure delegation to :func:`config.effective_enhance` — same shape,
    different name (``resolve_enhance_config`` reads as an action; the
    older ``effective_enhance`` name reads as a getter). The architect
    naming was preferred for the call-site clarity.
    """
    return effective_enhance(
        cli_enable=cli_enable,
        config_enhance=config_enhance,
        cli_model=cli_model,
        cli_temperature=cli_temperature,
    )


def maybe_enhance_for_command(
    *,
    eff_enhance: dict,
    backend_obj: Backend,
    iterations: list[Iteration],
) -> tuple[list[EnhanceResult], str | None]:
    """Optionally run the LLM enhancer, return results aligned with
    ``iterations``.

    Returns ``(enhance_results, enhance_model)``:

    * ``enhance_results`` — list of length ``len(iterations)``. Each
      EnhanceResult carries either the enhanced prompt (was_enhanced
      True) or the original prompt + a diagnostic fallback_reason.
      When enhancement is disabled at the CLI/config level every
      result is ``fallback_reason="user_opt_out"`` (no LLM invoked).
    * ``enhance_model`` — the resolved model name when enhancement
      ran, ``None`` when disabled (so history entries don't claim
      an unused model).

    Takes ``eff_enhance`` (the pre-resolved config dict from
    :func:`resolve_enhance_config`) rather than the args Namespace —
    keeps the wrapper decoupled from argparse so future config
    sources (CRON / HTTP API / library use) can drive the enhancer
    without spelunking through an ``args`` shape. v0.6.4 split per
    v0.5 architect IMP #3.

    Tests bypass this wrapper and call the underlying orchestrator
    :func:`enhance.enhance_iteration_prompts` directly with a mocked
    LLM callable.
    """
    eff = eff_enhance

    if not eff["enabled"]:
        # Build aligned skip-results so the run loop has something to
        # splice into history. ``user_opt_out`` covers both "user passed
        # --no-enhance" and "no flag, config default is false" — both
        # are user-controlled non-activation.
        results = [
            EnhanceResult(
                final_prompt=it.prompt,
                original_prompt=it.prompt,
                was_enhanced=False,
                fallback_reason="user_opt_out",
                was_truncated=False,
                raw_llm_output=None,
            )
            for it in iterations
        ]
        return results, None

    step(f"Enhancing {len(iterations)} prompt(s) via {eff['model']} "
         f"(temp={eff['temperature']}, max_tokens={eff['max_tokens']})...")
    results = enhance_iteration_prompts(
        iteration_prompts=[it.prompt for it in iterations],
        system_prompt=backend_obj.enhance_system_prompt,
        invariants=backend_obj.enhance_invariants,
        model=eff["model"],
        temperature=eff["temperature"],
        max_tokens=eff["max_tokens"],
        timeout_s=eff["timeout_s"],
    )

    # Surface a one-line summary so the user knows what happened —
    # especially the fallback paths.
    enhanced_n = sum(1 for r in results if r.was_enhanced)
    if enhanced_n == len(results):
        ok(f"Enhanced all {enhanced_n} prompt(s).")
    elif enhanced_n == 0:
        # Distinguish the all-runner-error case from per-prompt fallbacks
        # — those signal mlx_lm load failure / timeout / crash and are
        # the "your enhancer is broken, fix it" kind of feedback.
        reasons = {r.fallback_reason for r in results}
        if reasons == {"runner_error"}:
            # !r-format the runner-error message: fallback_detail here is
            # the str(RunnerError) which may contain ANSI escape bytes
            # bubbled up from mlx_lm / huggingface_hub error tracebacks.
            # Mirrors v0.4 security-reviewer IMP-2 pattern for any
            # user-supplied / library-supplied string reaching the
            # terminal — escapes become literal \x1b instead of clearing
            # the user's screen or setting their terminal title.
            #
            # v0.6.5 moved the message from raw_llm_output to
            # fallback_detail (see EnhanceResult docstring). Producer
            # and this consumer change together; no in-flight
            # EnhanceResult shape carries the old field for this path.
            warn("Enhance runner failed; running with original prompts. "
                 f"Reason: {results[0].fallback_detail!r}")
        else:
            warn(f"No prompts enhanced. Reasons: {sorted(reasons)}")
    else:
        warn(f"Enhanced {enhanced_n}/{len(results)} prompt(s); "
             f"fallback reasons: "
             f"{sorted({r.fallback_reason for r in results if r.fallback_reason})}")

    return results, eff["model"]


def apply_enhance_results_to_iterations(
    iterations: list[Iteration],
    enhance_results: list[EnhanceResult],
) -> list[Iteration]:
    """Splice enhanced prompts back into the iteration plan.

    Iteration is frozen+slots — we build NEW instances via
    :func:`dataclasses.replace` rather than mutating. For each
    iteration whose result was successfully enhanced, the new
    iteration carries the LLM-expanded prompt AND a freshly patched
    ``cmd`` (the ``--prompt`` argv slot is overwritten via
    :func:`enhance.replace_prompt_in_cmd`). Skipped / fallback /
    invariant-violated iterations are returned unchanged — their
    final_prompt equals their original ``it.prompt`` anyway, so
    rebuilding would be a no-op.

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
            new_cmd = replace_prompt_in_cmd(it.cmd, r.final_prompt)
            out.append(_dataclass_replace(it, prompt=r.final_prompt, cmd=new_cmd))
        else:
            out.append(it)
    return out


def apply_enhance_results_to_per_input(
    per_input_iters: list[PerInputBatch],
    enhance_results: list[EnhanceResult],
) -> list[PerInputBatch]:
    """Wrapper around :func:`apply_enhance_results_to_iterations` for
    ``cmd_batch``'s per-input shape — eliminates the sliding-cursor
    block that used to live inline in batch.py (v0.5 architect IMP #2,
    extracted in v0.6.4; signature promoted from 5-tuple to
    :class:`PerInputBatch` in v0.6.5 per architect IMP-3).

    ``per_input_iters`` is the cmd_batch shape: one
    :class:`~imgen.runs.PerInputBatch` per discovered input photo,
    each carrying its M-style iteration tuple. ``enhance_results`` is
    the FLAT list returned by :func:`enhance_iteration_prompts`
    aligned to ``[it for pib in per_input_iters for it in pib.iters]``.

    Returns a new list of :class:`PerInputBatch` instances with
    enhanced prompts spliced in. Per-input group lengths preserved
    (the helper doesn't assume uniform M; any future per-style skip
    logic that produces ragged groups stays intact).

    v0.6.5 (architect IMP-3): the per-input shape was a bare
    ``tuple[Path, Path, int, int, list[Iteration]]`` through v0.6.4,
    promoted here to :class:`PerInputBatch` for named-field access at
    callers.

    Pure: no I/O. Asserts the flat-shape count matches sum of group
    lengths — misalignment would silently miswire prompts.
    """
    expected_flat = sum(len(pib.iters) for pib in per_input_iters)
    if expected_flat != len(enhance_results):
        raise ValueError(
            f"enhance-result count mismatch: per-input groups sum to "
            f"{expected_flat} iterations vs {len(enhance_results)} results"
        )
    out: list[PerInputBatch] = []
    cursor = 0
    for pib in per_input_iters:
        group_len = len(pib.iters)
        group_results = enhance_results[cursor:cursor + group_len]
        cursor += group_len
        new_iters = apply_enhance_results_to_iterations(
            list(pib.iters), group_results,
        )
        # v0.6.5 architect FL-6: ``replace`` carries every non-overridden
        # field through unchanged, so a future :class:`PerInputBatch`
        # field addition (e.g. ``original_dimensions``, draw-marker)
        # propagates automatically. Manual field-by-field rebuild would
        # silently drop new fields. Matches the
        # ``apply_enhance_results_to_iterations`` pattern at line 529.
        out.append(_dataclass_replace(pib, iters=tuple(new_iters)))
    return out


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
    cmd = it.cmd

    if is_batch:
        step(f"Generating [{idx}/{total}] {style_name} → {output_path.name}")
    else:
        step(f"Generating {style_name} → {output_path.name}")
    print(f"   {C.DIM}backend: {ctx.backend} q{it.final_quantize}  "
          f"steps: {it.final_steps}  guidance: {it.final_guidance}  "
          f"strength: {it.final_strength}  seed: {ctx.seed}{C.END}")
    print(f"   {C.DIM}size: {ctx.width}x{ctx.height}  "
          f"input: {ctx.input_path.name} → output: {output_path}{C.END}")
    print()

    started = datetime.datetime.now()
    history_entry: dict = {
        "ts": started.isoformat(timespec="seconds"),
        "input": str(ctx.input_path),
        "output": str(output_path),
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
        "seed": ctx.seed,
        "steps": it.final_steps,
        "guidance": it.final_guidance,
        "strength": it.final_strength,
        "backend": ctx.backend,
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
        returncode = run_with_stderr_redaction(
            cmd,
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


# ── Results opening / preflight / summary / exit code ──────────────────


def open_results(
    succeeded: list[tuple[str, Path, int]],
    run_dir: Path | None,
    is_batch: bool,
    no_open: bool,
) -> None:
    """Auto-open results — Finder for batch runs, Preview for single.

    Skipped entirely on --no-open, on empty success list, or when the
    `open` binary is missing (FileNotFoundError swallowed — generation
    already succeeded, no point crashing the CLI). For single-style,
    re-checks the extension against SAFE_OUTPUT_EXTS — macOS ``open``
    delegates to the registered app for the suffix, so a .sh / .command
    target would auto-execute. The whitelist is the last guard.
    """
    if no_open or not succeeded:
        return
    if is_batch and run_dir is not None:
        # Belt-and-braces: only open if it's actually a directory.
        # If the dir somehow disappeared between mkdir and now (rare),
        # don't let `open <file>` auto-launch the registered app for
        # whatever path that resolved to. (security I3 from v0.2.3 review)
        if run_dir.is_dir():
            try:
                subprocess.run(["open", str(run_dir)], check=False)
            except FileNotFoundError:
                pass
        return
    last_path = succeeded[-1][1]
    if last_path.suffix.lower() not in SAFE_OUTPUT_EXTS:
        warn(f"Skipping auto-open: unsafe extension {last_path.suffix}")
        return
    try:
        subprocess.run(["open", str(last_path)], check=False)
    except FileNotFoundError:
        pass


def preflight_resources(
    *,
    backend: str,
    heaviest_quant: int,
    force: bool,
) -> None:
    """Check RAM / disk / battery / parallel-mflux against the heaviest
    quant in the batch.

    --force skips the entire check (caller already opted into the risk
    of swap thrashing). Otherwise:
      * another mflux PID detected → die(4); parallel runs OOM
      * insufficient RAM → die(4); list specific fixes (--preview,
        --quantize 4, --force)
      * low disk → warn (model download might still fit)
      * low battery → warn (charger may be nearby)

    The two hard failures share exit code 4 (resource class) so callers
    can grep by code without parsing messages.
    """
    if force:
        return
    res = check_resources(backend, heaviest_quant)

    if res["other_mflux_pid"] is not None:
        die(f"Another mflux process is already running (PID "
            f"{res['other_mflux_pid']}). Two parallel runs will OOM and "
            "trash each other.",
            code=4,
            hint="Wait for it to finish (check with: ps -p "
                 f"{res['other_mflux_pid']}), or pass --force.")

    if not res["ram_ok"]:
        die(f"Not enough RAM: need ~{res['ram_required_gb']} GB peak "
            f"for {backend} q{heaviest_quant}, only "
            f"{res['ram_available_gb']:.1f} GB available "
            f"(of {res['ram_total_gb']:.0f} GB total).",
            code=4,
            hint=("How to fix:\n"
                  "     • Close other apps (Chrome often eats 5+ GB)\n"
                  "     • Drop quant: --quantize 4 (needs ~9 GB for flux)\n"
                  "     • Or --preview (uses --quantize 4 automatically)\n"
                  "     • Or --force (swaps to disk, very slow, may freeze)"))

    if not res["disk_ok"]:
        warn(f"Only {res['disk_free_gb']:.1f} GB disk free — risky if "
             "model needs download. Consider: imgen clean")
    if not res["battery_ok"]:
        warn(f"Battery {res['battery_pct']}% on battery — long runs may "
             "not finish. Plug in for safety.")


def print_batch_summary(
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
    total: int,
) -> None:
    """Render the end-of-batch summary block (batch runs only).

    Caller gates on `is_batch` — single-style runs keep v0.2.x's lean
    output where the per-image "Done in 3m 12s" line is the only signal.
    Always lists every failed style so the user can re-run just those,
    not the whole batch."""
    print()
    step(f"Batch summary ({total} generation{'s' if total != 1 else ''})")
    if succeeded:
        ok(f"{len(succeeded)} ok")
    if failed:
        err(f"{len(failed)} failed:")
        for sn, rc, _ in failed:
            print(f"   {C.DIM}• {sn}: exit {rc}{C.END}")


def exit_code(
    *,
    is_batch: bool,
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
) -> int:
    """Map (is_batch, succeeded, failed) → process exit code.

    Single-style preserves v0.2.x semantics: mflux's returncode passes
    through so scripts that branch on exit code keep working. Batch
    runs use distinct codes so callers can tell apart all-ok / all-
    failed / partial without parsing output:

      * all ok   → 0
      * all failed → 1
      * partial  → 5  (distinct from user-input=2, missing-tool=3,
                        resource=4 — keeps grep-by-code scripting clean)
    """
    if not is_batch:
        if failed:
            return failed[0][1]
        return 0
    if failed and not succeeded:
        return 1
    if failed:
        return 5
    return 0


# ── Iteration plan + backend resolution + styles list ───────────────────


@dataclass(frozen=True, slots=True)
class IterationParams:
    """Resolved numeric params for one iteration after CLI > preset >
    defaults precedence is applied. Returned by
    :func:`_resolve_iteration_params` so the outer loop only assembles
    the :class:`Iteration` from named, validated values.

    Intentionally promoted past the single-use threshold (currently
    only ``build_iterations`` consumes it): the named-attribute form
    beats an anonymous 4-tuple for read-clarity at the call site, and
    future ``imgen draw`` work per [[project-v063-backlog]] FL-1 will
    extend IterationParams with a ``role``-aware optional field —
    keeping the dataclass shape here means the extension lands as a
    new field, not a tuple-shape break. (v0.6.4 architect NIT-1.)
    """
    final_steps: int
    final_quantize: int
    final_guidance: float
    final_strength: float


@dataclass(frozen=True, slots=True)
class LoraResolution:
    """Resolved LoRA stack for one iteration + the prompt with trigger
    words prepended. Returned by :func:`_resolve_iteration_loras` so
    the outer ``build_iterations`` loop reads as named-field access
    instead of 4-tuple positional unpacking.

    Same shape rationale as :class:`IterationParams` (v0.6.4 architect
    NIT-3): three of the four return values are LoRA tuples differing
    only in filter stage — a positional unpack would silently miswire
    if a future contributor swapped two LoRA-tuple slots. Frozen+slots
    is consistent with the project's other config dataclasses.
    """
    effective_loras: tuple
    compatible_loras: tuple
    incompat_loras: tuple
    prompt_with_triggers: str


def _resolve_iteration_params(
    *,
    args,
    preset: Style,
    merged_defaults: dict,
) -> IterationParams:
    """Apply v0.3.x parameter precedence rules and return the resolved
    numeric quartet (steps / quantize / guidance / strength).

    Precedence (locked by tests):
      * ``steps``    : CLI > preview > merged_defaults  (preset.steps
                       intentionally NOT honoured — preview must win
                       when the user picks it for speed)
      * ``quantize`` : CLI > preview > merged_defaults  (same reasoning)
      * ``guidance`` : CLI > preset  > merged_defaults
      * ``strength`` : CLI > preset  > merged_defaults

    Extracted v0.6.4 from ``build_iterations`` per the v0.6.2 architect
    IMP-2 split. Pure: no I/O, no mutation.
    """
    if args.steps is not None:
        final_steps = args.steps
    elif args.preview:
        final_steps = PREVIEW_OVERRIDES["steps"]
    else:
        final_steps = merged_defaults["steps"]

    if args.quantize is not None:
        final_quantize = args.quantize
    elif args.preview:
        final_quantize = PREVIEW_OVERRIDES["quantize"]
    else:
        final_quantize = merged_defaults["quantize"]

    if args.guidance is not None:
        final_guidance = args.guidance
    elif "guidance" in preset:
        final_guidance = preset["guidance"]
    else:
        final_guidance = merged_defaults["guidance"]

    if args.strength is not None:
        final_strength = args.strength
    elif "strength" in preset:
        final_strength = preset["strength"]
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
    ``--scope`` is photo-input-specific (i2i-only) and the future
    ``imgen draw`` subparser will omit it. Pre-emptive defence so this
    helper drops cleanly into the t2i path without a ``--scope=None``
    workaround on the draw parser.
    """
    scope = getattr(args, "scope", None)
    scene_suffix = preset.get("scene_suffix")
    preset_prompt = preset.get("prompt")
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
    be,
    prompt: str,
) -> LoraResolution:
    """Resolve the LoRA stack for one iteration + emit the prompt with
    trigger words prepended.

    Returns a :class:`LoraResolution`:

      * ``effective_loras`` — style + CLI stack post-``--no-lora``
        opt-out; what flows into ``build_mflux_cmd`` for argv emission.
      * ``compatible_loras`` — subset that matches the backend's
        ``lora_compat_group``; what lands on :class:`Iteration.loras`
        + drives trigger-word prepending.
      * ``incompat_loras`` — subset that does NOT match. Caller
        accumulates these into the cross-iteration dedup set.
      * ``prompt_with_triggers`` — input ``prompt`` with each
        compatible LoRA's trigger word prepended if missing.

    Extracted v0.6.4 from ``build_iterations`` per the v0.6.2 architect
    IMP-2 split (v0.6.4 architect NIT-3 promoted the 4-tuple return
    into a named LoraResolution dataclass). Pure: no I/O, no mutation
    of inputs. The warn for incompat LoRAs is emitted once-per-pair
    by the orchestrator after its loop completes (v0.6.x IMP-3 dedup).
    """
    cli_lora_list = getattr(args, "lora", None)
    no_lora = bool(getattr(args, "no_lora", False))
    effective_loras = resolve_effective_loras(preset, cli_lora_list, no_lora)
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


def build_iterations(
    *,
    styles_list: list[str],
    args,
    effective_custom_prompt: str | None,
    merged_defaults: dict,
    be,
    binary: Path,
    input_path: Path,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
    warned_incompat_loras: set[tuple[str, str]] | None = None,
) -> list[Iteration]:
    """Resolve per-style params + build the mflux command for each style.

    The whole batch is pre-built before any subprocess work so:
      * --dry-run can print every cmd that would be executed
      * resource preflight runs against ``max(it.final_quantize)`` —
        no surprise crash on the 3rd image if its quant is heavier
      * confirm gate can show the full list

    Parameter precedence (locked by tests):
      * ``steps``    : CLI > preview > merged_defaults  (preset.steps
                       intentionally NOT honoured — preview must win
                       when the user picks it for speed)
      * ``quantize`` : CLI > preview > merged_defaults  (same reasoning)
      * ``guidance`` : CLI > preset  > merged_defaults
      * ``strength`` : CLI > preset  > merged_defaults
      * ``prompt``   : custom_prompt verbatim (if set), else
                       preset["prompt"] with optional scope substitution
      * ``negative`` : preset.get("negative", "")

    ``output_path`` per iteration:
      * if ``explicit_output`` is set (legacy --output FILE) → that path
      * else ``run_dir / "<input.stem>-<style>.png"``

    Returns ``list[Iteration]`` (frozen) — caller may not mutate entries.
    """
    iterations: list[Iteration] = []
    # v0.6.x backlog python IMP-3: collect incompatible (backend_group,
    # lora.ref) pairs across the iteration loop, then emit ONE warn per
    # unique pair at end of function. ``incompat_details`` carries the
    # full compatible_with tuple for the first occurrence so the warn
    # message can still surface where the LoRA WOULD have fitted.
    # Optionally accepts a caller-provided ``warned_incompat_loras`` set
    # to dedup across multiple build_iterations calls (cmd_batch shares
    # one across N inputs → 1 warn per unique LoRA total instead of N).
    #
    # v0.6.4 python NIT-3: the dedup key is ``(backend_group, lora.ref)``
    # — intentionally NOT including ``weight`` because the warn is about
    # COMPAT (backend can't load the LoRA), not about weight value. Two
    # LoraRef instances with the same ref but different weights collapse
    # to the same warn, which is correct semantically. A future caller
    # that constructs the SAME ref with TWO different ``compatible_with``
    # tuples would have its second tuple silently swallowed by the
    # ``setdefault`` below (only the first occurrence's compat list
    # surfaces in the warn). That edge case is unreachable today via
    # CLI / TOML loader paths but worth a comment for the future.
    incompat_keys: set[tuple[str, str]] = set()
    incompat_details: dict[tuple[str, str], tuple[str, ...]] = {}
    # `args.style` is None when the parser fell back to merged_defaults
    # for the default style (no explicit --style passed). Used below to
    # gate augmentation: if user didn't explicitly pick a style, their
    # `--custom-prompt` should drive the prompt content entirely rather
    # than augment the default style's prompt — otherwise a bare
    # `imgen photo.jpg --custom-prompt "make sepia"` would produce
    # "Pixar 3D character + sepia" which is nonsense for that invocation
    # shape. (v0.3.5 UX wart fix.)
    style_was_explicit = bool(getattr(args, "style", None))

    for style_name in styles_list:
        preset = get_style(style_name)

        # 1. Prompt construction (3-way dispatch: augmentation /
        # custom-only / preset-only). Scope substitution baked in.
        prompt = _resolve_iteration_prompt(
            preset=preset,
            args=args,
            effective_custom_prompt=effective_custom_prompt,
            style_was_explicit=style_was_explicit,
        )

        negative = preset.get("negative", "")

        # 2. Numeric parameter precedence (CLI > preview > preset >
        # defaults; rules vary per field — locked by tests).
        params = _resolve_iteration_params(
            args=args, preset=preset, merged_defaults=merged_defaults,
        )

        # 3. Output path: explicit --output FILE (legacy single-file
        # path) wins; otherwise <run_dir>/<input.stem>-<style>.png.
        if explicit_output is not None:
            output_path = explicit_output
        else:
            output_path = run_dir / f"{input_path.stem}-{style_name}.png"

        # 4. LoRA stack: style + CLI minus --no-lora opt-out; filter for
        # backend compat; prepend trigger words for compatible LoRAs.
        # Incompatibles roll up into the dedup accumulators so the
        # post-loop warn block emits once per unique (group, ref).
        lora_resolution = _resolve_iteration_loras(
            preset=preset, args=args, be=be, prompt=prompt,
        )
        prompt = lora_resolution.prompt_with_triggers
        if lora_resolution.incompat_loras:
            incompat_keys.update(
                (be.lora_compat_group, lora.ref)
                for lora in lora_resolution.incompat_loras
            )
            for lora in lora_resolution.incompat_loras:
                incompat_details.setdefault(
                    (be.lora_compat_group, lora.ref),
                    tuple(sorted(lora.compatible_with)),
                )

        # 5. Argv assembly for this iteration.
        cmd = build_mflux_cmd(
            binary=binary,
            backend=be,
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            negative=negative,
            quantize=params.final_quantize,
            steps=params.final_steps,
            guidance=params.final_guidance,
            strength=params.final_strength,
            seed=seed,
            width=width,
            height=height,
            mlx_cache_gb=merged_defaults["mlx_cache_gb"],
            battery_stop=merged_defaults["battery_stop"],
            loras=lora_resolution.effective_loras,
        )

        iterations.append(Iteration(
            style_name=style_name,
            prompt=prompt,
            negative=negative,
            final_steps=params.final_steps,
            final_quantize=params.final_quantize,
            final_guidance=params.final_guidance,
            final_strength=params.final_strength,
            output_path=output_path,
            cmd=cmd,
            # The compat-filtered stack — incompatible LoRAs already
            # warn-and-skipped by filter_compatible_loras above. This is
            # exactly what landed on the argv, and what v=3 history
            # records for replay determinism.
            loras=lora_resolution.compatible_loras,
        ))

    # v0.6.x backlog python IMP-3: emit one warn per (backend_group, ref)
    # pair we haven't already warned about. The caller-provided set (if
    # any) accumulates across multiple build_iterations calls so cmd_batch
    # doesn't re-warn for every input in an N×M run.
    if incompat_keys:
        from .colors import warn
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
        already_warned.update(new_keys)
        # If the caller didn't provide a set, the `already_warned` we
        # built locally is discarded — fine, single-call dedup achieved.

    return iterations


def load_backend_and_token(
    args,
) -> tuple[str, Backend, str | None, Path, tuple[str, str] | None]:
    """Resolve backend metadata, HF token, binary path, and custom secret.

    Returns a 5-tuple:
    ``(backend_name, backend_dataclass, token_or_none, binary_path,
    backend_secret_or_none)``. The fifth slot is for v0.4 custom
    backends: a ``(env_var_name, value)`` pair that the subprocess
    env builder will inject under the declared name. None for
    built-ins and for custom backends whose ``[secret]`` section is
    absent.

    Exits with code 3 (missing-tool class) on:
      * gated built-in backend without an HF token (FLUX path)
      * custom backend declaring ``secret_env_var`` with
        ``required=True`` and the env var unset in the parent shell
      * venv / mflux not installed
      * the per-backend binary not present (on PATH for bare names,
        or absent at the declared absolute path)

    The HF token is loaded lazily — only when ``needs_token`` is True
    (FLUX). Open backends (qwen) and custom backends never touch
    ~/.imgen/hf_token.

    Binary resolution branches on shape:
      * Bare name (no '/') → ``VENV_BIN / be.binary``. Built-ins and
        user backends installed alongside mflux land here.
      * Absolute path (starts with '/') → used as-is. Lets a user
        point at a fork or experimental binary outside VENV_BIN.
      Schema validator enforces these two shapes; we trust that here.
    """
    backend = args.backend
    be = get_backend(backend)

    # ── HF token (FLUX-specific legacy path) ─────────────────────
    token: str | None = None
    if be.needs_token:
        token = load_token()
        if not token:
            die("FLUX backend requires HuggingFace token",
                code=3,
                hint="Run: imgen setup   (or use --backend qwen)")

    # ── Custom-backend secret (v0.4) ─────────────────────────────
    backend_secret: tuple[str, str] | None = None
    if be.secret_env_var is not None:
        value = os.environ.get(be.secret_env_var)
        # Falsy check (not `is not None`): an env var explicitly set to
        # empty string (`export MYBACK_API_KEY=`) is treated as missing.
        # An empty token is useless — forwarding it would produce a
        # confusing auth failure from the backend's binary. Same
        # contract as load_token() for the FLUX path. Locked by
        # test_load_custom_backend_dies_when_secret_env_var_set_to_empty
        # (v0.4 python-reviewer IMP-2.)
        if value:
            backend_secret = (be.secret_env_var, value)
        elif be.secret_required:
            die(
                f"Backend '{backend}' requires env var "
                f"{be.secret_env_var!r} to be set, but it's missing "
                "from the environment",
                code=3,
                hint=f"export {be.secret_env_var}=... in your shell rc "
                     "(or set secret.required=false in the backend TOML)",
            )
        # else: required=False — silent skip, subprocess inherits no
        # secret, backend's binary will handle its own auth failure.

    # ── venv + mflux sanity ──────────────────────────────────────
    if not check_venv() or not check_mflux():
        die("mflux not installed",
            code=3,
            hint="Run: imgen setup")

    # ── Binary path resolution ───────────────────────────────────
    if be.binary.startswith("/"):
        # Absolute path — validator already confirmed it exists at
        # schema time, but re-check here in case the file was removed
        # between TOML load and command execution.
        binary = Path(be.binary)
    else:
        # Bare name — resolve against VENV_BIN (mflux convention).
        binary = VENV_BIN / be.binary
    if not binary.is_file():
        # is_file() (not exists()) — a directory at the path would
        # crash subprocess.Popen with IsADirectoryError; reject earlier
        # with the imgen-flavoured error. (v0.4 python-reviewer IMP-1.)
        die(f"Backend binary not found (or not a regular file): {binary}",
            code=3,
            hint="Run: imgen upgrade")

    return backend, be, token, binary, backend_secret


def resolve_styles_list(args, merged_defaults: dict) -> list[str]:
    """Resolve ``args.style`` into a list of preset names.

    ``args.style`` is either ``None`` (not passed) or a pre-validated,
    de-duped list (parser already rejected unknown names). When unset,
    fall back to the config-merged default style and verify it exists —
    config.toml may point at a preset the user later removed from
    ``styles.d/``.

    **Pure**: this returns the resolved list and nothing else. The
    ``--output FILE`` + multi-style mutex check lives in
    ``commands/generate._check_output_style_mutex`` since
    ``imgen batch`` has no ``--output`` flag and the check would be a
    silent no-op there. Pre-v0.3.1 the mutex check was inline here with
    a ``getattr(args, "output", None)`` guard — that worked but was
    surprising for batch readers; the split makes the generate-only
    nature explicit. (v0.3.0 architect NIT-4 / NIT-6.)
    """
    if args.style:
        return list(args.style)
    default_name = merged_defaults["style"]
    try:
        get_style(default_name)
    except StyleNotFound:
        # Narrowed from `except KeyError` in v0.3.6 — StyleNotFound is
        # the only thing get_style can raise, and the narrower catch
        # lets a future generic `except KeyError:` elsewhere flag a
        # genuine bug instead of silently absorbing this path too.
        # (architect NIT-2 from v0.3.6 review.)
        die(f"Default style '{default_name}' not found",
            code=2,
            hint="Check ~/.imgen/config.toml [defaults] style, "
                 "or run: imgen --list-styles")
    return [default_name]

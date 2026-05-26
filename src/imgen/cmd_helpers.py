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
  **v0.8.3 M-NEW-B**: moved to :mod:`imgen.engine_dispatch`,
  re-exported here for back-compat.
* :func:`safe_append_history` — append-history wrapper that degrades
  unexpected exceptions to ``warn()`` instead of aborting the run loop.
  **v0.8.3 M-NEW-B**: moved to :mod:`imgen.engine_dispatch`,
  re-exported.

v0.8.3 M-NEW-B extraction: ``engine_dispatch`` owns the path from a
built :class:`Iteration` through the v0.8 Engine layer
(:class:`MfluxEngine` / :class:`DiffusersMpsEngine`) to a finished
generation subprocess. Moved out of this module to bring it under
the 800-line ceiling (still over: 1913 LoC at extraction time, down
from 2426; further build_iteration extraction is M-NEW-E for a
later tag). Functions moved + re-exported: ``_engine_for_model``,
``_genparams_from_iteration_inputs``,
``apply_enhance_results_to_iterations``,
``apply_enhance_results_to_groups``,
``emit_gated_repo_hint_if_failed``, ``run_one_iteration``,
``safe_append_history``, ``validate_engine_params_or_die``.

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
import re
import subprocess
import unicodedata
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path

from .backends import (
    Backend,
    build_mflux_cmd,
    filter_compatible_loras,
    get_backend,
)
from .checks import check_mflux, check_resources, check_venv
from .colors import C, die, err, info, ok, step, warn
from .config import effective_enhance, effective_output_dir
from .defaults import PREVIEW_OVERRIDES
from .enhance import (
    EnhanceResult,
    enhance_iteration_prompts,
)
# v0.8.3 M-NEW-B: engine-dispatch path (run_one_iteration, _engine_for_model,
# validate_engine_params_or_die, _genparams_from_iteration_inputs,
# apply_enhance_results_to_*, safe_append_history, emit_gated_repo_hint_if_failed)
# extracted to engine_dispatch. Re-exported below for back-compat with
# the ~15 test modules + production cmd_* importers that read these
# names from cmd_helpers.
from .engine_dispatch import (
    _engine_for_model,
    _genparams_from_iteration_inputs,
    apply_enhance_results_to_groups,
    apply_enhance_results_to_iterations,
    emit_gated_repo_hint_if_failed,
    run_one_iteration,
    safe_append_history,
    validate_engine_params_or_die,
)
from .history import entry_model_name
from .images import apply_scope
from .paths import DEFAULT_OUTPUT_DIR, SAFE_OUTPUT_EXTS, VENV_BIN
from .runs import (
    Iteration,
    auto_run_dirname,
    next_available_path,
    next_available_run_dir,
)
from .styles import LoraRef, Style, StyleNotFound, get_style
from .tokens import load_token

__all__ = [
    "apply_enhance_results_to_groups",
    "apply_enhance_results_to_iterations",
    "build_bare_i2i_iteration",
    "build_draw_iteration",
    "build_draw_iterations",
    "build_iterations",
    "build_refine_iteration",
    "emit_gated_repo_hint_if_failed",
    "check_prompt_style_compat",
    "require_style_or_prompt",
    "estimate_one_seconds",
    "exit_code",
    "format_duration",
    "megapixels_of",
    "load_backend_and_token",
    "maybe_enhance_for_command",
    "maybe_enhance_prompts",
    "open_results",
    "preflight_resources",
    "prepend_trigger_words",
    "print_batch_summary",
    "prompt_slug",
    "resolve_effective_loras",
    "resolve_enhance_config",
    "resolve_output_layout",
    "resolve_styles_list",
    "run_one_iteration",
    "safe_append_history",
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

    v0.8.0 commit 9 (§R.3 HIGH-1 fix): the matcher compares ``backend``
    (the resolver-translated v0.8 canonical name, e.g. ``"flux-kontext"``)
    against ``entry_model_name(e)`` — which handles BOTH the v=4 ``model``
    key AND the v=3 ``backend`` key fallback AND the v0.7→v0.8 rename
    map. Pre-fix the matcher compared ``e.get("backend") == backend``
    directly; for any user upgrading from v0.7 (history.jsonl full of
    v=3 ``"backend":"flux"`` entries), the v0.8 caller value
    ``"flux-kontext"`` never matched, so ETA went cold until 5 new
    post-upgrade entries accumulated. Pure UX regression, lock-in test
    in tests/test_v080_history_migration.py.
    """
    matching = [
        e for e in history_entries
        if e.get("status") == "success"
        and entry_model_name(e) == backend
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
    missing_prompt = [s for s in styles_list if not get_style(s).prompt]
    if missing_prompt:
        die(f"Style(s) without a prompt: {', '.join(missing_prompt)}. "
            "Pass --custom-prompt (or --prompt-file) to supply one.",
            code=2,
            hint="Param-only styles in ~/.imgen/styles.d/ need a "
                 "CLI-supplied prompt.")


def megapixels_of(width: int, height: int) -> float:
    """Convert pixel dimensions to megapixels (width × height / 1e6).

    v0.7.15 (architect Q6 advisory from v0.7.14 review): extracted from
    4 copy-pasted call sites (cmd_generate, cmd_batch, cmd_refine,
    cmd_draw) that each computed ``(w * h) / 1_000_000`` independently.
    Single tested helper eliminates the copy-paste-bug surface (e.g.
    accidental ``1_000`` instead of ``1_000_000`` typo would silently
    over-block by 1000×).

    The 1 MP canonical baseline at 1024² is exactly
    ``1024 * 1024 / 1_000_000 = 1.048576``, not 1.0 — keep the float
    return value so :func:`checks.ram_required_gb` sees the precise
    activation budget rather than a rounded integer.

    Pure: no I/O, no allocation beyond the float result.
    """
    return (width * height) / 1_000_000


def require_style_or_prompt(
    styles_list: list[str],
    effective_custom_prompt: str | None,
) -> None:
    """v0.7.13 (gap 8 behaviour pivot, architect S1 helper extraction):
    enforce the new contract that ``imgen generate`` / ``imgen batch``
    require EITHER ``--style NAME`` (preset mode) OR ``--custom-prompt
    TEXT`` / ``--prompt-file PATH`` (bare mode). Neither → die code 2
    with actionable hint.

    Pre-v0.7.13 bare ``imgen photo.jpg`` silently fell back to the
    configured default style (usually "pixar"), which leaked the
    preset's ``negative_prompt`` field into argv (the flux2-klein-edit-
    9b crash that gap 7 closed at the backend side, plus general
    "preset surprise" UX bugs). Now: explicit opt-in. The fallback
    convenience is gone — explicit is better than implicit.

    Single source of truth for the user-facing wording: both
    cmd_generate (commands/generate.py) and cmd_batch
    (commands/batch.py) call this so the migration message stays
    consistent. Pattern matches v0.3.1's ``_check_output_style_mutex``
    extraction (same architect IMP).

    Raises SystemExit(2) on the invalid combination. Returns None on
    success (caller routes through build_iterations OR
    build_bare_i2i_iteration based on whether styles_list is truthy).
    """
    if not styles_list and effective_custom_prompt is None:
        die(
            "specify --style NAME (preset mode) or --custom-prompt "
            "TEXT / --prompt-file PATH (bare mode, no preset baggage). "
            "Pre-v0.7.13 imgen fell back to the default style — see "
            "release notes for the migration.",
            code=2,
            hint="Run `imgen --list-styles` to see available presets, "
                 "or pass --custom-prompt to use a raw prompt without "
                 "any style preset applied.",
        )


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


def maybe_enhance_prompts(
    *,
    eff_enhance: dict,
    backend_obj: Backend,
    prompts: list[str],
) -> tuple[list[EnhanceResult], str | None]:
    """v0.7.3: lower-level helper underlying
    :func:`maybe_enhance_for_command`. Operates on raw prompt strings
    instead of :class:`Iteration` objects so cmd_draw's
    ``--num-iterations`` path can enhance ONE unique prompt and then
    broadcast the result to N seed-variant iterations, instead of
    paying N LLM calls for N identical prompts.

    Same return contract as :func:`maybe_enhance_for_command`:
    ``(enhance_results, enhance_model)`` aligned with ``prompts``.

    Pure-ish: prints a step / ok / warn line to stdout (same UX as
    the iteration-flavoured wrapper). No subprocess unless the LLM
    actually runs.
    """
    eff = eff_enhance

    if not eff["enabled"]:
        results = [
            EnhanceResult(
                final_prompt=p,
                original_prompt=p,
                was_enhanced=False,
                fallback_reason="user_opt_out",
                was_truncated=False,
                raw_llm_output=None,
            )
            for p in prompts
        ]
        return results, None

    step(f"Enhancing {len(prompts)} prompt(s) via {eff['model']} "
         f"(temp={eff['temperature']}, max_tokens={eff['max_tokens']})...")
    results = enhance_iteration_prompts(
        iteration_prompts=prompts,
        system_prompt=backend_obj.enhance_system_prompt,
        invariants=backend_obj.enhance_invariants,
        model=eff["model"],
        temperature=eff["temperature"],
        max_tokens=eff["max_tokens"],
        timeout_s=eff["timeout_s"],
    )
    return _summarise_enhance_results_and_pack(results, eff)


def _summarise_enhance_results_and_pack(
    results: list[EnhanceResult],
    eff: dict,
) -> tuple[list[EnhanceResult], str | None]:
    """Post-LLM-call summary path: prints the one-line outcome and
    returns ``(results, eff["model"])``. Reached ONLY on the enabled
    path — the disabled path (``eff["enabled"]=False``) in
    ``maybe_enhance_prompts`` short-circuits with ``(skip_results,
    None)`` before reaching this helper. Extracted v0.7.3 so both
    public entry points share the identical surface UX.

    Returns the model name (not None) because being here means the
    LLM ran; history rows downstream record ``enhance_model`` only
    when the LLM was actually invoked. The disabled path's caller-
    side ``None`` return for ``enhance_model`` is its own contract.
    """
    enhanced_n = sum(1 for r in results if r.was_enhanced)
    if enhanced_n == len(results):
        ok(f"Enhanced all {enhanced_n} prompt(s).")
    elif enhanced_n == 0:
        reasons = {r.fallback_reason for r in results}
        if reasons == {"runner_error"}:
            warn("Enhance runner failed; running with original prompts. "
                 f"Reason: {results[0].fallback_detail!r}")
        else:
            warn(f"No prompts enhanced. Reasons: {sorted(reasons)}")
    else:
        warn(f"Enhanced {enhanced_n}/{len(results)} prompt(s); "
             f"fallback reasons: "
             f"{sorted({r.fallback_reason for r in results if r.fallback_reason})}")
    return results, eff["model"]


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

    v0.7.3: delegates to :func:`maybe_enhance_prompts` so the two
    cmd-layer entry points (i2i iterations vs t2i raw prompts) share
    the same UX surface + LLM-call discipline.

    Tests bypass this wrapper and call the underlying orchestrator
    :func:`enhance.enhance_iteration_prompts` directly with a mocked
    LLM callable.
    """
    return maybe_enhance_prompts(
        eff_enhance=eff_enhance,
        backend_obj=backend_obj,
        prompts=[it.prompt for it in iterations],
    )


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
    model: str,
    heaviest_quant: int,
    force: bool,
    max_megapixels: float = 1.0,
) -> None:
    """Check RAM / disk / battery / parallel-mflux against the heaviest
    quant + largest output resolution in the batch.

    --force skips the entire check (caller already opted into the risk
    of swap thrashing). Otherwise:
      * another mflux PID detected → die(4); parallel runs OOM
      * insufficient RAM → die(4); list specific fixes (--preview,
        --quantize 4, --force)
      * low disk → warn (model download might still fit)
      * low battery → warn (charger may be nearby)

    The two hard failures share exit code 4 (resource class) so callers
    can grep by code without parsing messages.

    v0.7.14 (gap 6): ``max_megapixels`` argument added — caller computes
    ``max(it.width * it.height for it in iterations) / 1_000_000`` and
    passes it so RAM estimate scales with actual output resolution
    instead of the worst-case 2K² that pre-v0.7.14 baked into the
    table. Default 1.0 preserves pre-v0.7.14 behaviour for callers
    that haven't been updated yet (none in this codebase, but
    documented for forward-compat).
    """
    if force:
        return
    res = check_resources(model, heaviest_quant, max_megapixels)

    if res["other_mflux_pid"] is not None:
        die(f"Another mflux process is already running (PID "
            f"{res['other_mflux_pid']}). Two parallel runs will OOM and "
            "trash each other.",
            code=4,
            hint="Wait for it to finish (check with: ps -p "
                 f"{res['other_mflux_pid']}), or pass --force.")

    if not res["ram_ok"]:
        # v0.7.14 python NIT closure: format ram_required_gb to one
        # decimal — pre-v0.7.14 the value was a dict-int; now it's a
        # float from `ram_required_gb()` and the default __str__ would
        # surface "14.239999999999999 GB" garbage in user-facing
        # output. Matches the existing :.1f formatting on the
        # available/total lines for visual symmetry.
        die(f"Not enough RAM: need ~{res['ram_required_gb']:.1f} GB peak "
            f"for {model} q{heaviest_quant}, only "
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


# ── v0.8.0 commit 7 — Engine.validate wire-up ─────────────────────────


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
    from .backends import get_backend, model_from_backend
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
    model=None,
) -> IterationParams:
    """Apply v0.3.x + v0.8.0 commit 7 parameter precedence rules and
    return the resolved numeric quartet (steps / quantize / guidance /
    strength).

    Precedence (locked by tests):
      * ``steps``    : CLI > preview > model.default_steps > merged_defaults
                       (preset.steps intentionally NOT honoured —
                       preview must win when the user picks it for speed)
      * ``quantize`` : CLI > preview > merged_defaults  (same reasoning)
      * ``guidance`` : CLI > preset  > model.default_guidance > merged_defaults
      * ``strength`` : CLI > preset  > merged_defaults

    v0.8.0 commit 7 (§M) added the ``model`` parameter as an OPTIONAL
    layer in the steps/guidance precedence chain. When non-None (i.e.
    a built-in Model from BUILTIN_MODELS), ``model.default_steps`` and
    ``model.default_guidance`` slot between preview/preset and
    merged_defaults. When None (user-TOML lookup or test fixture not
    going through the Model layer), behavior reduces to the v0.7
    chain — back-compat with the ~30 existing test fixtures.

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
    # backend (Backend.supports_strength=False gates argv emission in
    # build_mflux_cmd). Same getattr pattern as v0.6.5's args.scope
    # FL-3 defence.
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
    ``--scope`` is photo-input-specific (i2i-only) and the future
    ``imgen draw`` subparser will omit it. Pre-emptive defence so this
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


def prompt_slug(
    prompt: str,
    *,
    max_words: int = 6,
    max_len: int = 60,
) -> str:
    """Derive a filesystem-safe slug from a (possibly post-enhance) prompt.

    v0.7.0 (architect §D): output naming for `imgen draw`. No input
    photo stem to anchor on, so the filename comes from the prompt
    itself. Six words is enough to be human-readable in a gallery view
    without overrunning macOS filename limits.

    Pipeline:
      1. Take first ``max_words`` whitespace-tokens.
      2. NFKD-normalize + strip non-ASCII (CJK → empty after this step).
      3. Lowercase + collapse non-alphanumeric runs to '-'.
      4. Strip leading/trailing '-'.
      5. Cap at ``max_len`` chars (well under macOS 255-byte limit).
      6. If empty after all of the above (e.g. emoji-only prompt),
         fall back to "draw".

    Pure: no I/O, no filesystem touch. Collision-handling
    (``-2``/``-3`` suffix when ``<slug>.png`` already exists) lives at
    the output-path resolution site, not here.
    """
    tokens = prompt.split()[:max_words]
    text = " ".join(tokens)
    # Unicode NFKD + strip combining marks; ASCII-only survives.
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    # Collapse anything-not-alphanumeric to a single '-'.
    slugged = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if len(slugged) > max_len:
        slugged = slugged[:max_len].rstrip("-")
    return slugged or "draw"


def _draw_output_path_for_index(
    *,
    run_dir: Path,
    slug: str,
    idx: int,
    num_iterations: int,
) -> Path:
    """v0.7.3: pick the per-iteration output path inside ``run_dir``.

    * N=1: ``<slug>.png`` (preserves v0.7.0 single-shot naming).
    * N>=2: ``<slug>-<idx>.png`` with ``idx`` 1-based — explicit index
      reads "1 of 5" naturally vs the next_available_path collision
      style (which would put the first one bare and start numbering at
      ``-2``).

    Collision still possible if the same run_dir already has the
    target name (rare but possible when --output-dir points at a
    reused tree). Fall back to ``next_available_path`` to suffix-
    insert ``-2``/``-3`` AFTER the explicit ``-<idx>`` part.

    Pure: probes the filesystem read-only.
    """
    if num_iterations == 1:
        return next_available_path(run_dir, slug, suffix=".png")
    indexed = f"{slug}-{idx}"
    return next_available_path(run_dir, indexed, suffix=".png")


def _assemble_iteration_no_style(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    binary: Path,
    input_path: Path | None,
    output_path: Path,
    width: int,
    height: int,
    seed: int,
    style_name: str,
    negative: str = "",
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
    # v0.8.0 commit 7 (§M): die on per-Model param violations
    # (quantize ∉ supported_quants, guidance out of [min, max]).
    # Replaces pre-commit-7 hardcoded special-cases in cmd_refine
    # (refine.py:238 flux2-klein-edit-9b guidance pin) and any
    # future per-binary cmd_* edits.
    validate_engine_params_or_die(
        model,
        quantize=params.final_quantize,
        guidance=params.final_guidance,
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
    cmd = build_mflux_cmd(
        binary=binary,
        model=be,
        input_path=input_path,
        output_path=output_path,
        prompt=lora_resolution.prompt_with_triggers,
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
    # v0.8.2 M-1A: alongside the pre-built mflux ``cmd``, attach the
    # resolved Model + GenParams so the future Engine.run dispatch in
    # ``run_one_iteration`` can route through ``engine.run(it.model,
    # it.params, ...)``. Both fields are None for user-TOML names not
    # in BUILTIN_MODELS today — v0.8.1 widened ``_model_for_validate``
    # to also resolve user TOMLs, so ``model`` is non-None for
    # every recognised --model; the None case remains a defensive
    # default for any callable build_* helper invoked with an
    # unrecognised name (the resolver dies upstream in production).
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
    )
    return Iteration(
        style_name=style_name,
        prompt=lora_resolution.prompt_with_triggers,
        negative=negative,
        final_steps=params.final_steps,
        final_quantize=params.final_quantize,
        final_guidance=params.final_guidance,
        final_strength=params.final_strength,
        output_path=output_path,
        cmd=cmd,
        loras=lora_resolution.compatible_loras,
        seed=seed,
        model=model,
        params=gen_params,
    )


def build_draw_iterations(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    binary: Path,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    base_seed: int,
    num_iterations: int = 1,
) -> list[Iteration]:
    """Build N :class:`Iteration` objects for `imgen draw` (t2i).

    v0.7.0 shipped this as singular ``build_draw_iteration``; v0.7.3
    promoted to plural to unlock ``--num-iterations N`` (explore-mode
    randomness ladder). Same workflow logic, looped N times with a
    deterministic seed ladder:

      * ``base_seed=X`` + ``num_iterations=5`` → seeds
        ``[X, X+1, X+2, X+3, X+4]``. Reproducible — re-running with
        the same ``--seed`` reproduces the same N images bit-for-bit.
      * If caller passed a random ``base_seed``, the ladder still
        applies but the run isn't reproducible without recording
        ``base_seed`` (cmd_draw records it in history.jsonl for replay).

    The N argv invocations share everything except ``--seed`` and
    ``--output`` — so the enhancer runs ONCE on the prompt at the
    cmd_draw layer (enhanced text is identical for all N seeds), then
    the same enhanced prompt threads through every Iteration here.

    Output naming: see :func:`_draw_output_path_for_index`. N=1 keeps
    ``<slug>.png``; N>=2 emits ``<slug>-1.png`` … ``<slug>-N.png``.

    ``explicit_output`` (``--output PATH``) is mutex with N>=2 — the
    parser layer rejects that combination because --output FILE
    can't fan out to N files. This helper still HONORS it when N=1
    (single-shot --output path) for backward compat with v0.7.0.

    Pure: no subprocess, no I/O beyond the ``next_available_path``
    probes. Caller (cmd_draw) does dry-run / preflight / confirm /
    run_one_iteration over the returned list.
    """
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
    if explicit_output is not None and num_iterations > 1:
        raise ValueError(
            "explicit_output is mutex with num_iterations > 1 "
            "(single --output FILE can't fan out to N files)"
        )

    slug = prompt_slug(prompt)
    iterations: list[Iteration] = []
    for i in range(num_iterations):
        # Seed ladder. Wrap at 2**32 to stay within mflux's seed range
        # (the parser guards 0..2**32-1; base_seed+(N-1) might overflow
        # if base_seed was near the cap. Modulo keeps every iteration
        # inside the valid range without crashing the ladder.)
        iter_seed = (base_seed + i) % (2**32)

        # Output path: explicit --output PATH wins (N=1 only, asserted
        # above); otherwise prompt-slug inside the run-dir.
        if explicit_output is not None:
            output_path = explicit_output
        else:
            if run_dir is None:
                raise ValueError(
                    "build_draw_iterations: either explicit_output or "
                    "run_dir must be provided"
                )
            output_path = _draw_output_path_for_index(
                run_dir=run_dir,
                slug=slug,
                idx=i + 1,
                num_iterations=num_iterations,
            )

        # v0.7.8: shared core with build_refine_iteration. Naked
        # iteration (empty Style preset, no incompat accumulator).
        # v0.7.11 (gap 1): forward --negative-prompt CLI value via the
        # `negative` parameter. `getattr(..., None)` keeps the helper
        # safe against older callers / Namespaces without the field
        # (mirrors the v0.6.5 args.scope `getattr` pattern). None or
        # empty → empty string (mflux argv emission gated on truthy).
        iterations.append(_assemble_iteration_no_style(
            args=args,
            prompt=prompt,
            merged_defaults=merged_defaults,
            be=be,
            binary=binary,
            input_path=None,  # t2i: no input photo
            output_path=output_path,
            width=width,
            height=height,
            seed=iter_seed,
            style_name="draw",
            negative=getattr(args, "negative_prompt", None) or "",
        ))

    return iterations


def build_draw_iteration(
    *,
    args,
    prompt: str,
    merged_defaults: dict,
    be,
    binary: Path,
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
        binary=binary,
        width=width,
        height=height,
        explicit_output=explicit_output,
        run_dir=run_dir,
        base_seed=seed,
        num_iterations=1,
    )[0]


def build_refine_iteration(
    *,
    args,
    input_path: Path,
    prompt: str,
    merged_defaults: dict,
    be,
    binary: Path,
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
        :func:`build_mflux_cmd`.
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
        binary=binary,
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
    binary: Path,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
) -> Iteration:
    """Build a single :class:`Iteration` for bare i2i (`imgen generate
    <photo> --custom-prompt "..."` / `imgen batch <dir> --custom-prompt
    "..."` without ``--style``).

    v0.7.13 (gap 8 behaviour pivot): when the user omits ``--style``
    and supplies a prompt source instead, run i2i with NO preset
    baggage — empty :class:`Style` preset, no scope substitution, no
    style-declared LoRA stacking, no preset ``negative_prompt`` field
    bleeding into argv. This was the silent footgun pre-v0.7.13: a
    bare ``imgen photo.jpg --custom "wearing red"`` augmented the
    DEFAULT style (pixar) with the user's text, and pixar's
    ``negative_prompt`` field then reached mflux argv — crashing on
    backends that reject negatives (flux2-klein-edit-9b) and silently
    biasing output on those that accept them.

    Differences from :func:`build_refine_iteration` (the other i2i
    "bare" path):
      * Output naming: ``<run_dir>/<input.stem>-bare.png`` (refine
        uses ``-refined`` suffix; bare uses ``-bare`` to mark the
        no-preset variant).
      * Prompt is user-supplied via ``--custom-prompt`` / ``--prompt-
        file`` (refine has a baked-in default + override). Caller is
        responsible for guaranteeing ``prompt`` is non-empty BEFORE
        calling this — empty-prompt mflux runs are a programmer error.

    Same shared core as :func:`build_draw_iterations` +
    :func:`build_refine_iteration`: empty :class:`Style` via
    :func:`_assemble_iteration_no_style`. CLI ``--lora REF`` still
    flows through (user can stack LoRAs in bare mode without picking
    a preset).

    Pure: no subprocess, no I/O beyond the ``next_available_path``
    probe.
    """
    if explicit_output is not None:
        output_path = explicit_output
    else:
        if run_dir is None:
            raise ValueError(
                "build_bare_i2i_iteration: either explicit_output or "
                "run_dir must be provided"
            )
        # `<input.stem>-bare.png`. next_available_path handles
        # collisions if the user re-bare-runs into the same run-dir.
        output_path = next_available_path(
            run_dir, f"{input_path.stem}-bare", suffix=".png",
        )

    # v0.7.13: shared core with build_draw_iterations / build_refine
    # _iteration. Style preset is intentionally empty — caller's
    # `prompt` IS the prompt body verbatim, no scope rewrite, no
    # preset negative.
    return _assemble_iteration_no_style(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        binary=binary,
        input_path=input_path,
        output_path=output_path,
        width=width,
        height=height,
        seed=seed,
        style_name="bare",
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
                       preset.prompt with optional scope substitution
      * ``negative`` : preset.negative (always a string, empty by default)

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

        negative = preset.negative

        # 2. Numeric parameter precedence (CLI > preview > preset >
        # defaults; rules vary per field — locked by tests).
        # v0.8.0 commit 7: per-Model defaults layer in via the
        # ``model`` argument; engine-level validation fires on the
        # resolved values per §M.
        model = _model_for_validate(args)
        params = _resolve_iteration_params(
            args=args, preset=preset, merged_defaults=merged_defaults,
            model=model,
        )
        validate_engine_params_or_die(
            model,
            quantize=params.final_quantize,
            guidance=params.final_guidance,
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
            model=be,
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

        # v0.8.2 M-1A: build GenParams in parallel with the legacy
        # argv ``cmd``. See identical block in
        # ``_assemble_iteration_no_style`` for the rationale.
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
            # v0.7.3: per-Iteration seed. i2i (cmd_generate/cmd_batch)
            # uses one seed across all M styles of a single input —
            # all iterations of one build_iterations call share the
            # same seed, equal to ctx.seed. Field set explicitly so
            # the run_one_iteration history serialiser reads from
            # ``it.seed`` (post-v0.7.3) uniformly across i2i + t2i.
            seed=seed,
            model=model,
            params=gen_params,
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
    # v0.8.0 commit 4b: argparse dest renamed `backend` → `model` in
    # lockstep with the registry source-of-truth flip. ``get_backend()``
    # remains a back-compat shim accepting BOTH v0.7 and v0.8 model
    # names — the value in ``args.model`` is v0.8-canonical post-4b
    # (resolver-translated), but the shim handles legacy v0.7 values
    # too (e.g. from Namespace fixtures or history-replay paths).
    backend = args.model
    be = get_backend(backend)

    # ── HF token (FLUX-specific legacy path) ─────────────────────
    token: str | None = None
    if be.needs_token:
        token = load_token()
        if not token:
            die("FLUX backend requires HuggingFace token",
                code=3,
                hint="Run: imgen setup   (or use --model qwen-image-edit-v1)")

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
    de-duped list (parser already rejected unknown names).

    v0.7.13 (gap 8 behaviour pivot): when ``--style`` is absent, return
    an empty list to signal "bare mode". The caller (cmd_generate /
    cmd_batch) routes the single bare iteration through
    :func:`_assemble_iteration_no_style` for pure-prompt i2i without
    preset baggage. Pre-v0.7.13 this fell back to
    ``merged_defaults["style"]`` (usually "pixar"), which silently
    leaked the preset's ``negative_prompt`` field into argv — the
    flux2-klein-edit-9b crash that gap 7 fixed at the backend side,
    plus the general "preset surprise" UX wart on every backend.

    The config.toml ``[defaults] style = "X"`` key is kept for backwards
    compatibility (won't fail schema validation) but is no longer
    consulted for fallback. Documented as deprecated in v0.7.13 release
    notes; targeted for removal in v0.8.

    **Pure**: this returns the resolved list and nothing else. The
    ``--output FILE`` + multi-style mutex check lives in
    ``commands/generate._check_output_style_mutex`` since
    ``imgen batch`` has no ``--output`` flag and the check would be a
    silent no-op there. (v0.3.0 architect NIT-4 / NIT-6.) The "bare
    mode" prompt-source check lives in cmd_generate / cmd_batch since
    the helper has no view of ``effective_custom_prompt`` resolution.
    """
    if args.style:
        return list(args.style)
    return []

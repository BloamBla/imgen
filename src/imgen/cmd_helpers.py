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

from .backends import Backend, get_backend
# v0.8.4 M-NEW-E: iteration-build helpers (build_iterations,
# build_draw_iterations + sibling builders, _assemble_iteration_no_style,
# _resolve_iteration_*, IterationParams + LoraResolution dataclasses,
# _model_for_validate, _flatten_cli_lora, resolve_effective_loras,
# prepend_trigger_words, check_prompt_style_compat, prompt_slug,
# _draw_output_path_for_index) extracted to build_iteration. Re-exported
# below for back-compat.
from .build_iteration import (
    IterationParams,
    LoraResolution,
    _assemble_iteration_no_style,
    _draw_output_path_for_index,
    _flatten_cli_lora,
    _model_for_validate,
    _resolve_iteration_loras,
    _resolve_iteration_params,
    _resolve_iteration_prompt,
    build_bare_i2i_iteration,
    build_draw_iteration,
    build_draw_iterations,
    build_iterations,
    build_refine_iteration,
    check_prompt_style_compat,
    prepend_trigger_words,
    prompt_slug,
    resolve_effective_loras,
)
from .checks import check_mflux, check_resources, check_venv
from .colors import C, die, err, info, ok, step, warn
from .config import effective_enhance, effective_output_dir
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


# ── Interactive confirm-gate helper ─────────────────────────────────────


def prompt_yes_no(question: str = "Continue? [y/N] ") -> bool:
    """Render ``question`` and return True iff the user types ``y`` /
    ``yes`` (case-insensitive, surrounding whitespace tolerated).

    Treats EOF (piped stdin closed) and KeyboardInterrupt (Ctrl-C) as
    "no" — prints a trailing newline so the shell prompt re-appears
    cleanly after the interrupt, then returns False.

    v0.9.4 D5 (python MED-5) consolidates the identical try/input/
    return-False boilerplate copied across 5 ``_confirm_*`` callers
    (cmd_draw / cmd_refine / cmd_generate / cmd_batch / cmd_video).
    Single source of truth for the [y/N] semantics: any future
    contract change (e.g. accept ``yeah``, treat ``yn`` as ambiguous)
    lands here.

    .. note::
        ``question`` flows verbatim to :func:`input` and reaches the
        terminal before blocking. Callers MUST pass either a literal
        constant or a string already wrapped via
        :func:`imgen._safe.safe_display` / :func:`safe_path_display`
        when any portion of it comes from untrusted input (history
        entries, hand-edited TOML values, file paths). v0.9.4 pre-tag
        security LOW contract note.
    """
    try:
        ans = input(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


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
    max_num_frames: int = 1,
) -> None:
    """Check RAM / disk / battery / parallel-mflux against the heaviest
    quant + largest output resolution + (v0.9) longest video in the
    batch. --force bypasses entirely. Hard failures share exit 4.

    Knobs scaling RAM math: ``max_megapixels`` (v0.7.14 gap 6),
    ``max_num_frames`` (v0.9 commit 7.1 §R.2 HIGH-1 — video frame-
    activations + §L "+3 GB" video buffer).
    """
    if force:
        return
    res = check_resources(
        model, heaviest_quant, max_megapixels,
        num_frames=max_num_frames,
    )

    if res["other_mflux_pid"] is not None:
        die(f"Another mflux process is already running (PID "
            f"{res['other_mflux_pid']}). Two parallel runs will OOM and "
            "trash each other.",
            code=4,
            hint="Wait for it to finish (check with: ps -p "
                 f"{res['other_mflux_pid']}), or pass --force.")

    if not res["ram_ok"]:
        # :.1f float format avoids "14.239999..." garbage in stderr
        # (ram_required_gb returns a float since v0.7.14).
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

    # v0.9 commit 7.1 (§R.2 HIGH-1 / §L locked buffer): video
    # Models get +3 GB headroom over the base estimate — T5 encoder
    # transient + transformer dispatch peak more sharply than image
    # inference. The v0.8.2 absolute < 4 GB safety net stays the
    # catastrophic backstop (architect §R.1 LOW-1 two-gate composition).
    if max_num_frames > 1:
        video_buffer_gb = 3.0
        required_with_buffer = res["ram_required_gb"] + video_buffer_gb
        if (res["ram_total_gb"] != 0 and
                res["ram_available_gb"] < required_with_buffer):
            # v0.9.2 B-8: mirror the image-preflight 'How to fix' bullet
            # shape with video-appropriate knobs (LTX has no --preview
            # and no --quantize so those don't apply; lower resolution
            # and shorter clip are the equivalent dials).
            die(
                f"Insufficient RAM for video generation: need "
                f"~{res['ram_required_gb']:.1f} GB + {video_buffer_gb:.1f} GB "
                f"safety buffer, have {res['ram_available_gb']:.1f} GB "
                f"available (of {res['ram_total_gb']:.0f} GB total).",
                code=4,
                hint=("How to fix:\n"
                      "     • Close other apps (Chrome often eats 5+ GB)\n"
                      "     • Lower resolution: --width 512 --height 384 "
                      "(or smaller)\n"
                      "     • Shorter clip: --num-frames 17 "
                      "(or --duration 0.7)\n"
                      "     • Or --force at your own risk "
                      "(video swap-thrashes hard)"),
            )

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


def load_backend_and_token(
    args,
) -> tuple[str, Backend, str | None, Path, tuple[str, str] | None]:
    """Resolve ``(backend_name, backend_dataclass, hf_token_or_None,
    binary_path, custom_backend_secret_or_None)``. The 5th slot is the
    v0.4 ``(env_var_name, value)`` pair for custom backends declaring
    ``[secret]``.

    Exits with code 3 on: gated built-in without HF token, custom
    backend missing required ``secret_env_var``, venv / mflux not
    installed, or binary not present (PATH for bare names, absolute
    for ``/``-prefixed). HF token loaded lazily only when
    ``needs_token``. v0.9 commit 7.1: diffusers_mps Models skip the
    mflux + binary check (Engine.run resolves .venv-diffusers/python
    internally with its own symlink + is_file guards).
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

    # ── venv + mflux + binary sanity (mflux engine only) ─────────
    #
    # v0.9 commit 7: diffusers_mps engine doesn't run mflux at all —
    # Engine.run dispatches via .venv-diffusers/bin/python directly.
    # Skip the mflux check + binary resolution for those Models;
    # DiffusersMpsEngine.run validates its own venv_python.is_file()
    # internally + ensure_video_deps_or_die handles deps.
    engine = getattr(be, "engine", "mflux")
    if engine == "diffusers_mps":
        # Sentinel — caller (orchestrator) unpacks as `_binary` and
        # ignores it for diffusers_mps Models. Path("") is a stable
        # falsy sentinel without breaking the Path return type.
        binary = Path("")
    else:
        if not check_venv() or not check_mflux():
            die("mflux not installed",
                code=3,
                hint="Run: imgen setup")
        # Binary path resolution
        if be.binary.startswith("/"):
            # Absolute path — validator already confirmed it exists at
            # schema time, but re-check here in case the file was
            # removed between TOML load and command execution.
            binary = Path(be.binary)
        else:
            # Bare name — resolve against VENV_BIN (mflux convention).
            binary = VENV_BIN / be.binary
        if not binary.is_file():
            # is_file() (not exists()) — a directory at the path
            # would crash subprocess.Popen with IsADirectoryError;
            # reject earlier with the imgen-flavoured error.
            # (v0.4 python-reviewer IMP-1.)
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

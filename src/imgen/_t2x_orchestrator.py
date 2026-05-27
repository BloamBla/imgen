"""v0.9 commit 7 — shared t2x (text-to-X) orchestrator.

Per [[project-v090-design]] §I.0. Extracted from the pre-v0.9 cmd_draw
body so cmd_draw and cmd_video share the 12-step pipeline without
duplication (architect §R.1 HIGH-3: "cmd_video SHALL NOT duplicate
80% of cmd_draw"). cmd_refine + cmd_generate may eventually route
through here too; their differences (style presets, i2i input flow)
make the extraction non-trivial and the architect deferred to v0.9.x.

The orchestrator parameterises three command-specific surfaces:

* ``build_iterations_fn`` — callable returning a list of Iteration.
  draw passes ``build_draw_iterations`` (N-iter ladder); video passes
  ``build_video_iteration`` (always returns a 1-element list).
* ``confirm_fn`` — callable rendering the per-command confirm gate.
  draw shows "N images + seed ladder"; video shows
  "duration + fps + dimensions".
* ``post_success_hint_fn`` — optional callback for the post-run hint.
  draw surfaces the refine chain; video skips (no v0.9.0 chained UX).

Lives in its own module (not cmd_helpers) per the 800-LoC ceiling
discipline in CLAUDE.md — cmd_helpers was already at 789 LoC at
v0.8.5; the orchestrator at ~150 LoC would push it over.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

from .colors import die, warn
from .defaults import DEFAULTS
from .history import load_history
from .prompt_input import PromptInputError, resolve_prompt
from .runs import BatchContext
from .subprocess_helpers import build_mflux_env

__all__ = ["_resolve_t2x_prompt", "_orchestrate_t2x"]


def _resolve_t2x_prompt(args, *, command: str) -> str:
    """Pick exactly one of: positional prompt / --prompt-file PATH /
    positional '-' (stdin). Shared between cmd_draw and cmd_video
    (both have identical prompt-input semantics — t2x is "type-of-X
    prompt-driven generation, no input media").

    The mutex check + size-cap + empty-check + stdin route are
    identical to the pre-v0.9 cmd_draw._resolve_draw_prompt;
    extracted here for cmd_video reuse per §I.0.

    ``command`` parameterises the error messages so users see
    "draw: prompt required" / "video: prompt required" rather than
    generic "t2x: ..." text. Pure UX cosmetic; logic identical.
    """
    positional = getattr(args, "prompt", None)
    prompt_file = getattr(args, "prompt_file", None)

    if positional and prompt_file:
        die(
            f"{command}: positional prompt and --prompt-file are "
            "mutually exclusive (pick one)",
            code=2,
        )

    if positional == "-":
        try:
            text = resolve_prompt(
                custom_prompt="-", prompt_file=None, stdin=sys.stdin,
            )
        except PromptInputError as e:
            die(f"{command}: {e}", code=2)
        if text is None or not text.strip():
            die(f"{command}: stdin prompt was empty", code=2)
        return text.strip()

    if positional:
        if not positional.strip():
            die(f"{command}: prompt must be non-empty", code=2)
        return positional.strip()

    if prompt_file:
        try:
            text = resolve_prompt(
                custom_prompt=None, prompt_file=prompt_file,
            )
        except PromptInputError as e:
            die(str(e), code=2)
        if text is None or not text.strip():
            die(f"{command}: prompt-file was empty", code=2)
        return text.strip()

    die(
        f"{command}: a prompt is required — pass it as the positional "
        "argument, --prompt-file PATH, or '-' for stdin",
        code=2,
    )


def _orchestrate_t2x(
    args,
    *,
    command: str,
    build_iterations_fn: Callable,
    confirm_fn: Callable,
    enhancer_die_early_message: Optional[str] = None,
    pre_dispatch_fn: Optional[Callable[[], None]] = None,
    post_success_hint_fn: Optional[Callable] = None,
) -> int:
    """v0.9 commit 7: 12-step t2x pipeline shared between cmd_draw +
    cmd_video. Returns the exit code (0 / 130 on KeyboardInterrupt /
    1+ on failure).

    Parameterised:

    * ``command`` — drives BatchContext.command, history replay
      routing, dry-run banner text. "draw" or "video" today.
    * ``build_iterations_fn`` — kwargs-only callable building a list
      of :class:`Iteration`. Signature must accept all of:
      ``args``, ``prompt``, ``merged_defaults``, ``be``, ``width``,
      ``height``, ``explicit_output``, ``run_dir``, ``base_seed``,
      ``num_iterations``. Both build_draw_iterations and
      build_video_iteration conform.
    * ``confirm_fn`` — kwargs-only callable returning bool (True =
      proceed, False = cancel). Signature: ``(prompt, iterations,
      run_dir, slug, eta_seconds)``. Each command renders its own
      confirm text (draw shows N+ladder; video shows duration+fps).
    * ``enhancer_die_early_message`` — when set, ``args.enhance=True``
      triggers ``die(message, code=2)`` BEFORE any enhancer dispatch.
      cmd_video uses this per §S.4 (LTX has no enhancer in v0.9.0).
      cmd_draw passes None (enhancer is supported).
    * ``pre_dispatch_fn`` — zero-arg callable called AFTER preflight +
      BEFORE confirm gate. cmd_video uses this for
      ``ensure_video_deps_or_die`` (per §E.5.7 dry-run skips this).
    * ``post_success_hint_fn`` — optional kwargs-only callable shown
      after a successful run. cmd_draw uses it for the refine hint;
      cmd_video passes None (no chained UX for v0.9.0).

    Pre-conditions:
    * args has all the t2x flags resolved (parser already validated).
    * For video commands, args.num_frames + args.fps are populated
      from the --duration/--num-frames mutex resolution.
    """
    # Lazy imports — keeps the orchestrator module's top-level import
    # graph shallow. cmd_helpers + engine_dispatch are big modules; we
    # only need their function exports inside this body.
    from .cmd_helpers import (
        emit_gated_repo_hint_if_failed,
        estimate_one_seconds,
        exit_code,
        load_backend_and_token,
        maybe_enhance_prompts,
        megapixels_of,
        open_results,
        preflight_resources,
        print_batch_summary,
        prompt_slug,
        resolve_enhance_config,
        resolve_output_layout,
        run_one_iteration,
    )

    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1. Prompt resolution (shared positional / --prompt-file / stdin).
    prompt = _resolve_t2x_prompt(args, command=command)

    # 2. Backend + token. flux-dev / LTX both reach load_backend_and_token;
    # gated-repo check fires inside.
    backend, be, token, _binary, backend_secret = load_backend_and_token(args)

    # 3. Output layout + N-iteration mutex check.
    explicit_output, run_dir = resolve_output_layout(args, config_output_dir)
    num_iterations = getattr(args, "num_iterations", 1)
    if explicit_output is not None and num_iterations > 1:
        die(
            f"--output PATH is mutex with --num-iterations "
            f"{num_iterations} (single output FILE can't fan out to "
            f"{num_iterations} outputs). Use --output-dir DIR instead.",
            code=2,
        )

    # 4. Base seed (explicit --seed → deterministic; else random).
    base_seed = (
        args.seed if args.seed is not None
        else int.from_bytes(os.urandom(4), "big")
    )

    # 5. Enhancer with per-command gate.
    if enhancer_die_early_message is not None and getattr(args, "enhance", False):
        die(enhancer_die_early_message, code=2)
    eff_enhance = resolve_enhance_config(
        cli_enable=getattr(args, "enhance", None),
        cli_model=getattr(args, "enhance_model", None),
        cli_temperature=getattr(args, "enhance_temperature", None),
        config_enhance=getattr(args, "imgen_config_enhance", {}),
    )
    unique_enhance_results, enhance_model = maybe_enhance_prompts(
        eff_enhance=eff_enhance,
        backend_obj=be,
        prompts=[prompt],
    )
    enhanced_prompt = unique_enhance_results[0].final_prompt
    # Replicate the SINGLE result across N iterations so each history
    # entry records the enhance metadata. Same v0.7.3 broadcast
    # rationale as cmd_draw.
    enhance_results = [unique_enhance_results[0]] * num_iterations

    # 6. Build N iterations (caller-provided factory).
    iterations = build_iterations_fn(
        args=args,
        prompt=enhanced_prompt,
        merged_defaults=merged_defaults,
        be=be,
        width=args.width,
        height=args.height,
        explicit_output=explicit_output,
        run_dir=run_dir,
        base_seed=base_seed,
        num_iterations=num_iterations,
    )

    total = len(iterations)
    is_batch = total >= 2

    # 7. Dry-run: print every cmd, exit clean. Skips
    # pre_dispatch_fn (per §E.5.7 for video deps install).
    if args.dry_run:
        from .engine_dispatch import iteration_dryrun_display
        for idx, it in enumerate(iterations, start=1):
            print(f"Dry run [{idx}/{total}] — would execute ({command}):")
            print()
            print(iteration_dryrun_display(it))
            print()
        return 0

    # 8. Resource preflight (per-Model RAM math via Engine.ram_estimate_gb).
    # v0.9 commit 7.1 (§R.2 HIGH-1): max_num_frames threaded so the
    # video frame-term (0.1 GB per frame) is included AND the §L
    # "+3 GB video safety buffer" gate fires when iteration plans
    # video output.
    heaviest_quant = max(it.final_quantize for it in iterations)
    max_megapixels = megapixels_of(args.width, args.height)
    max_num_frames = max(it.params.num_frames for it in iterations)
    preflight_resources(
        model=backend, heaviest_quant=heaviest_quant,
        force=args.force, max_megapixels=max_megapixels,
        max_num_frames=max_num_frames,
    )

    # 9. Pre-dispatch hook — e.g. ensure_video_deps_or_die for
    # cmd_video. Fires AFTER preflight (no point installing deps for
    # a job that won't fit in RAM) and BEFORE the confirm gate
    # (otherwise user might say yes only to hit the install prompt
    # and re-confirm; cleaner to surface install before confirm).
    if pre_dispatch_fn is not None:
        pre_dispatch_fn()

    # 10. Confirm gate (skipped under --yes).
    if not args.yes:
        one_eta = estimate_one_seconds(
            load_history(), backend, heaviest_quant,
            getattr(args, "preview", False),
        )
        proceed = confirm_fn(
            prompt=iterations[0].prompt,
            iterations=iterations,
            run_dir=run_dir,
            slug=prompt_slug(enhanced_prompt),
            eta_seconds=one_eta,
        )
        if not proceed:
            warn("Cancelled — nothing generated.")
            return 0

    # 11. mkdir.
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    elif explicit_output is not None:
        explicit_output.parent.mkdir(parents=True, exist_ok=True)

    # 12. BatchContext + run loop.
    env = build_mflux_env(token=token, backend_secret=backend_secret)
    ctx = BatchContext(
        model=backend,
        seed=base_seed,
        width=args.width,
        height=args.height,
        input_path=None,  # t2x — no source media
        effective_custom_prompt=None,
        args=args,
        batch_id=None,
        env=env,
        command=command,
    )

    succeeded: list[tuple[str, Path, int]] = []
    failed: list[tuple[str, int, Path]] = []
    for idx, it in enumerate(iterations, start=1):
        cont = run_one_iteration(
            it=it,
            idx=idx,
            total=total,
            is_batch=is_batch,
            ctx=ctx,
            logger=None,
            succeeded=succeeded,
            failed=failed,
            enhance_result=enhance_results[idx - 1],
            enhance_model=enhance_model,
        )
        if not cont:
            # KeyboardInterrupt during iteration. Surface partial-run UX.
            if succeeded and run_dir is not None and not args.no_open:
                open_results(
                    succeeded=succeeded,
                    run_dir=run_dir,
                    is_batch=True,
                    no_open=args.no_open,
                )
            if is_batch and (succeeded or failed):
                print_batch_summary(succeeded, failed, total=idx - 1)
            return 130

    # Post-run: gated-repo hint, results open, summary, hint, exit code.
    emit_gated_repo_hint_if_failed(failed=failed, backend_obj=be)

    # v0.9.4 D3: ``Path()`` (= cwd) fallback replaced with ``None``.
    # ``open_results`` skips the Finder open when ``run_dir`` is None;
    # the pre-fix expression would have silently opened the process's
    # cwd in Finder if both run_dir and explicit_output were ever None
    # (today unreachable per resolve_output_layout invariant). For
    # ``is_batch=True`` paths the Finder skip is the safer no-op vs the
    # pre-fix silent-wrong-dir open.
    open_results(
        succeeded=succeeded,
        run_dir=run_dir if run_dir is not None else (
            explicit_output.parent if explicit_output is not None else None
        ),
        is_batch=is_batch,
        no_open=args.no_open,
    )

    if is_batch:
        print_batch_summary(succeeded, failed, total=total)

    if post_success_hint_fn is not None:
        post_success_hint_fn(
            succeeded=succeeded,
            failed=failed,
            is_batch=is_batch,
            run_dir=run_dir,
        )

    return exit_code(is_batch=is_batch, succeeded=succeeded, failed=failed)

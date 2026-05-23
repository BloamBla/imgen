"""`imgen draw <prompt>` — text-to-image generation.

v0.7.0 — first non-i2i surface on imgen. The user-facing cash-in of
the identity pivot (see memory/project_identity_2026_05_23.md): imgen
goes from "photo style transfer CLI" to "local image generation CLI".

Single-shot in v0.7.0: one prompt → one image. Multi-iter randomness
ladders (`--num-iterations N`) and prompt-file fan-out
(`--from-file prompts.txt`) deferred to v0.7.x per
memory/project_v070_design.md §O anti-scope.

Pipeline mirrors :func:`~imgen.commands.generate.cmd_generate` with
the i2i-specific branches removed (no input photo, no HEIC
pre-conversion, no per-input loop, no `--style`/`--scope`/`--strength`).
Reuses every shared helper in :mod:`imgen.cmd_helpers`:

* :func:`~imgen.cmd_helpers.load_backend_and_token` — same gated-
  token contract as Kontext (flux-dev shares `~/.imgen/hf_token`).
* :func:`~imgen.cmd_helpers.resolve_output_layout` — `--output PATH`
  vs `--output-dir DIR`/timestamped run folder.
* :func:`~imgen.cmd_helpers.build_draw_iteration` — t2i sibling of
  build_iterations, returns one frozen Iteration.
* :func:`~imgen.cmd_helpers.maybe_enhance_for_command` — the LLM
  enhancer. FLUX.1-dev backend declares
  `enhance_invariants=()` so the runner_error / empty_llm_output /
  input_too_long paths still surface but no substring check fires.
* :func:`~imgen.cmd_helpers.run_one_iteration` — single subprocess.
  BatchContext.input_path is None, BatchContext.command="draw".
* :func:`~imgen.cmd_helpers.open_results`,
  :func:`~imgen.cmd_helpers.exit_code` — closing UX.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..colors import C, die, info, warn
from ..defaults import DEFAULTS
from ..history import load_history
from ..prompt_input import PromptInputError, resolve_prompt
from ..runs import BatchContext
from ..cmd_helpers import (
    build_draw_iterations,
    emit_gated_repo_hint_if_failed,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    maybe_enhance_prompts,
    open_results,
    preflight_resources,
    print_batch_summary,
    prompt_slug,
    resolve_enhance_config,
    resolve_output_layout,
    run_one_iteration,
)
from ..subprocess_helpers import build_mflux_env, format_cmd

__all__ = ["cmd_draw"]


def _resolve_draw_prompt(args) -> str:
    """Pick exactly one of: positional prompt / --prompt-file PATH /
    positional '-' (stdin).

    Parser declares the positional as optional + adds a separate
    --prompt-file flag. Mutex check + size-cap + empty-check happen
    here.

    v0.7.0 (security pre-tag review IMPORTANT): the stdin and
    --prompt-file paths both delegate to
    :func:`~imgen.prompt_input.resolve_prompt` which carries the
    64 KB cap (``cat /dev/zero | imgen draw -`` would OOM without
    it). The positional non-empty case stays inline since
    argparse-level argv lengths are already POSIX ARG_MAX-bounded
    (~256 KB-1 MB depending on platform) and the prompt would have
    been rejected long before reaching here.
    """
    positional = args.prompt
    prompt_file = getattr(args, "prompt_file", None)

    if positional and prompt_file:
        die(
            "draw: positional prompt and --prompt-file are mutually "
            "exclusive (pick one)",
            code=2,
        )

    # Stdin via positional '-' — mirror of cmd_generate's --custom-prompt -
    # contract. Hides prompt from `ps auxww`. Routes through
    # resolve_prompt to inherit the 64 KB cap + UTF-8 validation +
    # empty-check (security IMPORTANT, v0.7.0 pre-tag review).
    if positional == "-":
        # Pass `sys.stdin` explicitly so call-time mutations (test
        # monkeypatches; future programmatic redirections) take effect.
        # resolve_prompt's default-arg `stdin=sys.stdin` is bound at
        # def-time; monkeypatch.setattr("sys.stdin", ...) after import
        # wouldn't reach it otherwise.
        try:
            text = resolve_prompt(
                custom_prompt="-", prompt_file=None, stdin=sys.stdin,
            )
        except PromptInputError as e:
            die(f"draw: {e}", code=2)
        if text is None or not text.strip():
            die("draw: stdin prompt was empty", code=2)
        return text.strip()

    if positional:
        if not positional.strip():
            die("draw: prompt must be non-empty", code=2)
        return positional.strip()

    if prompt_file:
        # resolve_prompt carries the size-cap + permission-warn +
        # UTF-8 validation contract.
        try:
            text = resolve_prompt(
                custom_prompt=None, prompt_file=prompt_file,
            )
        except PromptInputError as e:
            die(str(e), code=2)
        if text is None or not text.strip():
            die("draw: prompt-file was empty", code=2)
        return text.strip()

    die(
        "draw: a prompt is required — pass it as the positional "
        "argument, --prompt-file PATH, or '-' for stdin",
        code=2,
    )


def _confirm_draw(
    *,
    prompt: str,
    num_iterations: int,
    first_output: Path,
    run_dir: Path | None,
    slug: str,
    eta_seconds: int | None,
) -> bool:
    """Confirm gate. Shows prompt preview, output target, ETA × N.

    For ``num_iterations == 1`` displays the exact output path (or
    ``--output FILE`` if set). For ``num_iterations >= 2`` displays
    the run-dir + count — listing N filenames would bloat the gate
    line without adding info (slug is shared, only the ``-1.png``
    suffix varies).

    v0.7.3 fix (python IMP / security NIT-2): the range display was
    deriving slug via ``first_output.stem.rsplit('-', 1)[0]`` which
    failed cosmetically when ``next_available_path`` inserted a
    collision suffix on the first iteration (e.g. ``<slug>-1-2.png``
    → stem ``<slug>-1-2`` → rsplit yielding ``<slug>-1`` not
    ``<slug>``). Pass the canonical slug explicitly.
    """
    preview = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    if num_iterations == 1:
        info("About to generate 1 image:")
        print(f"   {C.DIM}prompt:{C.END} {preview}")
        print(f"   {C.DIM}output:{C.END} {first_output}")
    else:
        info(f"About to generate {num_iterations} images (seed ladder):")
        print(f"   {C.DIM}prompt:{C.END} {preview}")
        # run_dir is non-None here because explicit_output is mutex
        # with num_iterations > 1 (validated at build_draw_iterations).
        print(f"   {C.DIM}output:{C.END} {run_dir}/")
        print(f"   {C.DIM}count:{C.END}  {num_iterations} variations "
              f"({slug}-1..-{num_iterations}.png)")
    if eta_seconds is not None:
        total_eta = eta_seconds * num_iterations
        per_image = format_duration(eta_seconds)
        if num_iterations == 1:
            print(f"   {C.DIM}eta:{C.END}    {per_image} (±50%)")
        else:
            print(f"   {C.DIM}eta:{C.END}    {format_duration(total_eta)} total "
                  f"({per_image} per image, ±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def cmd_draw(args) -> int:
    """`imgen draw <prompt>` — text-to-image with optional N-iteration
    seed-ladder explore mode (v0.7.3+).

    Pipeline:

    1. Resolve prompt (positional / --prompt-file / '-' stdin).
    2. Load backend + token; flux-dev needs the gated HF token.
    3. Resolve output layout. Mutex: ``--output FILE`` + N>=2 rejected
       (single file can't fan out to N images).
    4. Base seed: explicit ``--seed X`` (deterministic ladder X..X+N-1)
       or random.
    5. **Enhancer fires ONCE on the unique prompt** (regardless of N)
       via :func:`maybe_enhance_prompts` — same prompt → same enhanced
       text, so paying the LLM cost N× would be wasteful. The single
       EnhanceResult is replicated across the N iterations for history.
    6. Build N iterations via :func:`build_draw_iterations` using the
       enhanced prompt + seed ladder.
    7. Dry-run path: print all N cmds + exit.
    8. Resource preflight.
    9. Confirm gate (unless --yes) — shows N + ETA × N.
    10. mkdir, loop run_one_iteration N times, open the run-dir, exit.
    """
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Prompt resolution.
    prompt = _resolve_draw_prompt(args)

    # 2) Backend + token. flux-dev needs the gated HF token.
    backend, be, token, binary, backend_secret = load_backend_and_token(args)

    # 3) Output layout + N-iteration mutex check.
    explicit_output, run_dir = resolve_output_layout(args, config_output_dir)
    num_iterations = getattr(args, "num_iterations", 1)
    if explicit_output is not None and num_iterations > 1:
        die(
            f"--output PATH is mutex with --num-iterations {num_iterations} "
            f"(single output FILE can't fan out to {num_iterations} images). "
            f"Use --output-dir DIR instead.",
            code=2,
        )

    # 4) Base seed.
    base_seed = (
        args.seed if args.seed is not None
        else int.from_bytes(os.urandom(4), "big")
    )

    # 5) Enhancer fires ONCE for the unique prompt (v0.7.3 optimisation —
    # all N seed-variants share the same prompt; paying the LLM cost N×
    # would burn ~5s × N for an identical answer). The single
    # EnhanceResult is replicated below for history alignment.
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
    # entry records the enhance metadata (was_enhanced / fallback_reason
    # / fallback_detail / model). They all share the same enhance
    # decision since the prompt was the same.
    #
    # INVARIANT: this broadcast assumes all N iterations share ONE
    # input prompt. The v0.7.x roadmap mentions --prompt-modifier /
    # per-iteration prompt variation (v0.7.0 §O list); when that
    # lands, switch `maybe_enhance_prompts(prompts=[...])` to N
    # distinct prompts and drop this multiplication. EnhanceResult
    # is frozen+slots so the shared reference can't be mutated, but
    # the cardinality assumption WILL break if N prompts diverge.
    enhance_results = [unique_enhance_results[0]] * num_iterations

    # 6) Build N iterations using the (possibly enhanced) prompt + seed
    # ladder. `iters[0]` uses base_seed, `iters[1]` uses base_seed+1, etc.
    iterations = build_draw_iterations(
        args=args,
        prompt=enhanced_prompt,
        merged_defaults=merged_defaults,
        be=be,
        binary=binary,
        width=args.width,
        height=args.height,
        explicit_output=explicit_output,
        run_dir=run_dir,
        base_seed=base_seed,
        num_iterations=num_iterations,
    )

    total = len(iterations)
    is_batch = total >= 2

    # 7) Dry-run: print every N cmds, exit clean.
    if args.dry_run:
        for idx, it in enumerate(iterations, start=1):
            print(f"Dry run [{idx}/{total}] — would execute (draw):")
            print()
            print(format_cmd(it.cmd))
            print()
        return 0

    # 8) Resource preflight — heaviest quant in the batch (all N share
    # the same backend + quant today, but max() keeps it future-proof
    # if per-iteration quant ever lands).
    heaviest_quant = max(it.final_quantize for it in iterations)
    preflight_resources(
        backend=backend, heaviest_quant=heaviest_quant, force=args.force,
    )

    # 9) Confirm gate (unless --yes). Shows count + ETA × N for N>=2.
    if not args.yes:
        one_eta = estimate_one_seconds(
            load_history(), backend, heaviest_quant, args.preview,
        )
        proceed = _confirm_draw(
            prompt=iterations[0].prompt,
            num_iterations=total,
            first_output=iterations[0].output_path,
            run_dir=run_dir,
            slug=prompt_slug(enhanced_prompt),
            eta_seconds=one_eta,
        )
        if not proceed:
            warn("Cancelled — nothing generated.")
            return 0

    # 10) Materialise the output directory now that the user committed.
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    elif explicit_output is not None:
        explicit_output.parent.mkdir(parents=True, exist_ok=True)

    # 11) BatchContext for the run loop. input_path=None signals t2i to
    # run_one_iteration's history-entry + step() display gates.
    # command="draw" drives future replay routing.
    # v0.7.3: ctx.seed is the BASE seed of the ladder; each iteration's
    # actual seed lives on Iteration.cmd's argv. The ctx.seed value
    # ends up in history rows as a record of "what was the base seed"
    # — replay rehydrates per-iteration seeds from each entry's stored
    # `seed` field (already per-row in history).
    env = build_mflux_env(token=token, backend_secret=backend_secret)
    ctx = BatchContext(
        backend=backend,
        seed=base_seed,
        width=args.width,
        height=args.height,
        input_path=None,
        effective_custom_prompt=None,
        args=args,
        batch_id=None,
        env=env,
        command="draw",
    )

    # 12) Loop over the N iterations. KeyboardInterrupt mid-loop
    # returns 130 with whatever's been generated so far (open_results
    # below still opens the partial run-dir).
    # v0.7.4 python NIT-2: tighten the bare `list` annotations to the
    # concrete tuple shapes that `open_results` / `print_batch_summary`
    # / `exit_code` consume. Matches the i2i orchestrators' style.
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
            # KeyboardInterrupt during iteration `idx`. Partial-run UX:
            # surface what completed, open run_dir so the user can see
            # the N-1 (or fewer) images that did finish.
            if succeeded and run_dir is not None and not args.no_open:
                open_results(
                    succeeded=succeeded,
                    run_dir=run_dir,
                    is_batch=True,
                    no_open=args.no_open,
                )
            # v0.7.3 python IMP: surface the "K ok / J failed" summary
            # for the completed slots so the user sees what did finish.
            # `total=idx - 1` (completed slots) instead of `total` (the
            # full N) avoids "0/0 of N" confusion when interrupted
            # before the first image even started writing.
            if is_batch and (succeeded or failed):
                print_batch_summary(succeeded, failed, total=idx - 1)
            return 130

    # v0.7.0 (post-tag review UX-gap): if mflux exited non-zero AND
    # the backend declares a gated HF repo, surface a friendly hint
    # pointing at the per-model license page. v0.7.1 extracted to
    # cmd_helpers.emit_gated_repo_hint_if_failed so cmd_generate +
    # cmd_batch get the same hint for FLUX-Kontext cold installs.
    emit_gated_repo_hint_if_failed(failed=failed, backend_obj=be)

    # 13) Open result + summary + exit code. N>=2 → open Finder on the
    # run-dir; N=1 → open the single PNG in Preview (or run_dir if
    # the --output explicit-path case kept its parent in scope).
    open_results(
        succeeded=succeeded,
        run_dir=run_dir if run_dir is not None else (
            explicit_output.parent if explicit_output is not None else Path()
        ),
        is_batch=is_batch,
        no_open=args.no_open,
    )

    if is_batch:
        print_batch_summary(succeeded, failed, total=total)
    return exit_code(is_batch=is_batch, succeeded=succeeded, failed=failed)

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
    apply_enhance_results_to_iterations,
    build_draw_iteration,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    maybe_enhance_for_command,
    open_results,
    preflight_resources,
    print_batch_summary,
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
    --prompt-file flag. The mutex check happens here because argparse's
    mutually_exclusive_group doesn't compose cleanly with an optional
    positional.
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
    # contract. Hides prompt from `ps auxww`.
    if positional == "-":
        try:
            text = sys.stdin.read()
        except Exception as e:
            die(f"draw: failed to read prompt from stdin: {e}", code=2)
        if not text.strip():
            die("draw: stdin prompt was empty", code=2)
        return text.strip()

    if positional:
        if not positional.strip():
            die("draw: prompt must be non-empty", code=2)
        return positional.strip()

    if prompt_file:
        # Reuse the prompt_input.resolve_prompt size-cap + readability
        # checks by calling it with custom_prompt=None + prompt_file=path.
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


def _confirm_draw(prompt: str, output: Path, eta_seconds: int | None) -> bool:
    """Single-shot confirm gate. Shows the prompt preview, output path,
    and (when history has a comparable run) an ETA estimate."""
    preview = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    info("About to generate 1 image:")
    print(f"   {C.DIM}prompt:{C.END} {preview}")
    print(f"   {C.DIM}output:{C.END} {output}")
    if eta_seconds is not None:
        print(f"   {C.DIM}eta:{C.END}    {format_duration(eta_seconds)} (±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def cmd_draw(args) -> int:
    """`imgen draw <prompt>` — text-to-image single-shot.

    Pipeline:

    1. Resolve prompt (positional / --prompt-file / '-' stdin).
    2. Load backend + token; flux-dev needs the gated HF token.
    3. Resolve output layout (--output PATH or run-dir + slug).
    4. Build the single Iteration via build_draw_iteration.
    5. Optionally enhance prompt via the LLM (flux-dev's
       enhance_system_prompt is t2i-tuned per memory v0.7.0 design §K).
    6. Dry-run path: print cmd + exit.
    7. Resource preflight.
    8. Confirm gate (unless --yes).
    9. mkdir, run_one_iteration, open the result, summarise, exit.
    """
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Prompt resolution. Hidden-from-ps paths (stdin, --prompt-file)
    # work identically to cmd_generate's --custom-prompt = '-'.
    prompt = _resolve_draw_prompt(args)

    # 2) Backend + token. flux-dev needs the gated HF token (shared with
    # Kontext's `~/.imgen/hf_token` file). load_backend_and_token also
    # locates the mflux binary inside the venv.
    backend, be, token, binary, backend_secret = load_backend_and_token(args)

    # 3) Output layout. --output FILE wins; otherwise run-dir + slug.
    explicit_output, run_dir = resolve_output_layout(args, config_output_dir)

    # 4) Seed: explicit or random.
    seed = (
        args.seed if args.seed is not None
        else int.from_bytes(os.urandom(4), "big")
    )

    # 5) Build the single Iteration. Output path resolution lives inside
    # build_draw_iteration (it knows the prompt-slug naming convention).
    iteration = build_draw_iteration(
        args=args,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        binary=binary,
        width=args.width,
        height=args.height,
        explicit_output=explicit_output,
        run_dir=run_dir,
        seed=seed,
    )
    iterations = [iteration]

    # 6) Optional LLM enhancer. flux-dev declares enhance_invariants=()
    # so the substring-anchor checks short-circuit cleanly — runner_error
    # / empty_llm_output / input_too_long content paths still apply.
    eff_enhance = resolve_enhance_config(
        cli_enable=getattr(args, "enhance", None),
        cli_model=getattr(args, "enhance_model", None),
        cli_temperature=getattr(args, "enhance_temperature", None),
        config_enhance=getattr(args, "imgen_config_enhance", {}),
    )
    enhance_results, enhance_model = maybe_enhance_for_command(
        eff_enhance=eff_enhance,
        backend_obj=be,
        iterations=iterations,
    )
    iterations = apply_enhance_results_to_iterations(
        iterations, enhance_results,
    )

    # 7) Dry-run: print the cmd that would execute, exit clean.
    if args.dry_run:
        for it in iterations:
            print(f"Dry run — would execute (draw):")
            print()
            print(format_cmd(it.cmd))
            print()
        return 0

    # 8) Resource preflight — single iteration but the same RAM /
    # parallel-mflux / battery checks apply.
    heaviest_quant = iterations[0].final_quantize
    preflight_resources(
        backend=backend, heaviest_quant=heaviest_quant, force=args.force,
    )

    # 9) Confirm gate (unless --yes). Single-shot UX: brief prompt +
    # output preview + optional ETA from comparable history rows.
    if not args.yes:
        one_eta = estimate_one_seconds(
            load_history(), backend, heaviest_quant, args.preview,
        )
        proceed = _confirm_draw(
            prompt=iterations[0].prompt,
            output=iterations[0].output_path,
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
    env = build_mflux_env(token=token, backend_secret=backend_secret)
    ctx = BatchContext(
        backend=backend,
        seed=seed,
        width=args.width,
        height=args.height,
        input_path=None,
        effective_custom_prompt=None,
        args=args,
        batch_id=None,
        env=env,
        command="draw",
    )

    # 12) Run the single iteration.
    succeeded: list = []
    failed: list = []
    cont = run_one_iteration(
        it=iterations[0],
        idx=1,
        total=1,
        is_batch=False,
        ctx=ctx,
        logger=None,
        succeeded=succeeded,
        failed=failed,
        enhance_result=enhance_results[0] if enhance_results else None,
        enhance_model=enhance_model,
    )
    if not cont:
        return 130  # KeyboardInterrupt mid-iteration.

    # 13) Open result + summary + exit code.
    if run_dir is not None:
        open_results(
            succeeded=succeeded,
            run_dir=run_dir,
            is_batch=False,
            no_open=args.no_open,
        )
    elif explicit_output is not None and not args.no_open:
        # --output PATH path: open the file directly. open_results
        # doesn't know about the explicit-output shape; use the same
        # `open -R` pattern as the single-file branch.
        try:
            import subprocess
            subprocess.Popen(["open", str(explicit_output)])
        except Exception:
            pass

    print_batch_summary(succeeded, failed, total=1)
    return exit_code(is_batch=False, succeeded=succeeded, failed=failed)

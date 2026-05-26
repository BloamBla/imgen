"""`imgen draw <prompt>` — text-to-image generation.

v0.7.0 — first non-i2i surface on imgen. v0.9 commit 7 refactor:
shared 12-step pipeline extracted to
:mod:`imgen._t2x_orchestrator` per
[[project-v090-design]] §I.0 so ``cmd_video`` can reuse it without
80% duplication. ``cmd_draw`` is now a ~50 LoC glue passing the
draw-specific factories (build_draw_iterations, _confirm_draw, the
refine-chain hint) through ``_orchestrate_t2x``.

Pipeline (owned by _orchestrate_t2x):

1. Resolve prompt (positional / --prompt-file / '-' stdin).
2. Load backend + token; flux-dev needs the gated HF token.
3. Resolve output layout. Mutex: --output FILE + N>=2 rejected.
4. Base seed: explicit --seed X (deterministic ladder X..X+N-1) or random.
5. Enhancer fires ONCE on the unique prompt (regardless of N).
6. Build N iterations via build_draw_iterations.
7. Dry-run: print all N cmds + exit.
8. Resource preflight.
9. Confirm gate (unless --yes) — shows N + ETA × N.
10. mkdir, loop run_one_iteration N times, open the run-dir, exit.

draw-specific behaviour: N-iteration seed ladder, "About to generate
N images" confirm text, post-success refine hint when N=1.
"""
from __future__ import annotations

from pathlib import Path

from ..colors import C, info
from ..cmd_helpers import (
    build_draw_iterations,
    format_duration,
)
from .._t2x_orchestrator import _orchestrate_t2x

__all__ = ["cmd_draw"]


def _confirm_draw(
    *,
    prompt: str,
    iterations,
    run_dir: Path | None,
    slug: str,
    eta_seconds: int | None,
) -> bool:
    """Confirm gate. Shows prompt preview, output target, ETA × N.

    Signature matches the orchestrator's confirm_fn contract — receives
    the full iterations list + slug + eta. Derives N + first_output
    from iterations[0].
    """
    num_iterations = len(iterations)
    first_output = iterations[0].output_path
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


def _draw_post_success_hint(*, succeeded, failed, is_batch, run_dir) -> None:
    """v0.7.7 UX hint: refine chain isn't obvious from `imgen draw`
    alone — surface it on success so users discover the
    explore→refine workflow without reading the README.

    Single-shot only — N>=2 the user picks a winner via Finder first,
    so the hint there would be ambiguous. Skipped when --output sent
    the result to an explicit path the user controls.
    """
    if succeeded and not failed and not is_batch and run_dir is not None:
        _, single_output, _ = succeeded[0]
        info(
            f"Try `imgen refine {single_output}` "
            f"for sharper detail at 1.5×/2× resolution."
        )


def cmd_draw(args) -> int:
    """`imgen draw <prompt>` — t2i with optional N-iteration seed-ladder
    explore mode (v0.7.3+).

    v0.9 commit 7: delegates the entire 12-step pipeline to
    ``_orchestrate_t2x``. Provides three draw-specific factories:

    * ``build_draw_iterations`` — seed-ladder iteration list (N items).
    * ``_confirm_draw`` — "About to generate N images" confirm text.
    * ``_draw_post_success_hint`` — refine chain hint on single-shot
      success.

    Enhancer supported (no die-early gate); pre_dispatch_fn None (no
    lazy deps install for image path).
    """
    return _orchestrate_t2x(
        args,
        command="draw",
        build_iterations_fn=build_draw_iterations,
        confirm_fn=_confirm_draw,
        enhancer_die_early_message=None,
        pre_dispatch_fn=None,
        post_success_hint_fn=_draw_post_success_hint,
    )

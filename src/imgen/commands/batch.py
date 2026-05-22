"""`imgen batch <dir>` — apply M styles across every supported image
under ``<dir>``.

v0.3.0's only new subcommand. Designed as a thin orchestrator that
composes existing single-input helpers from ``commands/generate.py``:

* :func:`~imgen.commands.generate._resolve_styles_list` (output mutex
  is generate-specific; ``getattr`` lets the helper degrade cleanly
  here).
* :func:`~imgen.commands.generate._check_prompt_style_compat`
* :func:`~imgen.commands.generate._resolve_output_layout` (returns the
  ``run_dir`` branch — batch has no ``--output FILE`` flag).
* :func:`~imgen.commands.generate._load_backend_and_token`
* :func:`~imgen.commands.generate._build_iterations` — called once per
  input, the resulting lists are flattened for preflight.
* :func:`~imgen.commands.generate._preflight_resources` — guards against
  the heaviest quant across the full N×M grid.
* :func:`~imgen.commands.generate._run_one_iteration` — the single
  per-mflux unit, wrapped here in the per-input section + global index.
* :func:`~imgen.commands.generate._open_results`,
  :func:`~imgen.commands.generate._print_batch_summary`,
  :func:`~imgen.commands.generate._exit_code` — closing UX.

What's new in this module:

* :func:`_confirm_dir_batch` — N×M confirm gate (vs generate's 1×M one).
* HEIC pre-conversion via :mod:`~imgen.inputs` and a
  ``tempfile.TemporaryDirectory`` cache (auto-cleaned on exit, success
  or failure).
* Per-input section markers wrapping each input's M iterations in the
  shared per-batch log file.
* Global iteration numbering ``[k/N×M]`` so any line in the log is
  uniquely identifiable across the whole batch.
"""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
import uuid
from pathlib import Path

from ..colors import C, die, info, warn
from ..defaults import DEFAULTS
from ..history import load_history
from ..images import detect_resolution
from ..inputs import (
    check_input_stems,
    discover_inputs,
    resolve_to_mflux_input,
)
from ..prompt_input import PromptInputError, resolve_prompt
from ..runs import (
    BatchContext,
    BatchLogger,
    Iteration,
    auto_run_dirname,
    next_available_run_dir,
)
from ..subprocess_helpers import build_mflux_env, format_cmd
from .generate import (
    _build_iterations,
    _check_prompt_style_compat,
    _estimate_one_seconds,
    _exit_code,
    _format_duration,
    _load_backend_and_token,
    _open_results,
    _preflight_resources,
    _print_batch_summary,
    _resolve_output_layout,
    _resolve_styles_list,
    _run_one_iteration,
)

__all__ = ["cmd_batch"]


def _confirm_dir_batch(
    *,
    input_paths: list[Path],
    styles_list: list[str],
    directory: Path,
    output_root: Path,
    one_eta_seconds: int | None,
) -> bool:
    """Confirm gate for ``imgen batch``: N×M variant of the v0.2.3
    single-input gate.

    Shows the input directory, the discovered file count (with first
    few names + truncation when N is large — keeps the prompt readable
    at N=50+), the M styles, the output root, and a total-time ETA
    derived from recent successful mflux runs of the same backend/quant.

    Cancellation paths (``n``, empty, Ctrl-C, EOF) return False; caller
    prints a 'nothing generated' line.
    """
    n_inputs = len(input_paths)
    n_styles = len(styles_list)
    total = n_inputs * n_styles
    if n_inputs <= 5:
        input_names = ", ".join(p.name for p in input_paths)
    else:
        head = ", ".join(p.name for p in input_paths[:3])
        input_names = f"{head}, … (+{n_inputs - 3} more)"
    print()
    info(f"About to generate {total} images "
         f"({n_inputs} input{'s' if n_inputs != 1 else ''} × "
         f"{n_styles} style{'s' if n_styles != 1 else ''}):")
    print(f"   {C.DIM}from:{C.END}    {directory}")
    print(f"   {C.DIM}inputs:{C.END}  {input_names}")
    print(f"   {C.DIM}styles:{C.END}  {', '.join(styles_list)}")
    print(f"   {C.DIM}output:{C.END}  {output_root}")
    if one_eta_seconds is not None:
        total_time = _format_duration(one_eta_seconds * total)
        per_image = _format_duration(one_eta_seconds)
        print(f"   {C.DIM}eta:{C.END}     {total_time} total "
              f"({per_image} per image, ±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def cmd_batch(args) -> int:
    """`imgen batch <dir>` — N inputs × M styles into one run folder.

    Pipeline mirrors :func:`~imgen.commands.generate.cmd_generate` but
    with an outer loop over discovered inputs:

    1. Validate directory + collect supported images (non-recursive).
    2. Reject any input-stem collisions (would overwrite under flat
       output layout).
    3. Resolve styles, prompt, output run-dir, backend, token, seed.
    4. Open a HEIC cache (TemporaryDirectory, auto-cleaned).
    5. Per-input: convert HEIC → JPEG if needed, detect resolution,
       build that input's M Iterations.
    6. Preflight resources against the heaviest quant in the full grid.
    7. Confirm gate (unless --yes).
    8. mkdir run_dir, open BatchLogger.
    9. For each input: section_start → run M iterations with global
       indices → section_end with per-input ok/fail counts.
    10. Open Finder, print summary, return mapped exit code.
    """
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Discover inputs (non-recursive). discover_inputs handles
    # non-existent / not-a-dir via die(2) itself; we only need to die on
    # "0 supported images" — keeping the helper reusable for callers
    # that may treat empty as non-fatal (future doctor / preflight).
    directory = Path(args.directory).expanduser().resolve()
    input_paths = discover_inputs(directory)
    if not input_paths:
        die(f"0 supported images in {directory}",
            code=2,
            hint="Supported: jpg/jpeg/png/webp/heic/heif/bmp/tif/tiff/gif "
                 "(non-recursive; dotfiles skipped).")

    # 2) Stem-collision preflight — flat output layout
    # `<run_dir>/<stem>-<style>.png` would silently overwrite otherwise.
    check_input_stems(input_paths)

    # 3) Styles + prompt mutex (reused from generate.py).
    styles_list = _resolve_styles_list(args, merged_defaults)

    try:
        effective_custom_prompt = resolve_prompt(
            custom_prompt=args.custom_prompt,
            prompt_file=getattr(args, "prompt_file", None),
        )
    except PromptInputError as e:
        die(str(e), code=2)

    _check_prompt_style_compat(styles_list, effective_custom_prompt)

    if args.scope and effective_custom_prompt:
        warn(f"--scope={args.scope} ignored when using a custom prompt "
             "(--custom-prompt / --prompt-file)")

    # 4) Output layout — batch never has --output, so this branch always
    # returns (None, run_dir).
    _explicit_output, run_dir = _resolve_output_layout(args, config_output_dir)
    assert run_dir is not None, (
        "batch never uses --output FILE; _resolve_output_layout must "
        "return run_dir for this path"
    )

    # 5) Backend, token, binary (shared across the whole grid).
    backend, be, token, binary = _load_backend_and_token(args)

    # 6) Single seed for the whole batch so the same noise pattern
    # applies to every (input, style) — fair side-by-side preset
    # comparison.
    seed = (args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), "big"))

    # 7) HEIC pre-conversion cache. TemporaryDirectory wipes the
    # converted JPEGs on exit (they can contain identifiable subject
    # matter; no value beyond this run). prefix scoped to imgen so
    # `/tmp/imgen-heic-*` is greppable.
    fixed_dims = (
        (args.width, args.height)
        if args.width and args.height else None
    )
    with tempfile.TemporaryDirectory(prefix="imgen-heic-") as cache_str:
        cache_dir = Path(cache_str)

        # 8) Per-input: resolve to mflux-readable path, detect resolution,
        # build that input's M iterations. Keep per-input groups around
        # so the run loop can wrap each in a log section + assign per-
        # input width/height (mixed-aspect dirs would otherwise inherit
        # the first input's dims).
        per_input_iters: list[tuple[Path, Path, int, int, list[Iteration]]] = []
        for input_path in input_paths:
            mflux_input = resolve_to_mflux_input(input_path, cache_dir)
            if fixed_dims is not None:
                width, height = fixed_dims
            else:
                width, height = detect_resolution(
                    mflux_input, preview=args.preview)
            iters = _build_iterations(
                styles_list=styles_list,
                args=args,
                effective_custom_prompt=effective_custom_prompt,
                merged_defaults=merged_defaults,
                be=be,
                binary=binary,
                input_path=mflux_input,
                width=width,
                height=height,
                explicit_output=None,
                run_dir=run_dir,
                seed=seed,
            )
            per_input_iters.append(
                (input_path, mflux_input, width, height, iters)
            )

        all_iters: list[Iteration] = [
            it for _, _, _, _, group in per_input_iters for it in group
        ]
        total_iters = len(all_iters)

        # `imgen batch` always treats itself as a batch (even N=M=1) so
        # the per-batch log + Finder-open + summary UX is consistent.
        # Upgrade is a fresh batch_id (uuid4[:12] — 48 bits, plenty for
        # single-user uniqueness).
        batch_id: str = uuid.uuid4().hex[:12]

        # 9) Dry run — print every cmd that would execute, exit clean.
        # No mflux invocation, no log, no history. Conversion to JPEG
        # already happened above (mflux paths in cmd reference cache);
        # cache is wiped by TemporaryDirectory on context exit.
        if args.dry_run:
            for idx, it in enumerate(all_iters, start=1):
                print(f"Dry run [{idx}/{total_iters}] {it.style_name}:")
                print()
                print(format_cmd(it.cmd))
                print()
            return 0

        # 10) Preflight against the heaviest quant. Done once for the
        # whole N×M grid, not per-input.
        heaviest_quant = max(it.final_quantize for it in all_iters)
        _preflight_resources(
            backend=backend, heaviest_quant=heaviest_quant, force=args.force
        )

        # 11) Confirm gate. --yes skips. ETA hidden if no matching
        # successful history entries (don't fabricate a wild guess).
        if not args.yes:
            one_eta = _estimate_one_seconds(
                load_history(), backend, heaviest_quant, args.preview
            )
            proceed = _confirm_dir_batch(
                input_paths=input_paths,
                styles_list=styles_list,
                directory=directory,
                output_root=run_dir,
                one_eta_seconds=one_eta,
            )
            if not proceed:
                warn("Cancelled — nothing generated.")
                return 0

        # 12) Materialise run_dir now that the user committed.
        run_dir.mkdir(parents=True, exist_ok=True)

        # 13) BatchLogger lifecycle. Construction + write_header inside
        # the try block so an OSError during write_header doesn't leak
        # the lazily-opened fd. Logger held open for the whole batch
        # (persistent-fd design — saves O(N×M) open/close syscalls).
        env = build_mflux_env(token)
        succeeded: list[tuple[str, Path, int]] = []
        failed: list[tuple[str, int, Path]] = []
        logger: BatchLogger | None = None
        try:
            logger = BatchLogger(batch_id)
            logger.write_header(
                input_paths=input_paths,
                styles=styles_list,
                run_dir=run_dir,
                backend=backend,
                quant=heaviest_quant,
                preview=args.preview,
                scope=args.scope,
                seed=seed,
            )

            # 14) Outer loop = inputs; inner loop = styles. Global
            # iteration index is a flat counter incremented in lock-step
            # with the inner loop so any log line is uniquely numberable
            # across the whole N×M batch.
            #
            # The earlier shape `(n-1)*len(styles_list) + m` only worked
            # because `_build_iterations` happens to return exactly one
            # Iteration per style. If that ever changes (per-style skip
            # on incompatible scope, style-filter, etc.), the formula
            # would silently mis-number while tests stayed green —
            # `test_cmd_batch_log_global_iteration_numbering` exercises
            # only the equal-styles case. Flat counter eliminates the
            # latent breakage. (v0.3.0 python review IMP-2.)
            global_idx = 0
            for n, (input_path, _mflux_input, width, height, iters) in \
                    enumerate(per_input_iters, start=1):
                logger.input_section_start(n, len(input_paths), input_path.name)
                ok_before = len(succeeded)
                fail_before = len(failed)
                input_start = _dt.datetime.now()
                # BatchContext.input_path is the ORIGINAL path (not the
                # sips-converted JPEG) so history.input records what the
                # user typed. Iteration.cmd already references the
                # converted path via _build_iterations(input_path=
                # mflux_input). (v0.3.0 design)
                ctx = BatchContext(
                    backend=backend,
                    seed=seed,
                    width=width,
                    height=height,
                    input_path=input_path,
                    effective_custom_prompt=effective_custom_prompt,
                    args=args,
                    batch_id=batch_id,
                    env=env,
                )
                for it in iters:
                    global_idx += 1
                    cont = _run_one_iteration(
                        it=it,
                        idx=global_idx,
                        total=total_iters,
                        is_batch=True,
                        ctx=ctx,
                        logger=logger,
                        succeeded=succeeded,
                        failed=failed,
                    )
                    if not cont:
                        # KeyboardInterrupt mid-iteration: per-input
                        # section is left un-closed in the log (the
                        # CANCELLED iteration marker is enough signal).
                        # Exit 130 propagates through to cli.main.
                        return 130
                input_duration = int(
                    (_dt.datetime.now() - input_start).total_seconds()
                )
                logger.input_section_end(
                    idx_input=n,
                    total_inputs=len(input_paths),
                    name=input_path.name,
                    ok_count=len(succeeded) - ok_before,
                    fail_count=len(failed) - fail_before,
                    duration=input_duration,
                )

            # 15) Open Finder on the run folder.
            _open_results(
                succeeded=succeeded,
                run_dir=run_dir,
                is_batch=True,
                no_open=args.no_open,
            )

            # 16) End-of-batch summary.
            _print_batch_summary(succeeded, failed, total_iters)

            # 17) Exit code (all-ok=0 / all-failed=1 / partial=5).
            return _exit_code(
                is_batch=True, succeeded=succeeded, failed=failed
            )
        finally:
            if logger is not None:
                logger.close()

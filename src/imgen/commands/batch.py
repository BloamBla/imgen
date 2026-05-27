"""`imgen batch <dir>` — apply M styles across every supported image
under ``<dir>``.

v0.3.0's only new subcommand. Designed as a thin orchestrator that
composes shared pipeline helpers from :mod:`imgen.cmd_helpers`:

* :func:`~imgen.cmd_helpers.resolve_styles_list` — pure resolution
  (the ``--output FILE`` mutex is generate-only, kept there).
* :func:`~imgen.cmd_helpers.check_prompt_style_compat`
* :func:`~imgen.cmd_helpers.resolve_output_layout` — returns the
  ``run_dir`` branch (batch has no ``--output FILE`` flag).
* :func:`~imgen.cmd_helpers.load_backend_and_token`
* :func:`~imgen.cmd_helpers.build_iterations` — called once per
  input, the resulting lists are flattened for preflight.
* :func:`~imgen.cmd_helpers.preflight_resources` — guards against
  the heaviest quant across the full N×M grid.
* :func:`~imgen.cmd_helpers.run_one_iteration` — the single
  per-mflux unit, wrapped here in the per-input section + global index.
* :func:`~imgen.cmd_helpers.open_results`,
  :func:`~imgen.cmd_helpers.print_batch_summary`,
  :func:`~imgen.cmd_helpers.exit_code` — closing UX.

Pre-v0.3.1 these were leading-underscore helpers imported from
:mod:`imgen.commands.generate` — the v0.3.0 architect review flagged
the cross-module ``_private`` import smell. v0.3.1 promoted them to
:mod:`imgen.cmd_helpers` and dropped the underscore prefix.

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
    PerInputBatch,
)
from ..cmd_helpers import (
    apply_enhance_results_to_groups,
    build_bare_i2i_iteration,
    build_iterations,
    megapixels_of,
    require_style_or_prompt,
    check_prompt_style_compat,
    emit_gated_repo_hint_if_failed,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    maybe_enhance_for_command,
    open_results,
    preflight_resources,
    print_batch_summary,
    prompt_yes_no,
    resolve_enhance_config,
    resolve_output_layout,
    resolve_styles_list,
    run_one_iteration,
)
from ..subprocess_helpers import build_mflux_env

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
        total_time = format_duration(one_eta_seconds * total)
        per_image = format_duration(one_eta_seconds)
        print(f"   {C.DIM}eta:{C.END}     {total_time} total "
              f"({per_image} per image, ±50%)")
    print()
    return prompt_yes_no()


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

    # v0.7.0: same flux-dev redirect as cmd_generate. Batch is i2i by
    # construction; t2i prompts don't have a discovery directory.
    if args.model == "flux-dev":
        die(
            "--model flux-dev is text-to-image; cmd_batch requires "
            "input photos. Use `imgen draw \"<prompt>\"` for single-shot "
            "t2i (batch t2i over a prompt file is v0.7.x).",
            code=2,
        )

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
    # No mutex check needed: batch has no --output FILE flag (its
    # parser stanza omits it), so the generate-specific
    # _check_output_style_mutex doesn't apply here.
    styles_list = resolve_styles_list(args, merged_defaults)

    try:
        effective_custom_prompt = resolve_prompt(
            custom_prompt=args.custom_prompt,
            prompt_file=getattr(args, "prompt_file", None),
        )
    except PromptInputError as e:
        die(str(e), code=2)

    check_prompt_style_compat(styles_list, effective_custom_prompt)

    # v0.7.13 (gap 8) — symmetric with cmd_generate via shared helper
    # (architect S1 extraction). Single source of truth for the
    # bare-mode migration message — drift-free across both subcommands.
    require_style_or_prompt(styles_list, effective_custom_prompt)

    # v0.3.5: see commands/generate.py for the scope+custom semantics
    # commentary. Same applies here — augmentation = scope on preset
    # base; custom-only = no scope target. No noisy per-batch warn.

    # 4) Output layout — batch never has --output, so this branch always
    # returns (None, run_dir).
    _explicit_output, run_dir = resolve_output_layout(args, config_output_dir)
    assert run_dir is not None, (
        "batch never uses --output FILE; resolve_output_layout must "
        "return run_dir for this path"
    )

    # 5) Backend, token, binary, custom-secret (shared across the whole grid).
    # v0.8.5: binary is unused — Engine.run resolves it internally
    # via VENV_BIN / model.binary post-M-NEW-D. Kept in the tuple
    # shape so load_backend_and_token stays stable for v0.9.
    backend, be, token, _binary, backend_secret = load_backend_and_token(args)

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
        # v0.6.x backlog python IMP-3: shared dedup set so an
        # incompatible CLI --lora (or style-declared LoRA) warns exactly
        # ONCE for the entire N×M batch instead of once per input.
        warned_incompat_loras: set[tuple[str, str]] = set()
        # v0.6.5 (architect IMP-3): per-input shape promoted from a bare
        # 5-tuple to PerInputBatch — named-field access at the unpack +
        # flatten sites below replaces the ``for _, _, _, _, group``
        # underscore soup.
        per_input_iters: list[PerInputBatch] = []
        for input_path in input_paths:
            # sips-failure policy: a CalledProcessError or TimeoutExpired
            # from resolve_to_mflux_input here propagates uncaught, aborting
            # the whole batch. Design doc left "warn-and-skip vs fail-batch"
            # open; v0.3.0 chose fail-batch by inaction. Rationale: a
            # broken HEIC in the user's dir is rare AND surfacing it
            # early stops the user from waiting through N-1 mflux runs
            # before discovering the issue. Switch to per-input warn+skip
            # if colleague demand surfaces. (v0.3.0 architect NIT-7.)
            mflux_input = resolve_to_mflux_input(input_path, cache_dir)
            if fixed_dims is not None:
                width, height = fixed_dims
            else:
                width, height = detect_resolution(
                    mflux_input, preview=args.preview)
            # v0.7.13 (gap 8): bare mode per input — empty styles_list
            # (validated at step 3b above) + non-None custom_prompt →
            # one bare iteration per input, no preset baggage.
            if styles_list:
                iters = build_iterations(
                    styles_list=styles_list,
                    args=args,
                    effective_custom_prompt=effective_custom_prompt,
                    merged_defaults=merged_defaults,
                    be=be,
                    input_path=mflux_input,
                    width=width,
                    height=height,
                    explicit_output=None,
                    run_dir=run_dir,
                    seed=seed,
                    # v0.6.x backlog python IMP-3: share the dedup set
                    # across the N inputs so an incompatible CLI --lora
                    # warns ONCE for the whole batch instead of N times.
                    warned_incompat_loras=warned_incompat_loras,
                )
            else:
                assert effective_custom_prompt is not None
                iters = [build_bare_i2i_iteration(
                    args=args,
                    input_path=mflux_input,
                    prompt=effective_custom_prompt,
                    merged_defaults=merged_defaults,
                    be=be,
                    width=width,
                    height=height,
                    explicit_output=None,
                    run_dir=run_dir,
                    seed=seed,
                )]
            per_input_iters.append(PerInputBatch(
                input_path=input_path,
                mflux_input=mflux_input,
                width=width,
                height=height,
                iters=tuple(iters),
            ))

        all_iters: list[Iteration] = [
            it for pib in per_input_iters for it in pib.iters
        ]
        total_iters = len(all_iters)

        # 8b) v0.5 — optional LLM prompt enhancer.
        # Runs ONCE for the whole N×M batch (single mlx_lm.load
        # amortised across all prompts; the orchestrator handles per-
        # prompt skip + all-or-nothing runner fallback). Order matters:
        # this is BEFORE dry-run so the displayed cmd matches what mflux
        # would actually receive; BEFORE confirm gate so the user knows
        # enhancement happened before they say yes to a long batch.
        # The pre-enhance prompt is captured inside each EnhanceResult's
        # original_prompt field (no parallel list needed).
        eff_enhance = resolve_enhance_config(
            cli_enable=getattr(args, "enhance", None),
            cli_model=getattr(args, "enhance_model", None),
            cli_temperature=getattr(args, "enhance_temperature", None),
            config_enhance=getattr(args, "imgen_config_enhance", {}),
        )
        enhance_results, enhance_model = maybe_enhance_for_command(
            eff_enhance=eff_enhance,
            backend_obj=be,
            iterations=all_iters,
        )
        # Splice enhanced prompts back into the per-input shape via a
        # single call (v0.6.4 v0.5 architect IMP #2 — used to be a
        # sliding-cursor block inline here; the cursor moved into
        # apply_enhance_results_to_groups where it's encapsulated +
        # alignment-asserted). v0.7.4 retired the Protocol-typed
        # generalisation as didn't-earn — helper now strictly takes
        # list[PerInputBatch]. all_iters stays flat for downstream
        # dry-run / preflight / confirm gate consumption.
        per_input_iters = apply_enhance_results_to_groups(
            per_input_iters, enhance_results,
        )
        all_iters = [
            it for pib in per_input_iters for it in pib.iters
        ]

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
            from ..engine_dispatch import iteration_dryrun_display
            for idx, it in enumerate(all_iters, start=1):
                print(f"Dry run [{idx}/{total_iters}] {it.style_name}:")
                print()
                print(iteration_dryrun_display(it))
                print()
            return 0

        # 10) Preflight against the heaviest quant + largest output
        # resolution. v0.7.14 (gap 6): max_megapixels is per-input
        # max because batch may carry mixed-aspect inputs (per-input
        # detect_resolution at step 8). Each PerInputBatch carries
        # its own width/height — take the largest area for the
        # conservative estimate.
        heaviest_quant = max(it.final_quantize for it in all_iters)
        max_megapixels = max(
            megapixels_of(pi.width, pi.height) for pi in per_input_iters
        )
        preflight_resources(
            model=backend, heaviest_quant=heaviest_quant,
            force=args.force, max_megapixels=max_megapixels,
        )

        # 11) Confirm gate. --yes skips. ETA hidden if no matching
        # successful history entries (don't fabricate a wild guess).
        if not args.yes:
            one_eta = estimate_one_seconds(
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
        env = build_mflux_env(token=token, backend_secret=backend_secret)
        succeeded: list[tuple[str, Path, int]] = []
        failed: list[tuple[str, int, Path]] = []
        logger: BatchLogger | None = None
        try:
            logger = BatchLogger(batch_id)
            logger.write_header(
                input_paths=input_paths,
                styles=styles_list,
                run_dir=run_dir,
                model=backend,
                quant=heaviest_quant,
                preview=args.preview,
                # v0.6.5 architect IMP-A: scope is i2i-only — mirror
                # the _resolve_iteration_prompt getattr defence so the
                # future imgen draw subparser, which omits --scope,
                # passes through here without an AttributeError. cmd_batch
                # itself still binds args.scope unconditionally because
                # batch is i2i-only by definition; this is the call into
                # the shared logger surface that's also visited by draw.
                scope=getattr(args, "scope", None),
                seed=seed,
            )

            # 14) Outer loop = inputs; inner loop = styles. Global
            # iteration index is a flat counter incremented in lock-step
            # with the inner loop so any log line is uniquely numberable
            # across the whole N×M batch.
            #
            # The earlier shape `(n-1)*len(styles_list) + m` only worked
            # because `build_iterations` happens to return exactly one
            # Iteration per style. If that ever changes (per-style skip
            # on incompatible scope, style-filter, etc.), the formula
            # would silently mis-number while tests stayed green —
            # `test_cmd_batch_log_global_iteration_numbering` exercises
            # only the equal-styles case. Flat counter eliminates the
            # latent breakage. (v0.3.0 python review IMP-2.)
            global_idx = 0
            for n, pib in enumerate(per_input_iters, start=1):
                input_path = pib.input_path
                width = pib.width
                height = pib.height
                iters = pib.iters
                logger.input_section_start(n, len(input_paths), input_path.name)
                ok_before = len(succeeded)
                fail_before = len(failed)
                input_start = _dt.datetime.now()
                # BatchContext.input_path is the ORIGINAL path (not the
                # sips-converted JPEG) so history.input records what the
                # user typed. Iteration.params.input_path already
                # references the converted path via build_iterations(
                # input_path=mflux_input). (v0.3.0 design; v0.8.4 M-NEW-D
                # — MfluxEngine.run reads params at dispatch time.)
                ctx = BatchContext(
                    model=backend,
                    seed=seed,
                    width=width,
                    height=height,
                    input_path=input_path,
                    effective_custom_prompt=effective_custom_prompt,
                    args=args,
                    batch_id=batch_id,
                    env=env,
                    command="batch",
                )
                for it in iters:
                    global_idx += 1
                    # v0.5: thread enhance metadata. global_idx is 1-based;
                    # enhance_results is 0-based and aligned with the flat
                    # all_iters list above. The pre-enhance prompt is
                    # carried inside enhance_results[i].original_prompt —
                    # no parallel list to keep in sync.
                    cont = run_one_iteration(
                        it=it,
                        idx=global_idx,
                        total=total_iters,
                        is_batch=True,
                        ctx=ctx,
                        logger=logger,
                        succeeded=succeeded,
                        failed=failed,
                        enhance_result=enhance_results[global_idx - 1],
                        enhance_model=enhance_model,
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
            open_results(
                succeeded=succeeded,
                run_dir=run_dir,
                is_batch=True,
                no_open=args.no_open,
            )

            # v0.7.1: friendly HF license-grant hint when mflux failed
            # AND backend declares a gated repo. Mirror of cmd_draw's
            # v0.7.0 post-failure hint.
            emit_gated_repo_hint_if_failed(failed=failed, backend_obj=be)

            # 16) End-of-batch summary.
            print_batch_summary(succeeded, failed, total_iters)

            # 17) Exit code (all-ok=0 / all-failed=1 / partial=5).
            return exit_code(
                is_batch=True, succeeded=succeeded, failed=failed
            )
        finally:
            if logger is not None:
                logger.close()

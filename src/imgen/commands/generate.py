"""`imgen generate` (default subcommand) — the actual photo→style transfer.

Pipeline: validate input → build prompt → resolve params (CLI > preset >
preview > default) → detect resolution → resolve output path → load token
→ build mflux command → preflight (RAM/disk/battery/parallel-mflux) →
spawn mflux via the stderr-redacting wrapper → write history entry →
auto-open result in Preview.

Most of the pipeline pieces are shared with ``commands/batch.py`` (which
adds an outer loop over N inputs); they live in ``imgen.cmd_helpers`` so
both call sites compose the same primitives. Everything generate-specific
stays here:

* :func:`_confirm_batch` — 1×M confirm gate UI (batch has its own
  ``_confirm_dir_batch`` with N×M counts).
* :func:`_validate_input_path` — single-file validation (batch uses
  ``discover_inputs`` for its directory input).
* :func:`_check_output_style_mutex` — generate-only mutex (``--output
  FILE`` writes to one path, multi-style needs M files → a dir; batch
  has no ``--output`` flag so the check would be a no-op there).
* :func:`cmd_generate` — the orchestrator.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from ..cmd_helpers import (
    build_iterations,
    check_prompt_style_compat,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    open_results,
    preflight_resources,
    print_batch_summary,
    resolve_output_layout,
    resolve_styles_list,
    run_one_iteration,
)
from ..colors import C, die, info, warn
from ..defaults import DEFAULTS
from ..history import load_history
from ..images import detect_resolution
from ..inputs import resolve_to_mflux_input
from ..prompt_input import PromptInputError, resolve_prompt
from ..runs import BatchContext, BatchLogger, Iteration
from ..subprocess_helpers import build_mflux_env, format_cmd

__all__ = ["cmd_generate"]


def _confirm_batch(
    iterations: list[Iteration],
    input_name: str,
    output_root: Path,
    one_eta_seconds: int | None,
) -> bool:
    """Print summary + interactive y/N gate. Returns True to proceed.

    EOF (piped stdin closed), Ctrl-C, and any answer other than `y`/`yes`
    return False. Caller is responsible for printing a 'cancelled' line
    and exiting clean (no folder created, no history entries written).

    This is the generate-specific 1×M variant. ``cmd_batch`` has its
    own ``_confirm_dir_batch`` covering the N×M case with directory
    listing + N inputs / M styles counts.
    """
    n = len(iterations)
    style_names = [it.style_name for it in iterations]
    print()
    info(f"About to generate {n} images:")
    print(f"   {C.DIM}input:{C.END}   {input_name}")
    print(f"   {C.DIM}styles:{C.END}  {', '.join(style_names)}")
    print(f"   {C.DIM}output:{C.END}  {output_root}")
    if one_eta_seconds is not None:
        total = format_duration(one_eta_seconds * n)
        per_image = format_duration(one_eta_seconds)
        print(f"   {C.DIM}eta:{C.END}     {total} total "
              f"({per_image} per image, ±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def _check_output_style_mutex(args, styles_list: list[str]) -> None:
    """Reject ``--output FILE`` + multi-style upfront.

    ``--output FILE`` writes to a single path; multi-style produces M
    files which needs a directory. Catch this before any subprocess
    spawns so the user gets a clean hint.

    Generate-only — ``imgen batch`` has no ``--output FILE`` flag (its
    parser stanza omits it) so this mutex doesn't apply there. Pre-v0.3.1
    the check lived inside ``resolve_styles_list`` with a
    ``getattr(args, "output", None)`` guard for batch's missing
    attribute — correct but surprising for batch readers; the split
    here makes the generate-only nature explicit. (v0.3.0 architect
    NIT-4 / NIT-6.)
    """
    if args.output and len(styles_list) > 1:
        die(f"--output FILE writes to one path; multi-style "
            f"(--style {','.join(styles_list)} → {len(styles_list)} files) "
            "needs a directory.",
            code=2,
            hint="Drop --output, or use --output-dir PATH instead.")


def _validate_input_path(image_arg: str) -> Path:
    """Resolve, expand ~, and verify the user-supplied input image.

    Returns the absolute resolved Path. Exits with code 2 on missing
    file or non-file (directory, special device). Resolution happens
    here once per invocation so downstream mflux subprocess and history
    entries record absolute paths regardless of cwd at call time.
    """
    input_path = Path(image_arg).expanduser().resolve()
    if not input_path.exists():
        die(f"Image not found: {input_path}",
            code=2,
            hint="Check the path. Use absolute path if unsure.")
    if not input_path.is_file():
        die(f"Not a file: {input_path}", code=2)
    return input_path


def cmd_generate(args) -> int:
    # Config-aware defaults (config.toml [defaults] merged over module
    # DEFAULTS, populated in cli.main). For old callers that bypass
    # cli.main — e.g. direct test invocation — fall back to module DEFAULTS.
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Validate input ONCE for all iterations.
    input_path = _validate_input_path(args.image)

    # 2) Resolve --style → list[str] (pure), then enforce the
    # generate-only --output + multi-style mutex.
    styles_list = resolve_styles_list(args, merged_defaults)
    _check_output_style_mutex(args, styles_list)

    # 3) Resolve effective custom prompt — could be argv text, stdin
    # (--custom-prompt -), or a file (--prompt-file). File/stdin paths
    # keep prompt text out of `ps auxww`.
    try:
        effective_custom_prompt = resolve_prompt(
            custom_prompt=args.custom_prompt,
            prompt_file=getattr(args, "prompt_file", None),
        )
    except PromptInputError as e:
        die(str(e), code=2)

    # 3a) Pre-flight mutex per style (multi-style: ALL items must agree).
    check_prompt_style_compat(styles_list, effective_custom_prompt)

    # v0.3.5: scope semantics with --custom-prompt — applies to the
    # PRESET portion of an augmented prompt, NOT to the user's added
    # text (build_iterations passes the user text through verbatim).
    # In custom-only paths (no explicit --style, or param-only style)
    # there's no preset prompt to scope, so scope is effectively a
    # no-op there — but warning every time would be noisy and we now
    # document this behaviour in --scope --help instead.

    # 3c) HEIC pre-conversion (v0.3.0 bonus — also fixes the v0.2.x bug
    # where `imgen generate vacation.heic` died with a cryptic mflux
    # PIL error). resolve_to_mflux_input returns the input unchanged
    # when it isn't HEIC, so the only non-HEIC overhead is one mkdir +
    # one rmdir of an empty /tmp/imgen-heic-* — trivial vs the 30s-3min
    # mflux runtime. ``BatchContext.input_path`` stays the ORIGINAL so
    # history.input records what the user typed; ``Iteration.cmd``
    # references the converted JPEG via ``build_iterations(input_path=
    # mflux_input)``.
    with tempfile.TemporaryDirectory(prefix="imgen-heic-") as cache_str:
        mflux_input = resolve_to_mflux_input(input_path, Path(cache_str))

        # 4) Resolution (read from the JPEG — mflux's PIL can't open HEIC).
        if args.width and args.height:
            width, height = args.width, args.height
        else:
            width, height = detect_resolution(mflux_input, preview=args.preview)

        # 5) Output root + run_dir (one folder for all M iterations).
        explicit_output, run_dir = resolve_output_layout(args, config_output_dir)

        # 6) Backend, token, binary, custom-secret (same for all M).
        backend, be, token, binary, backend_secret = load_backend_and_token(args)

        # 7) Seed — one seed for the whole invocation so multi-style runs use
        # the same noise pattern (only style differs → fair preset comparison).
        seed = (args.seed if args.seed is not None
                else int.from_bytes(os.urandom(4), "big"))

        # 8) Pre-resolve each iteration's params + cmd so dry-run can show
        # all M and we can preflight resources against the heaviest one.
        iterations = build_iterations(
            styles_list=styles_list,
            args=args,
            effective_custom_prompt=effective_custom_prompt,
            merged_defaults=merged_defaults,
            be=be,
            binary=binary,
            input_path=mflux_input,
            width=width,
            height=height,
            explicit_output=explicit_output,
            run_dir=run_dir,
            seed=seed,
        )

        # is_batch threshold: "are we doing more than one image in this
        # invocation?" v0.2.x → len(iterations) >= 2 (single input × M
        # styles); v0.3.0 → N*M >= 2 (N inputs × M styles). Renamed from
        # `multi` in v0.2.4 (architect item F1) so the upstream definition
        # is the one place that changes when batch.py lands.
        is_batch = len(iterations) >= 2
        # batch_id stamps every history entry from this invocation when
        # is_batch is True, making `imgen history --batch <id>` (v0.3.0+)
        # trivial. Null for single-image runs to preserve v0.2.x history shape.
        # 12 hex chars = 48 bits — collision probability is astronomical at
        # single-user scale (one Mac, one human). Keeps log filenames short
        # and readable vs the 32-char full uuid4. ULID would give lex-
        # sortable IDs, but auto_run_dirname() already provides the
        # chronological sort via run-folder names; batch_id only needs to
        # be unique. (architect F4 from v0.2.3 review)
        batch_id: str | None = uuid.uuid4().hex[:12] if is_batch else None

        # 9) Dry run — show every M cmd, skip resource checks + history.
        if args.dry_run:
            for it in iterations:
                from ..colors import step
                step(f"Dry run — would execute ({it.style_name}):")
                print()
                print(format_cmd(it.cmd))
                print()
            return 0

        # 10) Resource preflight — check against the heaviest quant in the
        # batch (all iterations share a backend + quantize today since neither
        # is per-style; verifying the first is sufficient and future-proof
        # via max() if per-style quantize is ever added).
        heaviest_quant = max(it.final_quantize for it in iterations)
        preflight_resources(
            backend=backend, heaviest_quant=heaviest_quant, force=args.force
        )

        # 11) Confirm gate (multi-style only — single-style keeps v0.2.x's
        # zero-prompt UX). --yes skips. Fires AFTER preflight so we never
        # ask the user to confirm a batch we know we can't run anyway.
        if is_batch and not args.yes:
            one_eta = estimate_one_seconds(
                load_history(), backend, heaviest_quant, args.preview
            )
            proceed = _confirm_batch(
                iterations=iterations,
                input_name=input_path.name,
                output_root=run_dir if run_dir is not None else explicit_output.parent,
                one_eta_seconds=one_eta,
            )
            if not proceed:
                warn("Cancelled — nothing generated.")
                return 0

        # 12) mkdir output dir now that we'll actually run.
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            explicit_output.parent.mkdir(parents=True, exist_ok=True)

        # 13) Per-batch log (multi-style only). BatchLogger owns the lifecycle:
        # header here, iteration start/end/cancelled markers inside
        # run_one_iteration, mflux stderr-tee via borrow_fd() inside
        # run_with_stderr_redaction, retention via runs.prune_old_batch_logs
        # called from imgen clean. Held open for the whole batch (v0.2.5
        # IMP-4 — saves ~200 open/close syscalls at N×M=50); closed in the
        # try/finally below regardless of how cmd_generate exits.
        #
        # Construction + write_header live INSIDE the try block (v0.2.5
        # review NIT) so an OSError during the first write_header (rare
        # — disk full at exactly that moment) doesn't leak the fd that
        # _ensure_open() opened mid-header.
        logger: BatchLogger | None = None
        try:
            if is_batch:
                logger = BatchLogger(batch_id)
                logger.write_header(
                    input_paths=[input_path],
                    styles=[it.style_name for it in iterations],
                    run_dir=run_dir,
                    backend=backend,
                    quant=heaviest_quant,
                    preview=args.preview,
                    scope=args.scope,
                    seed=seed,
                )

            # 14) Minimal env (don't forward random secrets from parent shell).
            # Shared with cmd_batch via subprocess_helpers.build_mflux_env —
            # single source of truth for the allow-list (IMP-5 from v0.3.0
            # review; was duplicated between batch.py and here).
            env = build_mflux_env(token=token, backend_secret=backend_secret)

            # 15) The loop. Failures don't break the batch — log + continue,
            # surface the summary at the end. Single-style retains v0.2.x
            # exit semantics (mflux return code passes through).
            succeeded: list[tuple[str, Path, int]] = []
            failed: list[tuple[str, int, Path]] = []
            total = len(iterations)

            # Bundle the 9 batch-invariant args into a single BatchContext so
            # run_one_iteration's signature stays compact — and v0.3.0's
            # nested N×M loop in commands/batch.py can thread one value
            # through the inner loop instead of nine. (architect IMP-3 from
            # v0.2.4 review). input_path here is the ORIGINAL (HEIC or
            # otherwise) so history.input records what the user typed —
            # Iteration.cmd already points mflux at the sips-converted
            # JPEG via mflux_input above.
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

            for idx, it in enumerate(iterations, start=1):
                cont = run_one_iteration(
                    it=it,
                    idx=idx,
                    total=total,
                    is_batch=is_batch,
                    ctx=ctx,
                    logger=logger,
                    succeeded=succeeded,
                    failed=failed,
                )
                if not cont:
                    return 130

            # 16) Open in Preview (single-style) or Finder (multi-style);
            # silent no-op on --no-open or when `open` is missing.
            open_results(
                succeeded=succeeded,
                run_dir=run_dir,
                is_batch=is_batch,
                no_open=args.no_open,
            )

            # 17) End-of-batch summary (only for multi-style — single-style
            # keeps the v0.2.x lean output).
            if is_batch:
                print_batch_summary(succeeded, failed, total)

            # 18) Exit code (single-style passthrough vs multi-style 0/1/5).
            return exit_code(
                is_batch=is_batch, succeeded=succeeded, failed=failed
            )
        finally:
            if logger is not None:
                logger.close()

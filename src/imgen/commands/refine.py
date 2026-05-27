"""`imgen refine <input>` — Hires-Fix path (v0.7.5+).

Takes an existing 1024² image (typically the winner from an
`imgen draw "..." --num-iterations N --preview` explore run) and
re-renders it through FLUX.2-klein-9B (or FLUX-Kontext) at a higher
resolution with i2i refine. Composition + subject preserved;
detail / sharpness / texture quality go up.

Closes the canonical explore→refine pipeline:

  1. ``imgen draw "samurai" --num-iterations 5 --preview``  (~15 min,
     5 variants at 1024²)
  2. Pick the winner via Finder
  3. ``imgen refine <winner.png>``  (~10-20 min, polished 1536²/2048²)

Self-contained orchestrator (mirror of cmd_draw shape, NOT a
delegation to cmd_generate) — refine has its OWN prompt path
(default refine prompt or user override), NO style machinery, NO
scope substitution, NO trigger-word prepending against built-in
style LoRAs. Negative prompt is empty (style-inherited negatives
like pixar's "deformed, blurry..." would actively fight the refine
goal of preserving the input).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..colors import C, die, info, warn
from ..defaults import DEFAULTS
from ..history import load_history
from ..inputs import resolve_single_input_path
from ..runs import BatchContext
from ..cmd_helpers import (
    build_refine_iteration,
    emit_gated_repo_hint_if_failed,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    open_results,
    megapixels_of,
    preflight_resources,
    prompt_yes_no,
    resolve_output_layout,
    run_one_iteration,
)
from ..subprocess_helpers import build_mflux_env

__all__ = ["cmd_refine"]


# v0.7.5 NIT #4: lives here (runtime constant private to refine) instead
# of parser.py, where it forced a cross-module private-name import.
_DEFAULT_REFINE_PROMPT = (
    "Same scene and composition. Refine with sharper detail, "
    "ultra-detailed textures, professional photography quality, "
    "preserve subject identity, no artifacts, 8K clarity."
)


def _round_to_multiple_of_16(n: int) -> int:
    """FLUX architectures require width/height to be multiples of 16.
    Round to the NEAREST multiple — input 1024 stays 1024, input 1500
    rounds to 1504, etc.
    """
    return ((n + 8) // 16) * 16


def _read_image_dimensions(path: Path) -> tuple[int, int]:
    """Read raw (width, height) of an image via PIL. Differs from
    :func:`imgen.images.detect_resolution` which snaps to a smart
    table of model-preferred resolutions — refine wants the LITERAL
    input dimensions so the ``--scale`` multiplier behaves
    predictably.

    EXIF orientation is honoured so a portrait-shot photo returns
    its rendered dimensions rather than the camera-sensor orientation.

    Errors:
    - Pillow missing → die(code=2). Bootstrap should have installed
      it; surface a clear hint rather than a bare ImportError.
    - File unreadable / not a recognised image → die(code=2). Narrow
      to ``(OSError, UnidentifiedImageError)`` so unrelated
      programmer errors (AttributeError on a PIL version mismatch,
      KeyboardInterrupt, etc.) propagate rather than being silently
      reformatted as "couldn't read".
    """
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as e:
        die(
            f"refine: Pillow (PIL) not installed — required to read "
            f"image dimensions. Run bootstrap.sh to install: {e}",
            code=2,
        )
    try:
        with Image.open(path) as im:
            # Sec #S1: read .size BEFORE the with-block exits. After
            # exif_transpose, `im` may be a new Image OR the same
            # one (when no EXIF orientation tag is present) — and
            # PIL doesn't guarantee .size on a closed Image. Snap
            # the tuple now while the underlying file handle is
            # still open.
            transposed = ImageOps.exif_transpose(im)
            w, h = transposed.size
            return w, h
    except (OSError, UnidentifiedImageError) as e:
        die(
            f"refine: couldn't read image dimensions for {path}: {e}",
            code=2,
        )


def _resolve_target_dimensions(
    args, in_w: int, in_h: int,
) -> tuple[int, int]:
    """Compute (output_w, output_h) from --scale OR --width/--height.

    Mutex: --scale + --width/--height rejected. Both --width and
    --height required together when used as explicit dims. Default
    (neither passed) = --scale 1.5.

    Pure: dims come in via ``(in_w, in_h)`` so the function does NO
    PIL I/O. cmd_refine reads dims ONCE upfront and threads them
    here AND into _confirm_refine, killing the v0.7.5 double-open
    (TOCTOU smell flagged by python-reviewer IMPORTANT #2).
    """
    scale_set = args.scale is not None
    dims_set = args.width is not None or args.height is not None

    if scale_set and dims_set:
        die(
            "--scale and --width/--height are mutually exclusive — pass "
            "ONE of them, not both",
            code=2,
        )

    if dims_set and not (args.width is not None and args.height is not None):
        die(
            "--width and --height must be set together when used as "
            "explicit dimensions (or omit both and use --scale).",
            code=2,
        )

    if dims_set:
        return (
            _round_to_multiple_of_16(args.width),
            _round_to_multiple_of_16(args.height),
        )

    scale = args.scale if scale_set else 1.5
    return (
        _round_to_multiple_of_16(int(in_w * scale)),
        _round_to_multiple_of_16(int(in_h * scale)),
    )


def _confirm_refine(
    *,
    input_path: Path,
    in_w: int,
    in_h: int,
    target_w: int,
    target_h: int,
    output: Path,
    eta_seconds: int | None,
) -> bool:
    """Confirm gate for `imgen refine`. Shows input → output dims,
    output path, and optional ETA. Input dims threaded in by
    cmd_refine — no second PIL open here."""
    info("About to refine 1 image:")
    print(f"   {C.DIM}input: {C.END} {input_path.name} ({in_w}×{in_h})")
    print(f"   {C.DIM}output:{C.END} {output.name} ({target_w}×{target_h})")
    if eta_seconds is not None:
        print(f"   {C.DIM}eta:{C.END}    {format_duration(eta_seconds)} (±50%)")
    print()
    return prompt_yes_no()


def cmd_refine(args) -> int:
    """`imgen refine <input>` — Hires-Fix orchestrator.

    Pipeline mirrors cmd_draw structure:
      1. Validate input file.
      2. Resolve target dimensions (--scale OR --width/--height).
      3. Resolve prompt (default refine prompt or --prompt override).
      4. Load backend + token (default flux2-klein-edit-9b).
      5. Resolve output layout (--output PATH or run-dir + slug).
      6. Build single Iteration via build_refine_iteration.
      7. Dry-run path: print cmd + exit.
      8. Preflight + confirm gate.
      9. Run one iteration, open result, exit.

    NO enhancer path (refine deliberately doesn't inflate the prompt
    — inflating risks shifting AWAY from input composition, which
    defeats Hires-Fix). User wanting an enhanced refine prompt can
    pass --prompt directly.
    """
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Validate input (v0.7.7 Sec #S2: shared helper now also
    # enforces control-byte hygiene on the filename, parity with
    # the batch path's discover_inputs filter).
    input_path = resolve_single_input_path(args.input, subcommand="refine")

    # 2) Read input dims ONCE — threaded into both
    # _resolve_target_dimensions (scale path) and _confirm_refine
    # (display). Single PIL open per invocation, no TOCTOU window
    # between target-resolve and confirm.
    in_w, in_h = _read_image_dimensions(input_path)
    target_w, target_h = _resolve_target_dimensions(args, in_w, in_h)

    # 3) Prompt resolution (refine has a baked-in default; no
    # --prompt-file / stdin path — refine prompts are short and
    # deterministic, no need for the secret-from-ps machinery).
    # v0.7.7 fix: use `is not None` so an explicit `--prompt ""`
    # passes through as the empty string (deliberate user choice
    # per _clean_prompt_arg's docstring) rather than silently
    # substituting the default via `or` truthy-check.
    prompt = args.prompt if args.prompt is not None else _DEFAULT_REFINE_PROMPT

    # 4) Backend + token
    # v0.8.5: binary unused — Engine.run resolves it internally
    # via VENV_BIN / model.binary post-M-NEW-D.
    backend, be, token, _binary, backend_secret = load_backend_and_token(args)

    # v0.8.0 commit 7 (§M): the pre-commit-7 hardcoded
    # ``if backend == "flux2-klein-edit-9b": args.guidance = 1.0`` pin
    # was REMOVED here. mflux-generate-flux2-edit's `--guidance 1.0`
    # contract is now enforced at the per-Model level via
    # ``Model.min_guidance = Model.max_guidance = 1.0`` for the
    # flux2-klein-edit-9b row; MfluxEngine.validate (called from
    # ``validate_engine_params_or_die`` inside the iteration builder)
    # rejects out-of-range guidance with a clean exit-2. The new
    # approach scales: any future FLUX.2 variant inherits the same
    # enforcement without per-binary cmd_* edits.

    # 5) Output layout
    explicit_output, run_dir = resolve_output_layout(args, config_output_dir)

    # 6) Seed
    seed = (
        args.seed if args.seed is not None
        else int.from_bytes(os.urandom(4), "big")
    )

    # 7) Build the single iteration
    iteration = build_refine_iteration(
        args=args,
        input_path=input_path,
        prompt=prompt,
        merged_defaults=merged_defaults,
        be=be,
        width=target_w,
        height=target_h,
        explicit_output=explicit_output,
        run_dir=run_dir,
        seed=seed,
    )

    # 8) Dry-run
    if args.dry_run:
        print(f"Refine target: {target_w}×{target_h}")
        # v0.7.7 Arch #D: mark baked-in default vs user override so
        # the dry-run viewer knows whether they're seeing the value
        # they typed or the implicit fallback.
        if args.prompt is None:
            info("Using default refine prompt (override with --prompt).")
        print(f"   {C.DIM}prompt:{C.END} {prompt[:80]}"
              f"{'...' if len(prompt) > 80 else ''}")
        print()
        print(f"Dry run — would execute (refine):")
        print()
        from ..engine_dispatch import iteration_dryrun_display
        print(iteration_dryrun_display(iteration))
        print()
        return 0

    # 9) Preflight. v0.7.14 (gap 6): pass max_megapixels so refine at
    # 1024² no longer hits the 2K²-calibrated ceiling. ``target_w`` and
    # ``target_h`` are in scope from step 2 (resolved from --scale or
    # explicit --width/--height) — refine has exactly one iteration so
    # one value covers it. (v0.7.17 fix: pre-v0.7.17 referenced bare
    # `width, height` here → NameError on every non-dry-run refine.)
    heaviest_quant = iteration.final_quantize
    max_megapixels = megapixels_of(target_w, target_h)
    preflight_resources(
        model=backend, heaviest_quant=heaviest_quant,
        force=args.force, max_megapixels=max_megapixels,
    )

    # 10) Confirm gate (--yes skips)
    if not args.yes:
        one_eta = estimate_one_seconds(
            load_history(), backend, heaviest_quant, args.preview,
        )
        # Arch #B: refine is the longest-running subcommand. When the
        # history has no prior runs for (backend, quant), the confirm
        # gate would otherwise show NO time estimate at all. For the
        # default flux2-klein-edit-9b first-run case in particular,
        # users also pay a ~15 GB FLUX.2-klein-9B download before any
        # generation starts — surface that explicitly so the [y/N]
        # isn't a blind agreement.
        if one_eta is None and backend == "flux2-klein-edit-9b":
            info(
                "First run on flux2-klein-edit-9b — expect ~15-25 min "
                "of generation plus a one-time ~15 GB model download "
                "if FLUX.2-klein-9B isn't cached yet."
            )
        proceed = _confirm_refine(
            input_path=input_path,
            in_w=in_w,
            in_h=in_h,
            target_w=target_w,
            target_h=target_h,
            output=iteration.output_path,
            eta_seconds=one_eta,
        )
        if not proceed:
            warn("Cancelled — nothing generated.")
            return 0

    # 11) Materialise output dir
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    elif explicit_output is not None:
        explicit_output.parent.mkdir(parents=True, exist_ok=True)

    # 12) BatchContext. input_path is the ORIGINAL input (history will
    # record it correctly). command="refine" routes future replays
    # through cmd_refine via the v0.7.0 history command discriminator.
    env = build_mflux_env(token=token, backend_secret=backend_secret)
    ctx = BatchContext(
        model=backend,
        seed=seed,
        width=target_w,
        height=target_h,
        input_path=input_path,
        effective_custom_prompt=prompt,
        args=args,
        batch_id=None,
        env=env,
        command="refine",
    )

    # 13) Run one iteration
    succeeded: list[tuple[str, Path, int]] = []
    failed: list[tuple[str, int, Path]] = []
    cont = run_one_iteration(
        it=iteration,
        idx=1,
        total=1,
        is_batch=False,
        ctx=ctx,
        logger=None,
        succeeded=succeeded,
        failed=failed,
        enhance_result=None,
        enhance_model=None,
    )
    if not cont:
        return 130  # KeyboardInterrupt

    emit_gated_repo_hint_if_failed(failed=failed, backend_obj=be)

    # 14) Open + summary + exit. v0.9.4 D3: ``Path()`` (= cwd) fallback
    # replaced with ``None``. ``open_results`` handles None by skipping
    # the Finder open for is_batch=False (current refine shape always);
    # the pre-fix expression would have silently opened the process's
    # cwd in Finder if both run_dir and explicit_output were ever None
    # (today unreachable per resolve_output_layout invariant — exactly
    # one of the pair is non-None).
    open_results(
        succeeded=succeeded,
        run_dir=run_dir if run_dir is not None else (
            explicit_output.parent if explicit_output is not None else None
        ),
        is_batch=False,
        no_open=args.no_open,
    )
    return exit_code(is_batch=False, succeeded=succeeded, failed=failed)

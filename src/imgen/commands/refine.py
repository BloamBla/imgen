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
from ..parser import _DEFAULT_REFINE_PROMPT
from ..runs import BatchContext
from ..cmd_helpers import (
    build_refine_iteration,
    emit_gated_repo_hint_if_failed,
    estimate_one_seconds,
    exit_code,
    format_duration,
    load_backend_and_token,
    open_results,
    preflight_resources,
    print_batch_summary,
    resolve_output_layout,
    run_one_iteration,
)
from ..subprocess_helpers import build_mflux_env, format_cmd

__all__ = ["cmd_refine"]


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
    """
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.size[0], im.size[1]
    except Exception as e:
        die(f"refine: couldn't read image dimensions for {path}: {e}",
            code=2)


def _resolve_target_dimensions(args, input_path: Path) -> tuple[int, int]:
    """Compute (output_w, output_h) from --scale OR --width/--height.

    Mutex: --scale + --width/--height rejected. Both --width and
    --height required together when used as explicit dims. Default
    (neither passed) = --scale 1.5.
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
    in_w, in_h = _read_image_dimensions(input_path)
    return (
        _round_to_multiple_of_16(int(in_w * scale)),
        _round_to_multiple_of_16(int(in_h * scale)),
    )


def _confirm_refine(
    *,
    input_path: Path,
    target_w: int,
    target_h: int,
    output: Path,
    eta_seconds: int | None,
) -> bool:
    """Confirm gate for `imgen refine`. Shows input → output dims,
    output path, and optional ETA."""
    in_w, in_h = _read_image_dimensions(input_path)
    info("About to refine 1 image:")
    print(f"   {C.DIM}input: {C.END} {input_path.name} ({in_w}×{in_h})")
    print(f"   {C.DIM}output:{C.END} {output.name} ({target_w}×{target_h})")
    if eta_seconds is not None:
        print(f"   {C.DIM}eta:{C.END}    {format_duration(eta_seconds)} (±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


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

    # 1) Validate input
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        die(f"refine: input not found: {input_path}", code=2)
    if not input_path.is_file():
        die(f"refine: input is not a file: {input_path}", code=2)

    # 2) Target dimensions
    target_w, target_h = _resolve_target_dimensions(args, input_path)

    # 3) Prompt resolution (refine has a baked-in default; no
    # --prompt-file / stdin path — refine prompts are short and
    # deterministic, no need for the secret-from-ps machinery).
    prompt = args.prompt or _DEFAULT_REFINE_PROMPT

    # 4) Backend + token
    backend, be, token, binary, backend_secret = load_backend_and_token(args)

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
        binary=binary,
        width=target_w,
        height=target_h,
        explicit_output=explicit_output,
        run_dir=run_dir,
        seed=seed,
    )

    # 8) Dry-run
    if args.dry_run:
        print(f"Refine target: {target_w}×{target_h}")
        print(f"   {C.DIM}prompt:{C.END} {prompt[:80]}"
              f"{'...' if len(prompt) > 80 else ''}")
        print()
        print(f"Dry run — would execute (refine):")
        print()
        print(format_cmd(iteration.cmd))
        print()
        return 0

    # 9) Preflight
    heaviest_quant = iteration.final_quantize
    preflight_resources(
        backend=backend, heaviest_quant=heaviest_quant, force=args.force,
    )

    # 10) Confirm gate (--yes skips)
    if not args.yes:
        one_eta = estimate_one_seconds(
            load_history(), backend, heaviest_quant, args.preview,
        )
        proceed = _confirm_refine(
            input_path=input_path,
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
        backend=backend,
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

    # 14) Open + summary + exit
    open_results(
        succeeded=succeeded,
        run_dir=run_dir if run_dir is not None else (
            explicit_output.parent if explicit_output is not None else Path()
        ),
        is_batch=False,
        no_open=args.no_open,
    )
    return exit_code(is_batch=False, succeeded=succeeded, failed=failed)

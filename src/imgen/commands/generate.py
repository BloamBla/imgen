"""`imgen generate` (default subcommand) — the actual photo→style transfer.

Pipeline: validate input → build prompt → resolve params (CLI > preset >
preview > default) → detect resolution → resolve output path → load token
→ build mflux command → preflight (RAM/disk/battery/parallel-mflux) →
spawn mflux via the stderr-redacting wrapper → write history entry →
auto-open result in Preview.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from ..backends import BACKENDS, Backend, build_mflux_cmd
from ..checks import check_mflux, check_resources, check_venv
from ..colors import C, die, err, info, ok, step, warn
from ..config import effective_output_dir
from ..defaults import DEFAULTS, PREVIEW_OVERRIDES
from ..history import append_history, load_history
from ..images import apply_scope, detect_resolution
from ..inputs import resolve_to_mflux_input
from ..paths import (
    DEFAULT_OUTPUT_DIR,
    SAFE_OUTPUT_EXTS,
    VENV_BIN,
)
from ..prompt_input import PromptInputError, resolve_prompt
from ..runs import (
    BatchContext,
    BatchLogger,
    Iteration,
    auto_run_dirname,
    next_available_run_dir,
)
from ..styles import get_style
from ..subprocess_helpers import build_mflux_env, format_cmd, run_with_stderr_redaction
from ..tokens import load_token


def _estimate_one_seconds(
    history_entries: list[dict],
    backend: str,
    quantize: int,
    preview: bool,
) -> int | None:
    """Average duration of recent successful generations matching params.

    Returns None when no matching successes — caller suppresses ETA display
    rather than guessing from a coarse fallback table that would be wildly
    off across M1/M2/M3/M4 hardware variance.
    """
    matching = [
        e for e in history_entries
        if e.get("status") == "success"
        and e.get("backend") == backend
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


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"~{seconds // 60} min"


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
    """
    n = len(iterations)
    style_names = [it.style_name for it in iterations]
    print()
    info(f"About to generate {n} images:")
    print(f"   {C.DIM}input:{C.END}   {input_name}")
    print(f"   {C.DIM}styles:{C.END}  {', '.join(style_names)}")
    print(f"   {C.DIM}output:{C.END}  {output_root}")
    if one_eta_seconds is not None:
        total = _format_duration(one_eta_seconds * n)
        per_image = _format_duration(one_eta_seconds)
        print(f"   {C.DIM}eta:{C.END}     {total} total "
              f"({per_image} per image, ±50%)")
    print()
    try:
        ans = input("Continue? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def _check_prompt_style_compat(
    styles_list: list[str],
    effective_custom_prompt: str | None,
) -> None:
    """Reject incompatible (prompt, style) combinations upfront.

    Strict mutex: every listed style must either HAVE its own ``prompt``
    (then no --custom-prompt allowed) OR be param-only (then a CLI
    prompt is required). Mixed lists fail with the full offender list so
    the user can split into two invocations in one shot, not iteratively.

    Raises SystemExit(2) on incompatibility. Returns None on success.
    """
    if effective_custom_prompt:
        prompt_bearing = [s for s in styles_list if get_style(s).get("prompt")]
        if prompt_bearing:
            die(f"Style(s) with their own prompt can't combine with "
                f"--custom-prompt / --prompt-file: {', '.join(prompt_bearing)}.",
                code=2,
                hint="Split into two invocations, or use only param-only "
                     "styles (from ~/.imgen/styles.d/, no `prompt` field).")
    else:
        missing_prompt = [s for s in styles_list if not get_style(s).get("prompt")]
        if missing_prompt:
            die(f"Style(s) without a prompt: {', '.join(missing_prompt)}. "
                "Pass --custom-prompt (or --prompt-file) to supply one.",
                code=2,
                hint="Param-only styles in ~/.imgen/styles.d/ need a "
                     "CLI-supplied prompt.")


def _resolve_output_layout(
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
    """
    # `imgen batch` has no --output flag (always run-dir layout), so its
    # argparse Namespace lacks `output`. getattr keeps this helper
    # composable between generate (--output supported) and batch.
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


def _safe_append_history(entry: dict) -> None:
    """Append to history, warn on unexpected failure.

    history.append_history already swallows OSError and returns 0 on
    disk-level problems (lock contention, ENOSPC). This wrapper exists
    so any *other* exception class — JSON encoding error on a weird
    value, unicode mistake in a path — degrades to a warn() instead of
    aborting `_run_one_iteration` between the subprocess success and
    the log end-marker. Without it, a raise here would skip the
    iteration_end marker, leaving the next iteration's start marker
    flush against this one (looks like a hung iteration in the log).
    (v0.2.4 review IMP-2 — wrap landed in v0.2.5)
    """
    try:
        append_history(entry)
    except Exception as e:  # noqa: BLE001 — degrade-don't-die is the point
        warn(f"history entry not recorded: {type(e).__name__}: {e}")


def _run_one_iteration(
    *,
    it: Iteration,
    idx: int,
    total: int,
    is_batch: bool,
    ctx: BatchContext,
    logger: BatchLogger | None,
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
) -> bool:
    """Execute one mflux iteration end-to-end.

    Steps: print banner → write log start-marker → run subprocess →
    update history → write log end-marker → append to succeeded or
    failed. Mutates the two lists (caller owns the storage; the helper
    is the producer of entries).

    `ctx` is the batch-wide BatchContext (backend, seed, dimensions,
    input path, custom prompt, args namespace, batch_id, env) — built
    once in cmd_generate, shared across every iteration.

    Returns ``True`` to keep the batch loop going, ``False`` if the user
    pressed Ctrl-C (caller should early-exit with 130). The KeyboardInterrupt
    handler writes a `cancelled` history entry and the matching log
    marker before returning so a re-run via `imgen history --replay`
    can pick up where the interrupted batch left off.
    """
    style_name = it.style_name
    output_path = it.output_path
    cmd = it.cmd

    if is_batch:
        step(f"Generating [{idx}/{total}] {style_name} → {output_path.name}")
    else:
        step(f"Generating {style_name} → {output_path.name}")
    print(f"   {C.DIM}backend: {ctx.backend} q{it.final_quantize}  "
          f"steps: {it.final_steps}  guidance: {it.final_guidance}  "
          f"strength: {it.final_strength}  seed: {ctx.seed}{C.END}")
    print(f"   {C.DIM}size: {ctx.width}x{ctx.height}  "
          f"input: {ctx.input_path.name} → output: {output_path}{C.END}")
    print()

    started = datetime.datetime.now()
    history_entry: dict = {
        "ts": started.isoformat(timespec="seconds"),
        "input": str(ctx.input_path),
        "output": str(output_path),
        # `style` stored as the per-iteration style name when there's
        # no custom prompt — replay uses it to reload the same preset.
        "style": style_name if not ctx.effective_custom_prompt else None,
        "custom_prompt": ctx.effective_custom_prompt,
        "scope": ctx.args.scope,
        "preview": ctx.args.preview,
        "prompt": it.prompt,
        "negative": it.negative,
        "seed": ctx.seed,
        "steps": it.final_steps,
        "guidance": it.final_guidance,
        "strength": it.final_strength,
        "backend": ctx.backend,
        "quantize": it.final_quantize,
        "width": ctx.width,
        "height": ctx.height,
        # v0.2.3: ties multi-style entries together. Null for single-
        # style invocations (preserves v0.2.x shape).
        "batch_id": ctx.batch_id,
        "batch_index": f"{idx}/{total}" if is_batch else None,
    }

    if logger is not None:
        logger.iteration_start(idx, total, style_name, started)

    try:
        returncode = run_with_stderr_redaction(
            cmd,
            env=ctx.env,
            log_file=logger.borrow_fd() if logger else None,
        )
    except KeyboardInterrupt:
        warn("Cancelled by user")
        cancel_duration = int(
            (datetime.datetime.now() - started).total_seconds())
        history_entry["status"] = "cancelled"
        history_entry["duration_sec"] = cancel_duration
        _safe_append_history(history_entry)
        if logger is not None:
            logger.iteration_cancelled(idx, total, style_name, cancel_duration)
        return False

    duration = int((datetime.datetime.now() - started).total_seconds())
    history_entry["duration_sec"] = duration
    history_entry["status"] = "success" if returncode == 0 else "failed"
    _safe_append_history(history_entry)

    if logger is not None:
        logger.iteration_end(idx, total, style_name, returncode, duration)

    if returncode != 0:
        err(f"mflux exited with code {returncode} after {duration}s "
            f"— {style_name}")
        failed.append((style_name, returncode, output_path))
        # Continue with next style — don't waste already-done work.
        print()
        return True

    succeeded.append((style_name, output_path, duration))
    print()
    ok(f"Done in {duration // 60}m {duration % 60}s — {output_path}")
    print()
    return True


def _open_results(
    succeeded: list[tuple[str, Path, int]],
    run_dir: Path | None,
    is_batch: bool,
    no_open: bool,
) -> None:
    """Auto-open results — Finder for multi-style runs, Preview for single.

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


def _preflight_resources(
    *,
    backend: str,
    heaviest_quant: int,
    force: bool,
) -> None:
    """Check RAM / disk / battery / parallel-mflux against the heaviest
    quant in the batch.

    --force skips the entire check (caller already opted into the risk
    of swap thrashing). Otherwise:
      * another mflux PID detected → die(4); parallel runs OOM
      * insufficient RAM → die(4); list specific fixes (--preview,
        --quantize 4, --force)
      * low disk → warn (model download might still fit)
      * low battery → warn (charger may be nearby)

    The two hard failures share exit code 4 (resource class) so callers
    can grep by code without parsing messages.
    """
    if force:
        return
    res = check_resources(backend, heaviest_quant)

    if res["other_mflux_pid"] is not None:
        die(f"Another mflux process is already running (PID "
            f"{res['other_mflux_pid']}). Two parallel runs will OOM and "
            "trash each other.",
            code=4,
            hint="Wait for it to finish (check with: ps -p "
                 f"{res['other_mflux_pid']}), or pass --force.")

    if not res["ram_ok"]:
        die(f"Not enough RAM: need ~{res['ram_required_gb']} GB peak "
            f"for {backend} q{heaviest_quant}, only "
            f"{res['ram_available_gb']:.1f} GB available "
            f"(of {res['ram_total_gb']:.0f} GB total).",
            code=4,
            hint=("How to fix:\n"
                  "     • Close other apps (Chrome often eats 5+ GB)\n"
                  "     • Drop quant: --quantize 4 (needs ~9 GB for flux)\n"
                  "     • Or --preview (uses --quantize 4 automatically)\n"
                  "     • Or --force (swaps to disk, very slow, may freeze)"))

    if not res["disk_ok"]:
        warn(f"Only {res['disk_free_gb']:.1f} GB disk free — risky if "
             "model needs download. Consider: imgen clean")
    if not res["battery_ok"]:
        warn(f"Battery {res['battery_pct']}% on battery — long runs may "
             "not finish. Plug in for safety.")


def _print_batch_summary(
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
    total: int,
) -> None:
    """Render the end-of-batch summary block (multi-style only).

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


def _exit_code(
    *,
    is_batch: bool,
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
) -> int:
    """Map (is_batch, succeeded, failed) → process exit code.

    Single-style preserves v0.2.x semantics: mflux's returncode passes
    through so scripts that branch on exit code keep working. Multi-style
    uses distinct codes so callers can tell apart all-ok / all-failed /
    partial without parsing output:

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


def _build_iterations(
    *,
    styles_list: list[str],
    args,
    effective_custom_prompt: str | None,
    merged_defaults: dict,
    be,
    binary: Path,
    input_path: Path,
    width: int,
    height: int,
    explicit_output: Path | None,
    run_dir: Path | None,
    seed: int,
) -> list[Iteration]:
    """Resolve per-style params + build the mflux command for each style.

    The whole batch is pre-built before any subprocess work so:
      * --dry-run can print every cmd that would be executed
      * resource preflight runs against ``max(it.final_quantize)`` —
        no surprise crash on the 3rd image if its quant is heavier
      * confirm gate can show the full list

    Parameter precedence (locked by tests):
      * ``steps``    : CLI > preview > merged_defaults  (preset.steps
                       intentionally NOT honoured — preview must win
                       when the user picks it for speed)
      * ``quantize`` : CLI > preview > merged_defaults  (same reasoning)
      * ``guidance`` : CLI > preset  > merged_defaults
      * ``strength`` : CLI > preset  > merged_defaults
      * ``prompt``   : custom_prompt verbatim (if set), else
                       preset["prompt"] with optional scope substitution
      * ``negative`` : preset.get("negative", "")

    ``output_path`` per iteration:
      * if ``explicit_output`` is set (legacy --output FILE) → that path
      * else ``run_dir / "<input.stem>-<style>.png"``

    Returns ``list[Iteration]`` (frozen) — caller may not mutate entries.
    """
    iterations: list[Iteration] = []
    for style_name in styles_list:
        preset = get_style(style_name)

        if effective_custom_prompt:
            prompt = effective_custom_prompt
        else:
            prompt = preset["prompt"]
            if args.scope:
                prompt = apply_scope(prompt, args.scope)

        negative = preset.get("negative", "")

        if args.steps is not None:
            final_steps = args.steps
        elif args.preview:
            final_steps = PREVIEW_OVERRIDES["steps"]
        else:
            final_steps = merged_defaults["steps"]

        if args.quantize is not None:
            final_quantize = args.quantize
        elif args.preview:
            final_quantize = PREVIEW_OVERRIDES["quantize"]
        else:
            final_quantize = merged_defaults["quantize"]

        if args.guidance is not None:
            final_guidance = args.guidance
        elif "guidance" in preset:
            final_guidance = preset["guidance"]
        else:
            final_guidance = merged_defaults["guidance"]

        if args.strength is not None:
            final_strength = args.strength
        elif "strength" in preset:
            final_strength = preset["strength"]
        else:
            final_strength = merged_defaults["strength"]

        if explicit_output is not None:
            output_path = explicit_output
        else:
            output_path = run_dir / f"{input_path.stem}-{style_name}.png"

        cmd = build_mflux_cmd(
            binary=binary,
            backend=be,
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            negative=negative,
            quantize=final_quantize,
            steps=final_steps,
            guidance=final_guidance,
            strength=final_strength,
            seed=seed,
            width=width,
            height=height,
            mlx_cache_gb=merged_defaults["mlx_cache_gb"],
            battery_stop=merged_defaults["battery_stop"],
        )

        iterations.append(Iteration(
            style_name=style_name,
            prompt=prompt,
            negative=negative,
            final_steps=final_steps,
            final_quantize=final_quantize,
            final_guidance=final_guidance,
            final_strength=final_strength,
            output_path=output_path,
            cmd=cmd,
        ))

    return iterations


def _load_backend_and_token(
    args,
) -> tuple[str, Backend, str | None, Path]:
    """Resolve backend metadata, HF token, and binary path.

    Returns ``(backend_name, backend_dataclass, token_or_none, binary_path)``.
    Exits with code 3 (missing-tool class) on:
      * gated backend without a token
      * venv / mflux not installed
      * the per-backend binary not present in VENV_BIN

    The token is loaded lazily — only invoked when the backend's
    ``needs_token`` is True so open backends (qwen) don't touch the
    keyring/disk for nothing.
    """
    backend = args.backend
    be = BACKENDS[backend]
    token: str | None = None
    if be.needs_token:
        token = load_token()
        if not token:
            die("FLUX backend requires HuggingFace token",
                code=3,
                hint="Run: imgen setup   (or use --backend qwen)")

    if not check_venv() or not check_mflux():
        die("mflux not installed",
            code=3,
            hint="Run: imgen setup")

    binary = VENV_BIN / be.binary
    if not binary.exists():
        die(f"Backend binary not found: {binary}",
            code=3,
            hint="Run: imgen upgrade")

    return backend, be, token, binary


def _resolve_styles_list(args, merged_defaults: dict) -> list[str]:
    """Resolve --style into a list of preset names.

    args.style is either None (not passed) or a pre-validated, de-duped
    list (parser.parse_style_list rejected unknown names already). When
    unset, fall back to the config-merged default style and verify it
    exists — config.toml may point at a preset the user later removed
    from styles.d/.

    Also enforces the --output FILE + multi-style mutex: --output writes
    to one path, M styles → M files needs a directory. Reject upfront so
    the user gets a clear hint before any subprocess spawns.
    """
    if args.style:
        styles_list: list[str] = args.style
    else:
        default_name = merged_defaults["style"]
        try:
            get_style(default_name)
        except KeyError:
            die(f"Default style '{default_name}' not found",
                code=2,
                hint="Check ~/.imgen/config.toml [defaults] style, "
                     "or run: imgen --list-styles")
        styles_list = [default_name]

    # getattr: `imgen batch` doesn't expose --output (mutex with batch's
    # N inputs × M styles always needing a dir layout), so its argparse
    # Namespace lacks the `output` attribute. The check below is generate-
    # specific — silently no-op for batch callers.
    if getattr(args, "output", None) and len(styles_list) > 1:
        die(f"--output FILE writes to one path; multi-style "
            f"(--style {','.join(styles_list)} → {len(styles_list)} files) "
            "needs a directory.",
            code=2,
            hint="Drop --output, or use --output-dir PATH instead.")

    return styles_list


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

    # 2) Resolve --style → list[str], rejecting --output + multi-style upfront.
    styles_list = _resolve_styles_list(args, merged_defaults)

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
    _check_prompt_style_compat(styles_list, effective_custom_prompt)

    # 3b) Scope on custom prompts is a no-op — warn once, not per style.
    if args.scope and effective_custom_prompt:
        warn(f"--scope={args.scope} ignored when using a custom prompt "
             "(--custom-prompt / --prompt-file)")

    # 3c) HEIC pre-conversion (v0.3.0 bonus — also fixes the v0.2.x bug
    # where `imgen generate vacation.heic` died with a cryptic mflux
    # PIL error). resolve_to_mflux_input returns the input unchanged
    # when it isn't HEIC, so the only non-HEIC overhead is one mkdir +
    # one rmdir of an empty /tmp/imgen-heic-* — trivial vs the 30s-3min
    # mflux runtime. ``BatchContext.input_path`` stays the ORIGINAL so
    # history.input records what the user typed; ``Iteration.cmd``
    # references the converted JPEG via ``_build_iterations(input_path=
    # mflux_input)``.
    with tempfile.TemporaryDirectory(prefix="imgen-heic-") as cache_str:
        mflux_input = resolve_to_mflux_input(input_path, Path(cache_str))

        # 4) Resolution (read from the JPEG — mflux's PIL can't open HEIC).
        if args.width and args.height:
            width, height = args.width, args.height
        else:
            width, height = detect_resolution(mflux_input, preview=args.preview)

        # 5) Output root + run_dir (one folder for all M iterations).
        explicit_output, run_dir = _resolve_output_layout(args, config_output_dir)

        # 6) Backend, token, binary (same for all M).
        backend, be, token, binary = _load_backend_and_token(args)

        # 7) Seed — one seed for the whole invocation so multi-style runs use
        # the same noise pattern (only style differs → fair preset comparison).
        seed = (args.seed if args.seed is not None
                else int.from_bytes(os.urandom(4), "big"))

        # 8) Pre-resolve each iteration's params + cmd so dry-run can show
        # all M and we can preflight resources against the heaviest one.
        iterations = _build_iterations(
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
        _preflight_resources(
            backend=backend, heaviest_quant=heaviest_quant, force=args.force
        )

        # 11) Confirm gate (multi-style only — single-style keeps v0.2.x's
        # zero-prompt UX). --yes skips. Fires AFTER preflight so we never
        # ask the user to confirm a batch we know we can't run anyway.
        if is_batch and not args.yes:
            one_eta = _estimate_one_seconds(
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
        # _run_one_iteration, mflux stderr-tee via borrow_fd() inside
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
            env = build_mflux_env(token)

            # 15) The loop. Failures don't break the batch — log + continue,
            # surface the summary at the end. Single-style retains v0.2.x
            # exit semantics (mflux return code passes through).
            succeeded: list[tuple[str, Path, int]] = []
            failed: list[tuple[str, int, Path]] = []
            total = len(iterations)

            # Bundle the 9 batch-invariant args into a single BatchContext so
            # _run_one_iteration's signature stays compact — and v0.3.0's
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
                cont = _run_one_iteration(
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
            _open_results(
                succeeded=succeeded,
                run_dir=run_dir,
                is_batch=is_batch,
                no_open=args.no_open,
            )

            # 17) End-of-batch summary (only for multi-style — single-style
            # keeps the v0.2.x lean output).
            if is_batch:
                _print_batch_summary(succeeded, failed, total)

            # 18) Exit code (single-style passthrough vs multi-style 0/1/5).
            return _exit_code(
                is_batch=is_batch, succeeded=succeeded, failed=failed
            )
        finally:
            if logger is not None:
                logger.close()

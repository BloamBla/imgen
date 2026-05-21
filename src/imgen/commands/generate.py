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
import shutil
import subprocess
from pathlib import Path

from ..backends import BACKENDS, build_mflux_cmd
from ..checks import check_mflux, check_resources, check_venv
from ..colors import C, die, err, ok, step, warn
from ..config import effective_output_dir
from ..defaults import DEFAULTS, PREVIEW_OVERRIDES
from ..history import append_history
from ..images import apply_scope, detect_resolution
from ..paths import (
    DEFAULT_OUTPUT_DIR,
    SAFE_OUTPUT_EXTS,
    VENV_BIN,
    auto_run_dirname,
    next_available_run_dir,
)
from ..prompt_input import PromptInputError, resolve_prompt
from ..styles import get_style
from ..subprocess_helpers import format_cmd, run_with_stderr_redaction
from ..tokens import load_token


def cmd_generate(args) -> int:
    # Config-aware defaults (config.toml [defaults] merged over module
    # DEFAULTS, populated in cli.main). For old callers that bypass
    # cli.main — e.g. direct test invocation — fall back to module DEFAULTS.
    merged_defaults = getattr(args, "imgen_merged_defaults", DEFAULTS)
    config_output_dir = getattr(args, "imgen_config_output_dir", None)

    # 1) Validate input
    input_path = Path(args.image).expanduser().resolve()
    if not input_path.exists():
        die(f"Image not found: {input_path}",
            code=2,
            hint="Check the path. Use absolute path if unsure.")
    if not input_path.is_file():
        die(f"Not a file: {input_path}", code=2)

    # 2) Resolve style preset if --style passed
    preset: dict | None = None
    if args.style:
        try:
            preset = get_style(args.style)
        except KeyError:
            die(f"Unknown style: {args.style}",
                code=2, hint="See: imgen --list-styles")

    # 2a) Resolve effective custom prompt — could be argv text, stdin
    # (--custom-prompt -), or a file (--prompt-file). Both file/stdin
    # paths keep prompt text out of `ps auxww`.
    try:
        effective_custom_prompt = resolve_prompt(
            custom_prompt=args.custom_prompt,
            prompt_file=getattr(args, "prompt_file", None),
        )
    except PromptInputError as e:
        die(str(e), code=2)

    # 3) Determine prompt source. Four valid combos + two errors:
    #    a. --style (w/ prompt)            → use style.prompt
    #    b. --style (param-only, no prompt) + custom-prompt → use both
    #    c. custom-prompt (no --style)     → use custom, no preset params
    #    d. no --style and no custom-prompt → use default style
    #    ERR: --style (w/ prompt) AND custom-prompt → mutex
    #    ERR: --style (param-only) AND no custom-prompt → no prompt anywhere
    if effective_custom_prompt:
        if preset and preset.get("prompt"):
            die("--style (which has a prompt) and a custom prompt "
                "(--custom-prompt / --prompt-file) are mutually exclusive.",
                code=2,
                hint="To use only a style's parameters with your own prompt, "
                     "create a param-only TOML (no `prompt` field) in "
                     "~/.imgen/styles.d/.")
        prompt = effective_custom_prompt
        negative = preset.get("negative", "") if preset else ""
        style_name = args.style or "custom"
    else:
        if preset is None:
            # No --style at all → load the default style
            style_name = merged_defaults["style"]
            try:
                preset = get_style(style_name)
            except KeyError:
                die(f"Default style '{style_name}' not found",
                    code=2,
                    hint="Check ~/.imgen/config.toml [defaults] style, "
                         "or run: imgen --list-styles")
        else:
            style_name = args.style
        if not preset.get("prompt"):
            die(f"Style '{style_name}' has no prompt — pass --custom-prompt "
                "to supply one.",
                code=2,
                hint="Param-only styles in ~/.imgen/styles.d/ need a "
                     "CLI-supplied prompt.")
        prompt = preset["prompt"]
        negative = preset.get("negative", "")

    # 3a) Apply --scope (warn if combined with a custom prompt — scope
    # works by string-replacing tokens in built-in preset prompts and
    # can't reliably apply to arbitrary user text).
    if args.scope:
        if effective_custom_prompt:
            warn(f"--scope={args.scope} ignored when using a custom prompt "
                 "(--custom-prompt / --prompt-file)")
        else:
            prompt = apply_scope(prompt, args.scope)

    # 3b) Resolve final parameter values:
    #   CLI flag (if set) > style preset > preview override > config/global default
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
    elif preset and "guidance" in preset:
        final_guidance = preset["guidance"]
    else:
        final_guidance = merged_defaults["guidance"]

    if args.strength is not None:
        final_strength = args.strength
    elif preset and "strength" in preset:
        final_strength = preset["strength"]
    else:
        final_strength = merged_defaults["strength"]

    # 4) Resolution
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = detect_resolution(input_path, preview=args.preview)

    # 5) Output path. Precedence on the auto-derived dir:
    #    env IMGEN_OUTPUT_DIR > config.toml [defaults] output_dir > module default
    if args.output:
        # Explicit single-file output — bypass the folder-per-invocation
        # layout entirely. Existing v0.2.x scripts that pin a path keep
        # working unchanged. The directory is mkdir'd later, only if we
        # actually run (not on --dry-run).
        output_path = Path(args.output).expanduser().resolve()
        run_dir: Path | None = None
    else:
        # New in v0.2.3: each invocation gets its own timestamped folder
        # under the resolved output root. File is <basename>-<style>.png;
        # mtime gives completion-time ordering in Finder for free, so we
        # don't repeat a timestamp inside the filename.
        # Precedence: --output-dir > $IMGEN_OUTPUT_DIR > config.toml > default.
        # mkdir is deferred until after the dry-run check so a dry-run
        # doesn't pollute ~/Desktop/imgen/ with empty timestamped dirs.
        parent = effective_output_dir(
            cli_value=getattr(args, "output_dir", None),
            config_value=config_output_dir,
            module_default=DEFAULT_OUTPUT_DIR,
        )
        run_dir = next_available_run_dir(parent, auto_run_dirname())
        output_path = run_dir / f"{input_path.stem}-{style_name}.png"

    # 6) Backend & token
    backend = args.backend
    be = BACKENDS[backend]
    token: str | None = None
    if be.needs_token:
        token = load_token()
        if not token:
            die("FLUX backend requires HuggingFace token",
                code=3,
                hint="Run: imgen setup   (or use --backend qwen)")

    # 7) Build mflux command
    if not check_venv() or not check_mflux():
        die("mflux not installed",
            code=3,
            hint="Run: imgen setup")

    binary = VENV_BIN / be.binary
    if not binary.exists():
        die(f"Backend binary not found: {binary}",
            code=3,
            hint="Run: imgen upgrade")

    seed = (args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), "big"))

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

    # 8) Dry run (skip resource checks — just show what would run)
    if args.dry_run:
        step("Dry run — would execute:")
        print()
        print(format_cmd(cmd))
        print()
        return 0

    # 8a) mkdir the output dir now that we know we'll actually run.
    # Deferred from path-resolution above so --dry-run doesn't leave
    # empty timestamped folders behind.
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 8a) Resource preflight — block runs that can't reasonably finish
    if not args.force:
        res = check_resources(backend, final_quantize)

        # HARD: another mflux is already crunching — would compete for GPU+RAM
        if res["other_mflux_pid"] is not None:
            die(f"Another mflux process is already running (PID "
                f"{res['other_mflux_pid']}). Two parallel runs will OOM and "
                "trash each other.",
                code=4,
                hint="Wait for it to finish (check with: ps -p "
                     f"{res['other_mflux_pid']}), or pass --force.")

        # HARD: not enough RAM for chosen backend+quant
        if not res["ram_ok"]:
            die(f"Not enough RAM: need ~{res['ram_required_gb']} GB peak "
                f"for {backend} q{final_quantize}, only "
                f"{res['ram_available_gb']:.1f} GB available "
                f"(of {res['ram_total_gb']:.0f} GB total).",
                code=4,
                hint=("How to fix:\n"
                      "     • Close other apps (Chrome often eats 5+ GB)\n"
                      "     • Drop quant: --quantize 4 (needs ~9 GB for flux)\n"
                      "     • Or --preview (uses --quantize 4 automatically)\n"
                      "     • Or --force (swaps to disk, very slow, may freeze)"))

        # SOFT: disk low — might fail mid-run if download needed
        if not res["disk_ok"]:
            warn(f"Only {res['disk_free_gb']:.1f} GB disk free — risky if "
                 "model needs download. Consider: imgen clean")
        # SOFT: low battery
        if not res["battery_ok"]:
            warn(f"Battery {res['battery_pct']}% on battery — long runs may "
                 "not finish. Plug in for safety.")

    # 9) Pre-flight info
    step(f"Generating {style_name} → {output_path.name}")
    print(f"   {C.DIM}backend: {backend} q{final_quantize}  "
          f"steps: {final_steps}  guidance: {final_guidance}  "
          f"strength: {final_strength}  seed: {seed}{C.END}")
    print(f"   {C.DIM}size: {width}x{height}  "
          f"input: {input_path.name} → output: {output_path}{C.END}")
    print()

    # 10) Run — minimal env (don't forward random secrets from parent shell)
    env = {}
    for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
                "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "MLX_METAL_PRECOMPILE_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    if token:
        env["HF_TOKEN"] = token

    # tqdm sees stderr=PIPE as non-tty and falls back to a narrow default
    # width. Forward the real terminal size so the progress bar fills the
    # window like it did before stderr redaction was added in v0.1.1.
    term = shutil.get_terminal_size(fallback=(80, 24))
    env["COLUMNS"] = str(term.columns)
    env["LINES"] = str(term.lines)

    started = datetime.datetime.now()
    # Id is assigned by append_history() under flock to avoid parallel collisions
    history_entry = {
        "ts": started.isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(output_path),
        # Store the EFFECTIVE prompt (resolved from --custom-prompt /
        # --prompt-file / stdin) so `imgen replay` reproduces the actual
        # text rather than re-reading stdin or a file that may have moved.
        "style": style_name if not effective_custom_prompt else None,
        "custom_prompt": effective_custom_prompt,
        "scope": args.scope,
        "preview": args.preview,
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "steps": final_steps,
        "guidance": final_guidance,
        "strength": final_strength,
        "backend": backend,
        "quantize": final_quantize,
        "width": width,
        "height": height,
    }

    try:
        returncode = run_with_stderr_redaction(cmd, env=env)
    except KeyboardInterrupt:
        warn("Cancelled by user")
        history_entry["status"] = "cancelled"
        history_entry["duration_sec"] = int(
            (datetime.datetime.now() - started).total_seconds())
        append_history(history_entry)
        return 130

    duration = int((datetime.datetime.now() - started).total_seconds())
    history_entry["duration_sec"] = duration
    history_entry["status"] = "success" if returncode == 0 else "failed"
    append_history(history_entry)

    if returncode != 0:
        err(f"mflux exited with code {returncode} after {duration}s")
        return returncode

    print()
    ok(f"Done in {duration // 60}m {duration % 60}s — {output_path}")

    # 11) Open in Preview (defence-in-depth: re-check ext before `open`,
    # since macOS `open` would auto-launch the registered app for the suffix)
    if not args.no_open:
        if output_path.suffix.lower() not in SAFE_OUTPUT_EXTS:
            warn(f"Skipping auto-open: unsafe extension {output_path.suffix}")
        else:
            try:
                subprocess.run(["open", str(output_path)], check=False)
            except FileNotFoundError:
                pass

    return 0

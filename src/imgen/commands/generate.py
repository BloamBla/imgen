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
import uuid
from pathlib import Path

from ..backends import BACKENDS, build_mflux_cmd
from ..checks import check_mflux, check_resources, check_venv
from ..colors import C, die, err, info, ok, step, warn
from ..config import effective_output_dir
from ..defaults import DEFAULTS, PREVIEW_OVERRIDES
from ..history import append_history, load_history
from ..images import apply_scope, detect_resolution
from ..paths import (
    DEFAULT_OUTPUT_DIR,
    SAFE_OUTPUT_EXTS,
    VENV_BIN,
)
from ..prompt_input import PromptInputError, resolve_prompt
from ..runs import (
    LOGS_DIR,
    auto_run_dirname,
    ensure_logs_dir,
    next_available_run_dir,
    open_log_file_append,
)
from ..styles import get_style
from ..subprocess_helpers import format_cmd, run_with_stderr_redaction
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
    iterations: list[dict],
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
    style_names = [it["style_name"] for it in iterations]
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

    if args.output and len(styles_list) > 1:
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
    # Strict rule: if a custom prompt is set, every listed style must be
    # param-only (no `prompt:` key). If no custom prompt, every listed
    # style must HAVE a prompt. Mixed lists → reject with offender list
    # so the user splits into two invocations.
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

    # 3b) Scope on custom prompts is a no-op — warn once, not per style.
    if args.scope and effective_custom_prompt:
        warn(f"--scope={args.scope} ignored when using a custom prompt "
             "(--custom-prompt / --prompt-file)")

    # 4) Resolution (input-derived — same for all M iterations).
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = detect_resolution(input_path, preview=args.preview)

    # 5) Output root + run_dir (one folder for all M iterations).
    if args.output:
        # Single file, bypass run-folder layout entirely.
        explicit_output = Path(args.output).expanduser().resolve()
        run_dir: Path | None = None
    else:
        parent = effective_output_dir(
            cli_value=getattr(args, "output_dir", None),
            config_value=config_output_dir,
            module_default=DEFAULT_OUTPUT_DIR,
        )
        run_dir = next_available_run_dir(parent, auto_run_dirname())
        explicit_output = None

    # 6) Backend, token, binary (same for all M).
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

    # 7) Seed — one seed for the whole invocation so multi-style runs use
    # the same noise pattern (only style differs → fair preset comparison).
    seed = (args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), "big"))

    # 8) Pre-resolve each iteration's params + cmd so dry-run can show
    # all M and we can preflight resources against the heaviest one.
    iterations: list[dict] = []
    for style_name in styles_list:
        preset = get_style(style_name)

        if effective_custom_prompt:
            prompt = effective_custom_prompt
        else:
            prompt = preset["prompt"]
            if args.scope:
                prompt = apply_scope(prompt, args.scope)

        negative = preset.get("negative", "")

        # Per-style param resolution. CLI > preset > preview > merged_defaults.
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

        iterations.append({
            "style_name": style_name,
            "prompt": prompt,
            "negative": negative,
            "final_steps": final_steps,
            "final_quantize": final_quantize,
            "final_guidance": final_guidance,
            "final_strength": final_strength,
            "output_path": output_path,
            "cmd": cmd,
        })

    multi = len(iterations) >= 2
    # batch_id stamps every history entry from this invocation when M >= 2,
    # making `imgen history --batch <id>` (v0.3.0+) trivial. Null for
    # single-style runs to preserve v0.2.x history shape.
    # 12 hex chars = 48 bits — collision probability is astronomical at
    # single-user scale (one Mac, one human). Keeps log filenames short
    # and readable vs the 32-char full uuid4. ULID would give lex-
    # sortable IDs, but auto_run_dirname() already provides the
    # chronological sort via run-folder names; batch_id only needs to
    # be unique. (architect F4 from v0.2.3 review)
    batch_id: str | None = uuid.uuid4().hex[:12] if multi else None

    # 9) Dry run — show every M cmd, skip resource checks + history.
    if args.dry_run:
        for it in iterations:
            step(f"Dry run — would execute ({it['style_name']}):")
            print()
            print(format_cmd(it["cmd"]))
            print()
        return 0

    # 10) Resource preflight — check against the heaviest quant in the
    # batch (all iterations share a backend + quantize today since neither
    # is per-style; verifying the first is sufficient and future-proof
    # via max() if per-style quantize is ever added).
    heaviest_quant = max(it["final_quantize"] for it in iterations)
    if not args.force:
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

    # 11) Confirm gate (multi-style only — single-style keeps v0.2.x's
    # zero-prompt UX). --yes skips. Fires AFTER preflight so we never
    # ask the user to confirm a batch we know we can't run anyway.
    if multi and not args.yes:
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

    # 13) Per-batch log file (multi-style only). One log per invocation,
    # named after batch_id. mflux stderr (after token redaction) is
    # appended; cmd_generate writes per-image markers below for grep-
    # ability. Retention (30 days) is enforced by `imgen clean`. All
    # writes go through open_log_file_append (binary mode, 0o600 from
    # creation — see paths.py for the umask rationale).
    log_path: Path | None = None
    if multi:
        ensure_logs_dir()
        log_path = LOGS_DIR / f"{batch_id}.log"
        header = (
            f"# imgen batch {batch_id}\n"
            f"# started:  {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"# input:    {input_path}\n"
            f"# styles:   {', '.join(it['style_name'] for it in iterations)}\n"
            f"# output:   {run_dir}\n"
            f"# backend:  {backend} q{heaviest_quant}  "
            f"preview={args.preview}  scope={args.scope}  seed={seed}\n"
        )
        with open_log_file_append(log_path) as f:
            f.write(header.encode())

    # 14) Minimal env (don't forward random secrets from parent shell).
    env: dict[str, str] = {}
    for key in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR",
                "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "MLX_METAL_PRECOMPILE_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    if token:
        env["HF_TOKEN"] = token
    term = shutil.get_terminal_size(fallback=(80, 24))
    env["COLUMNS"] = str(term.columns)
    env["LINES"] = str(term.lines)

    # 15) The loop. Failures don't break the batch — log + continue,
    # surface the summary at the end. Single-style retains v0.2.x exit
    # semantics (mflux return code passes through).
    succeeded: list[tuple[str, Path, int]] = []
    failed: list[tuple[str, int, Path]] = []
    total = len(iterations)

    for idx, it in enumerate(iterations, start=1):
        style_name = it["style_name"]
        output_path = it["output_path"]
        cmd = it["cmd"]

        if multi:
            step(f"Generating [{idx}/{total}] {style_name} → {output_path.name}")
        else:
            step(f"Generating {style_name} → {output_path.name}")
        print(f"   {C.DIM}backend: {backend} q{it['final_quantize']}  "
              f"steps: {it['final_steps']}  guidance: {it['final_guidance']}  "
              f"strength: {it['final_strength']}  seed: {seed}{C.END}")
        print(f"   {C.DIM}size: {width}x{height}  "
              f"input: {input_path.name} → output: {output_path}{C.END}")
        print()

        started = datetime.datetime.now()
        history_entry: dict = {
            "ts": started.isoformat(timespec="seconds"),
            "input": str(input_path),
            "output": str(output_path),
            # `style` stored as the per-iteration style name when there's
            # no custom prompt — replay uses it to reload the same preset.
            "style": style_name if not effective_custom_prompt else None,
            "custom_prompt": effective_custom_prompt,
            "scope": args.scope,
            "preview": args.preview,
            "prompt": it["prompt"],
            "negative": it["negative"],
            "seed": seed,
            "steps": it["final_steps"],
            "guidance": it["final_guidance"],
            "strength": it["final_strength"],
            "backend": backend,
            "quantize": it["final_quantize"],
            "width": width,
            "height": height,
            # NEW v0.2.3: ties multi-style entries together. Null for
            # single-style invocations (preserves v0.2.x shape).
            "batch_id": batch_id,
            "batch_index": f"{idx}/{total}" if multi else None,
        }

        if log_path is not None:
            marker = (f"\n=== [{idx}/{total}] {style_name} → "
                      f"{started.isoformat(timespec='seconds')} ===\n")
            with open_log_file_append(log_path) as f:
                f.write(marker.encode())
        try:
            returncode = run_with_stderr_redaction(cmd, env=env, log_path=log_path)
        except KeyboardInterrupt:
            warn("Cancelled by user")
            cancel_duration = int(
                (datetime.datetime.now() - started).total_seconds())
            history_entry["status"] = "cancelled"
            history_entry["duration_sec"] = cancel_duration
            append_history(history_entry)
            if log_path is not None:
                marker = (f"\n=== [{idx}/{total}] {style_name} → "
                          f"CANCELLED in {cancel_duration}s ===\n")
                with open_log_file_append(log_path) as f:
                    f.write(marker.encode())
            return 130

        duration = int((datetime.datetime.now() - started).total_seconds())
        history_entry["duration_sec"] = duration
        history_entry["status"] = "success" if returncode == 0 else "failed"
        append_history(history_entry)

        if log_path is not None:
            status = ("ok" if returncode == 0
                      else f"FAILED exit={returncode}")
            marker = (f"\n=== [{idx}/{total}] {style_name} → {status} "
                      f"in {duration}s ===\n")
            with open_log_file_append(log_path) as f:
                f.write(marker.encode())

        if returncode != 0:
            err(f"mflux exited with code {returncode} after {duration}s "
                f"— {style_name}")
            failed.append((style_name, returncode, output_path))
            # Continue with next style — don't waste already-done work.
            print()
            continue

        succeeded.append((style_name, output_path, duration))
        print()
        ok(f"Done in {duration // 60}m {duration % 60}s — {output_path}")
        print()

    # 16) Open in Preview (defence-in-depth: re-check ext before `open`,
    # since macOS `open` would auto-launch the registered app for the suffix).
    # For multi-style we open the run folder so the user sees all results
    # in Finder at once; for single-style keep v0.2.x behaviour (open the
    # one file in Preview).
    if not args.no_open and succeeded:
        if multi and run_dir is not None:
            # Belt-and-braces: only open if it's actually a directory.
            # `open <file>` would auto-launch the registered app for the
            # extension, which the SAFE_OUTPUT_EXTS guard below
            # protects against for the single-file branch. Symbolic
            # link or other exotic path → skip the open. (security I3
            # from v0.2.3 review)
            if run_dir.is_dir():
                try:
                    subprocess.run(["open", str(run_dir)], check=False)
                except FileNotFoundError:
                    pass
        else:
            last_path = succeeded[-1][1]
            if last_path.suffix.lower() not in SAFE_OUTPUT_EXTS:
                warn(f"Skipping auto-open: unsafe extension {last_path.suffix}")
            else:
                try:
                    subprocess.run(["open", str(last_path)], check=False)
                except FileNotFoundError:
                    pass

    # 17) End-of-batch summary (only for multi-style — single-style keeps
    # the v0.2.x lean output).
    if multi:
        print()
        step(f"Batch summary ({total} generation{'s' if total != 1 else ''})")
        if succeeded:
            ok(f"{len(succeeded)} ok")
        if failed:
            err(f"{len(failed)} failed:")
            for sn, rc, _ in failed:
                print(f"   {C.DIM}• {sn}: exit {rc}{C.END}")

    # 18) Exit code. Single-style: mflux's returncode passes through
    # (preserves v0.2.x scripted-usage). Multi-style:
    #   - all ok → 0
    #   - all failed → 1
    #   - mixed (some ok, some failed) → 5 (distinct from user-input 2,
    #     missing-tool 3, resource 4 — keeps grep-by-code scripting clean)
    if not multi:
        # Single-style: succeeded has 1 if ok, 0 if failed. Pass through.
        if failed:
            return failed[0][1]
        return 0
    if failed and not succeeded:
        return 1
    if failed:
        return 5
    return 0

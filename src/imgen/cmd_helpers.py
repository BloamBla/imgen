"""Shared helpers for ``cmd_generate`` and ``cmd_batch``.

Extracted v0.3.1 from ``commands/generate.py`` after the v0.3.0
architect review flagged the cross-module underscore-prefix import
pattern: ``batch.py`` was reaching into ``generate._helpers`` for 12
functions, which by Python convention says "private to that module"
while actually being shared. With two command modules consuming the
same pipeline pieces, a dedicated helpers module is the cleaner seam.

What lives here (alphabetical):

* :func:`build_iterations` — pre-build the M (or N×M) iteration plan
  before any subprocess work so dry-run / preflight / confirm gate
  can all reason about the full grid.
* :func:`check_prompt_style_compat` — reject incompatible
  (prompt, style) combinations upfront.
* :func:`estimate_one_seconds` — ETA helper backed by recent
  successful history entries matching backend/quant/preview.
* :func:`exit_code` — single-style passthrough vs batch 0/1/5 mapping.
* :func:`format_duration` — short human duration formatter.
* :func:`load_backend_and_token` — resolve backend dataclass + HF
  token (if needed) + mflux binary path; exits 3 on missing tool.
* :func:`open_results` — Finder/Preview launch with extension safety
  re-check; silent no-op on ``--no-open``.
* :func:`preflight_resources` — RAM / disk / battery / parallel-mflux
  gate; ``--force`` skips.
* :func:`print_batch_summary` — end-of-batch ok/fail count block.
* :func:`resolve_output_layout` — single-file ``--output`` vs run-dir
  layout (pure; mutex-with-multi-style check lives in generate.py
  as ``_check_output_style_mutex`` since batch has no ``--output``).
* :func:`resolve_styles_list` — args.style (parser-validated list) or
  fallback to merged-defaults' single name; pure.
* :func:`run_one_iteration` — one mflux invocation end-to-end:
  banner + log markers + subprocess + history append + result list.
* :func:`safe_append_history` — append-history wrapper that degrades
  unexpected exceptions to ``warn()`` instead of aborting the run loop.

What deliberately stays in ``commands/generate.py``:

* ``_confirm_batch`` — generate's 1×M confirm gate UI (batch has
  its own ``_confirm_dir_batch`` with N×M counts).
* ``_validate_input_path`` — generate-only (batch uses
  ``discover_inputs`` for its dir-of-files input).
* ``_check_output_style_mutex`` — generate-only mutex (batch has no
  ``--output FILE`` flag).
* ``cmd_generate`` — the orchestrator.

Naming convention: the moved functions drop the leading underscore.
They were "private to generate.py" by labelling; now they're a
documented shared surface used by both command modules. Functions
that genuinely stay generate-private keep the underscore.
"""
from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from .backends import BACKENDS, Backend, build_mflux_cmd
from .checks import check_mflux, check_resources, check_venv
from .colors import C, die, err, ok, step, warn
from .config import effective_output_dir
from .defaults import PREVIEW_OVERRIDES
from .history import append_history
from .images import apply_scope
from .paths import DEFAULT_OUTPUT_DIR, SAFE_OUTPUT_EXTS, VENV_BIN
from .runs import (
    BatchContext,
    BatchLogger,
    Iteration,
    auto_run_dirname,
    next_available_run_dir,
)
from .styles import get_style
from .subprocess_helpers import run_with_stderr_redaction
from .tokens import load_token

__all__ = [
    "build_iterations",
    "check_prompt_style_compat",
    "estimate_one_seconds",
    "exit_code",
    "format_duration",
    "load_backend_and_token",
    "open_results",
    "preflight_resources",
    "print_batch_summary",
    "resolve_output_layout",
    "resolve_styles_list",
    "run_one_iteration",
    "safe_append_history",
]


# ── ETA helpers ─────────────────────────────────────────────────────────


def estimate_one_seconds(
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


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"~{seconds // 60} min"


# ── Prompt / style compatibility ────────────────────────────────────────


def check_prompt_style_compat(
    styles_list: list[str],
    effective_custom_prompt: str | None,
) -> None:
    """Reject only the genuinely incompatible (prompt, style) combos.

    v0.3.5: `--custom-prompt` now AUGMENTS full-style prompts rather
    than replacing them — the augmentation logic lives in
    :func:`build_iterations`. The only remaining incompatibility is
    "param-only style + no prompt source": a style with no built-in
    `prompt` field and no `--custom-prompt` / `--prompt-file` leaves
    the iteration with nothing to send mflux.

    Pre-v0.3.5 there was a second mutex (full-style + custom-prompt)
    that's now lifted — see :func:`build_iterations` for the
    augmentation semantics. The UX wart where a bare
    ``imgen photo.jpg --custom-prompt "..."`` invocation died because
    the default style "pixar" had a prompt is also fixed by the lift.

    Raises SystemExit(2) on the remaining incompatibility. Returns
    None on success.
    """
    if effective_custom_prompt:
        # v0.3.5: full-style + custom-prompt now augment — see
        # build_iterations. Nothing to reject here.
        return
    # No custom prompt → every listed style must have its own.
    missing_prompt = [s for s in styles_list if not get_style(s).get("prompt")]
    if missing_prompt:
        die(f"Style(s) without a prompt: {', '.join(missing_prompt)}. "
            "Pass --custom-prompt (or --prompt-file) to supply one.",
            code=2,
            hint="Param-only styles in ~/.imgen/styles.d/ need a "
                 "CLI-supplied prompt.")


# ── Output layout ───────────────────────────────────────────────────────


def resolve_output_layout(
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

    ``imgen batch`` always lands in the run-dir branch — its parser
    stanza omits ``--output FILE`` entirely. ``getattr`` accommodates
    that so the same helper composes between generate and batch.
    """
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


# ── History append (guarded) ────────────────────────────────────────────


def safe_append_history(entry: dict) -> None:
    """Append to history, warn on unexpected failure.

    history.append_history already swallows OSError and returns 0 on
    disk-level problems (lock contention, ENOSPC). This wrapper exists
    so any *other* exception class — JSON encoding error on a weird
    value, unicode mistake in a path — degrades to a warn() instead of
    aborting :func:`run_one_iteration` between the subprocess success
    and the log end-marker. Without it, a raise here would skip the
    iteration_end marker, leaving the next iteration's start marker
    flush against this one (looks like a hung iteration in the log).
    (v0.2.4 review IMP-2 — wrap landed in v0.2.5)
    """
    try:
        append_history(entry)
    except Exception as e:  # noqa: BLE001 — degrade-don't-die is the point
        warn(f"history entry not recorded: {type(e).__name__}: {e}")


# ── One iteration: the subprocess workhorse ─────────────────────────────


def run_one_iteration(
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
    once in the caller, shared across every iteration.

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
        safe_append_history(history_entry)
        if logger is not None:
            logger.iteration_cancelled(idx, total, style_name, cancel_duration)
        return False

    duration = int((datetime.datetime.now() - started).total_seconds())
    history_entry["duration_sec"] = duration
    history_entry["status"] = "success" if returncode == 0 else "failed"
    safe_append_history(history_entry)

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


# ── Results opening / preflight / summary / exit code ──────────────────


def open_results(
    succeeded: list[tuple[str, Path, int]],
    run_dir: Path | None,
    is_batch: bool,
    no_open: bool,
) -> None:
    """Auto-open results — Finder for batch runs, Preview for single.

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


def preflight_resources(
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


def print_batch_summary(
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
    total: int,
) -> None:
    """Render the end-of-batch summary block (batch runs only).

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


def exit_code(
    *,
    is_batch: bool,
    succeeded: list[tuple[str, Path, int]],
    failed: list[tuple[str, int, Path]],
) -> int:
    """Map (is_batch, succeeded, failed) → process exit code.

    Single-style preserves v0.2.x semantics: mflux's returncode passes
    through so scripts that branch on exit code keep working. Batch
    runs use distinct codes so callers can tell apart all-ok / all-
    failed / partial without parsing output:

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


# ── Iteration plan + backend resolution + styles list ───────────────────


def build_iterations(
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
    # `args.style` is None when the parser fell back to merged_defaults
    # for the default style (no explicit --style passed). Used below to
    # gate augmentation: if user didn't explicitly pick a style, their
    # `--custom-prompt` should drive the prompt content entirely rather
    # than augment the default style's prompt — otherwise a bare
    # `imgen photo.jpg --custom-prompt "make sepia"` would produce
    # "Pixar 3D character + sepia" which is nonsense for that invocation
    # shape. (v0.3.5 UX wart fix.)
    style_was_explicit = bool(getattr(args, "style", None))

    for style_name in styles_list:
        preset = get_style(style_name)
        preset_prompt = preset.get("prompt")

        if effective_custom_prompt and preset_prompt and style_was_explicit:
            # v0.3.5 augmentation: explicit full-style + custom-prompt
            # → preset prompt is the BASE (scope applied to it), then
            # user's --custom-prompt text is appended as a final detail.
            # Lets the user share one common addition ("wearing a red
            # kimono") across multiple styles in the same invocation
            # via `-s anime,ghibli,pixar --custom-prompt "..."`.
            #
            # Scope applies only to the base — the user's added text is
            # passed through verbatim so scope-mode replacements don't
            # accidentally touch user wording (e.g. their literal
            # "this person" stays "this person", not rewritten).
            base = preset_prompt
            if args.scope:
                base = apply_scope(base, args.scope)
            prompt = base + ", " + effective_custom_prompt
        elif effective_custom_prompt:
            # Custom-only path:
            #   * param-only style (no `prompt` field) — style provides
            #     params, user provides the prompt
            #   * OR no explicit --style — default style's params apply
            #     but its prompt is bypassed (UX wart fix: a bare
            #     `imgen photo.jpg --custom-prompt "..."` no longer
            #     blends the default Pixar prompt with the user's text)
            prompt = effective_custom_prompt
        else:
            # No custom-prompt → preset prompt is the prompt.
            prompt = preset_prompt
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


def load_backend_and_token(
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


def resolve_styles_list(args, merged_defaults: dict) -> list[str]:
    """Resolve ``args.style`` into a list of preset names.

    ``args.style`` is either ``None`` (not passed) or a pre-validated,
    de-duped list (parser already rejected unknown names). When unset,
    fall back to the config-merged default style and verify it exists —
    config.toml may point at a preset the user later removed from
    ``styles.d/``.

    **Pure**: this returns the resolved list and nothing else. The
    ``--output FILE`` + multi-style mutex check lives in
    ``commands/generate._check_output_style_mutex`` since
    ``imgen batch`` has no ``--output`` flag and the check would be a
    silent no-op there. Pre-v0.3.1 the mutex check was inline here with
    a ``getattr(args, "output", None)`` guard — that worked but was
    surprising for batch readers; the split makes the generate-only
    nature explicit. (v0.3.0 architect NIT-4 / NIT-6.)
    """
    if args.style:
        return list(args.style)
    default_name = merged_defaults["style"]
    try:
        get_style(default_name)
    except KeyError:
        die(f"Default style '{default_name}' not found",
            code=2,
            hint="Check ~/.imgen/config.toml [defaults] style, "
                 "or run: imgen --list-styles")
    return [default_name]

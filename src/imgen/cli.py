"""
imgen — Photo style-transfer CLI for Apple Silicon Macs.

Uses mflux (MLX-native) under the hood. Default backend is FLUX Kontext Dev
(gated, requires HF token + license). Qwen-Image-Edit available as fallback.

Usage:
    imgen photo.jpg                              # default: pixar style
    imgen photo.jpg --style anime
    imgen photo.jpg --custom-prompt "..."
    imgen photo.jpg -s simpsons --steps 30 --strength 0.7
    imgen photo.jpg --backend qwen               # use Qwen Edit instead of FLUX

    imgen --list-styles
    imgen --dry-run photo.jpg --style anime

    imgen setup                                  # first-time install / token
    imgen doctor                                 # check environment
    imgen upgrade                                # update mflux
    imgen clean [--all]                          # cleanup HF cache
    imgen history [--last N]                     # show generation history
    imgen last                                   # repeat last generation
    imgen replay <id>                            # repeat generation by id
"""
from __future__ import annotations

import argparse
import datetime
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from . import __version__
from .backends import BACKENDS
from .checks import (
    check_disk_gb,
    check_mflux,
    check_pillow,
    check_resources,
    check_venv,
    find_running_mflux,
    get_battery,
    get_memory_gb,
)
from .colors import C, dim, die, err, info, ok, step, warn
from .defaults import (
    DEFAULTS,
    HISTORY_SCHEMA_VERSION,
    MFLUX_PIN,
    MIN_BATTERY_PCT,
    MIN_DISK_GB,
    PREVIEW_OVERRIDES,
    RAM_REQUIRED_GB,
)
from .history import append_history, load_history
from .images import apply_scope, detect_resolution
from .paths import (
    DEFAULT_OUTPUT_DIR,
    HF_CACHE,
    IMGEN_HOME,
    SAFE_OUTPUT_EXTS,
    STATE_DIR,
    TOKEN_FILE,
    VENV_BIN,
)
from .styles import STYLES, get_style, list_styles
from .subprocess_helpers import format_cmd, run_with_stderr_redaction
from .tokens import (
    check_token_perms,
    load_token,
    save_token_atomic,
    validate_token,
)


# ── Validators for argparse ──────────────────────────────────────────────

def _int_range(lo: int, hi: int):
    def validator(s: str) -> int:
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be an integer, got '{s}'")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"must be {lo}..{hi}, got {v}")
        return v
    return validator


def _float_range(lo: float, hi: float):
    def validator(s: str) -> float:
        try:
            v = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a number, got '{s}'")
        if not (lo <= v <= hi):
            raise argparse.ArgumentTypeError(f"must be {lo}..{hi}, got {v}")
        return v
    return validator


def _safe_output_path(s: str) -> str:
    """argparse validator: reject output paths with non-image extensions."""
    ext = Path(s).suffix.lower()
    if ext not in SAFE_OUTPUT_EXTS:
        raise argparse.ArgumentTypeError(
            f"output extension must be one of "
            f"{sorted(SAFE_OUTPUT_EXTS)}, got '{ext or '(none)'}'")
    return s


# ── Subcommand: doctor ───────────────────────────────────────────────────

def cmd_doctor(_args):
    issues = 0  # count blocking problems; return non-zero if any
    step("Checking environment")
    print()

    # Python
    py_version = sys.version.split()[0]
    if sys.version_info >= (3, 12):
        ok(f"Python {py_version}")
    else:
        warn(f"Python {py_version} (3.12+ recommended)")

    # venv
    venv_ok = check_venv()
    if venv_ok:
        ok(f"venv at {VENV_BIN.parent}")
    else:
        err(f"venv missing at {VENV_BIN.parent}")
        print(f"   {C.DIM}Run: imgen setup{C.END}")
        issues += 1

    # mflux (only if venv exists, otherwise can't even check)
    mflux_ver = check_mflux() if venv_ok else None
    if mflux_ver:
        ok(f"mflux {mflux_ver}")
    elif venv_ok:
        err("mflux not installed")
        print(f"   {C.DIM}Run: imgen setup  (or: imgen upgrade){C.END}")
        issues += 1

    # pillow
    pil_ver = check_pillow() if venv_ok else None
    if pil_ver:
        ok(f"Pillow {pil_ver}")
    elif venv_ok:
        warn("Pillow not installed (auto-aspect-ratio will fail)")

    # styles
    if STYLES:
        ok(f"Styles loaded: {', '.join(list_styles())}")
    else:
        err("styles registry is empty")
        issues += 1

    # disk
    free_gb = check_disk_gb()
    if free_gb >= 20:
        ok(f"Disk: {free_gb:.1f} GB free")
    elif free_gb >= MIN_DISK_GB:
        warn(f"Disk: {free_gb:.1f} GB free — low (need 20+ GB for FLUX download)")
    else:
        err(f"Disk: {free_gb:.1f} GB free — critically low")

    # System resources
    print()
    info("System resources right now")
    total_ram, available_ram = get_memory_gb()
    if total_ram:
        ok(f"RAM: {available_ram:.1f} GB available of {total_ram:.0f} GB")
    else:
        warn("RAM: couldn't read system memory info")

    battery_pct, on_ac = get_battery()
    if battery_pct is None:
        ok("Power: desktop / AC (no battery)")
    elif on_ac:
        ok(f"Battery: {battery_pct}% (plugged in)")
    elif battery_pct >= MIN_BATTERY_PCT:
        ok(f"Battery: {battery_pct}% (on battery)")
    else:
        warn(f"Battery: {battery_pct}% (on battery, low — plug in)")

    running_pid = find_running_mflux()
    if running_pid:
        warn(f"Another mflux is running (PID {running_pid}) — "
             "new runs will be blocked")
    else:
        ok("No other mflux process running")

    # RAM forecast per backend/quant
    print()
    info("Will this fit in RAM?")
    headers = ["backend × quant", "needs", "have", "verdict"]
    print(f"   {C.DIM}{headers[0]:<18} {headers[1]:>6}  "
          f"{headers[2]:>6}  {headers[3]}{C.END}")
    if total_ram:
        for (backend, q), need in sorted(RAM_REQUIRED_GB.items()):
            verdict = (f"{C.OK}✅ fits{C.END}" if available_ram >= need
                       else f"{C.ERR}❌ no{C.END}")
            label = f"{backend} q{q}"
            print(f"   {label:<18} {need:>4} GB  "
                  f"{available_ram:>4.1f} GB  {verdict}")

    # HF token
    print()
    info("Checking HuggingFace access")
    tok = load_token()
    if tok:
        ok(f"HF_TOKEN found ({tok[:8]}...{tok[-4:]})")
        if not check_token_perms():
            warn(f"~/.hf_token permissions not 600 — run: chmod 600 {TOKEN_FILE}")
    else:
        warn("No HF token found")
        print(f"   {C.DIM}FLUX backend won't work without token.{C.END}")
        print(f"   {C.DIM}Run: imgen setup  to configure.{C.END}")

    # HF cache models
    print()
    info("Cached models")
    cached_any = False
    if HF_CACHE.exists():
        for model_dir in sorted(HF_CACHE.glob("models--*")):
            try:
                # Count only blobs/ — snapshots/ are symlinks pointing here
                blobs = model_dir / "blobs"
                if blobs.exists():
                    size = sum(p.stat().st_size for p in blobs.iterdir()
                               if p.is_file() and not p.is_symlink())
                else:
                    size = sum(p.lstat().st_size for p in model_dir.rglob("*")
                               if p.is_file() and not p.is_symlink())
                size_gb = size / (1024 ** 3)
                if size_gb > 0.1:
                    name = model_dir.name.replace("models--", "").replace("--", "/")
                    ok(f"{name}: {size_gb:.1f} GB")
                    cached_any = True
            except OSError:
                pass
    if not cached_any:
        dim("   (no models cached yet — first run will download)")

    # Install mode
    print()
    info("Install mode")
    if IMGEN_HOME and (IMGEN_HOME / ".git").exists():
        ok(f"git checkout at {IMGEN_HOME} (use `imgen upgrade` for updates)")
    elif IMGEN_HOME:
        ok(f"unpacked at {IMGEN_HOME} (no git — manual reinstall to update)")
    else:
        ok("pipx install (use `pipx upgrade imgen` for updates)")

    print()
    if issues == 0 and mflux_ver and tok:
        step("Everything ready")
        return 0
    if issues > 0:
        step(f"{issues} issue(s) found — see ❌ above")
        return 1
    step("Some setup needed (see ⚠️  above)")
    return 0


# ── Subcommand: setup ────────────────────────────────────────────────────

def cmd_setup(_args):
    step("imgen auto-setup")
    print()

    # Apple Silicon check (MLX requires arm64)
    import platform
    if platform.system() != "Darwin":
        die(f"macOS only — detected {platform.system()}",
            code=3,
            hint="MLX (mflux backend) is Apple-only.")
    if platform.machine() != "arm64":
        die(f"Apple Silicon required — detected {platform.machine()}",
            code=3,
            hint="MLX does not support Intel Macs.")
    ok(f"macOS {platform.mac_ver()[0]} on {platform.machine()}")

    # venv + mflux: in v0.2 install mode is set up by either bootstrap.sh
    # (creates ~/imgen/.venv, pip install -e ., installs mflux from
    # pyproject deps) or by pipx (manages its own venv). So `imgen setup`
    # only verifies these and points to the right installer on failure.
    if not check_venv():
        if IMGEN_HOME:
            die("venv missing", code=3,
                hint=f"Run: {IMGEN_HOME / 'bootstrap.sh'}")
        else:
            die("venv missing", code=3,
                hint="Reinstall: pipx install --force git+https://github.com/BloamBla/imgen")
    ok(f"venv at {VENV_BIN.parent}")

    mflux_ver = check_mflux()
    if not mflux_ver:
        if IMGEN_HOME:
            die(f"mflux not installed in {VENV_BIN.parent}",
                code=3,
                hint=f"Run: {IMGEN_HOME / 'bootstrap.sh'}")
        else:
            die("mflux not installed (should have come with the pipx install)",
                code=3,
                hint="Reinstall: pipx install --force git+https://github.com/BloamBla/imgen")
    ok(f"mflux {mflux_ver}")

    # HF token
    print()
    if load_token():
        ok("HF token already configured")
    else:
        info("HuggingFace token setup (optional)")
        print(f"   {C.DIM}Token enables FLUX Kontext (best quality).{C.END}")
        print(f"   {C.DIM}Without token, only Qwen Edit backend works.{C.END}")
        print()
        print(f"   {C.BOLD}Get token:{C.END}")
        print(f"     1. https://huggingface.co/settings/tokens")
        print(f"        Create classic 'Read' token (NOT fine-grained).")
        print(f"     2. Accept FLUX license:")
        print(f"        https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev")
        print()
        try:
            tok = input("   Paste token (or Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            tok = ""

        if tok:
            if not tok.startswith("hf_"):
                warn("Token doesn't start with 'hf_' — saving anyway")
            try:
                if TOKEN_FILE.exists():
                    TOKEN_FILE.unlink()
                save_token_atomic(tok)
            except OSError as e:
                die(f"Couldn't write {TOKEN_FILE}: {e}", code=3)
            ok(f"Token saved to {TOKEN_FILE} (chmod 600)")
            user = validate_token(tok)
            if user:
                ok(f"Token valid (HF user: {user})")
            else:
                warn("Token saved but couldn't validate — could be invalid, "
                     "expired, or network issue. Check at: "
                     "https://huggingface.co/settings/tokens")
        else:
            dim("   Skipped. Run `imgen setup` later to add token.")

    # State dirs
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Shell alias — only for bootstrap-installed users. pipx users have
    # `imgen` in PATH via ~/.local/bin/ already; an alias would shadow.
    if IMGEN_HOME:
        print()
        info("Shell alias")
        shell_path = os.environ.get("SHELL", "")
        shell_name = Path(shell_path).name if shell_path else ""
        rc_files = {
            "zsh": Path.home() / ".zshrc",
            "bash": Path.home() / ".bash_profile",
            "fish": Path.home() / ".config" / "fish" / "config.fish",
        }
        rc_file = rc_files.get(shell_name)
        # shlex.quote() safely escapes paths containing quotes, spaces,
        # $, ;, etc. so a repo cloned into a weird directory can't inject
        # shell code.
        alias_line = f"alias imgen={shlex.quote(str(IMGEN_HOME / 'imgen'))}"

        if rc_file is None:
            warn(f"Unknown shell '{shell_name}' — skipping alias setup")
            print(f"   {C.DIM}Add manually to your shell rc: {alias_line}{C.END}")
        else:
            try:
                existing = rc_file.read_text() if rc_file.exists() else ""
            except OSError:
                existing = ""
            if alias_line in existing:
                ok(f"Alias already in {rc_file}")
            else:
                try:
                    rc_file.parent.mkdir(parents=True, exist_ok=True)
                    with rc_file.open("a") as f:
                        f.write(f"\n# imgen — photo style transfer\n{alias_line}\n")
                    ok(f"Added alias to {rc_file}")
                    print(f"   {C.DIM}Restart terminal or: source {rc_file}{C.END}")
                except OSError as e:
                    warn(f"Couldn't write {rc_file}: {e}")
                    print(f"   {C.DIM}Add manually: {alias_line}{C.END}")
    else:
        print()
        ok("Pipx install — `imgen` already in your PATH, no alias needed")

    print()
    step("Setup complete!")
    print(f"   {C.DIM}Try: imgen photo.jpg{C.END}")
    return 0


# ── Subcommand: clean ────────────────────────────────────────────────────

def cmd_clean(args):
    step("Cleaning HuggingFace cache")
    print()

    if not HF_CACHE.exists():
        ok("Cache is empty")
        return 0

    # Always: delete .incomplete files older than 24h
    cutoff = datetime.datetime.now().timestamp() - 24 * 3600
    incomplete_removed = 0
    incomplete_size = 0
    for blob in HF_CACHE.rglob("*.incomplete"):
        try:
            if blob.stat().st_mtime < cutoff:
                incomplete_size += blob.stat().st_size
                blob.unlink()
                incomplete_removed += 1
        except OSError:
            pass
    if incomplete_removed:
        ok(f"Removed {incomplete_removed} stale .incomplete files "
           f"({incomplete_size / (1024**3):.1f} GB)")
    else:
        dim("No stale .incomplete files to remove")

    if args.all:
        models = sorted(HF_CACHE.glob("models--*"))
        if not models:
            ok("No cached models to remove")
            return 0
        print()
        info("Cached models to be deleted:")
        total = 0
        for m in models:
            try:
                blobs = m / "blobs"
                if blobs.exists():
                    size = sum(p.stat().st_size for p in blobs.iterdir()
                               if p.is_file() and not p.is_symlink())
                else:
                    size = sum(p.lstat().st_size for p in m.rglob("*")
                               if p.is_file() and not p.is_symlink())
                total += size
                name = m.name.replace("models--", "").replace("--", "/")
                print(f"   • {name}: {size / (1024**3):.1f} GB")
            except OSError:
                pass
        print(f"   {C.BOLD}Total: {total / (1024**3):.1f} GB{C.END}")
        print()
        if args.dry_run:
            warn("Dry run — nothing deleted")
            return 0
        try:
            confirm = input("   Delete all? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            confirm = ""
        if confirm != "y":
            dim("   Cancelled")
            return 0
        # Don't follow a symlinked model dir — its target may be outside
        # HF_CACHE (e.g. shared models tree on another volume). rmtree would
        # silently delete the target. Internal symlinks (HF's snapshots/ →
        # blobs/) stay safe — rmtree unlinks them, doesn't follow.
        cleanup_errors: list[tuple[str, BaseException]] = []

        def _log_rmtree_error(func, path, exc):
            cleanup_errors.append((str(path), exc))

        deleted = 0
        skipped_symlinks = []
        for m in models:
            if m.is_symlink():
                skipped_symlinks.append(m)
                continue
            if not m.is_dir():
                continue
            shutil.rmtree(m, onexc=_log_rmtree_error)
            deleted += 1

        if skipped_symlinks:
            warn(f"Skipped {len(skipped_symlinks)} symlinked model dir(s) "
                 f"— refusing to follow:")
            for s in skipped_symlinks:
                print(f"   {C.DIM}• {s}{C.END}")
        if cleanup_errors:
            warn(f"{len(cleanup_errors)} error(s) during deletion:")
            for path, exc in cleanup_errors[:5]:
                print(f"   {C.DIM}• {path}: {exc}{C.END}")
            if len(cleanup_errors) > 5:
                more = len(cleanup_errors) - 5
                print(f"   {C.DIM}  ... and {more} more{C.END}")

        ok(f"Deleted {deleted} model(s) "
           f"({total / (1024**3):.1f} GB freed)")

    return 0


# ── Subcommand: upgrade ──────────────────────────────────────────────────

def _self_update() -> None:
    """git pull --ff-only in IMGEN_HOME, then re-install the package into
    the venv so the new code is actually loaded next run.

    Warns (doesn't fail) on any issue so the mflux-upgrade step still runs.
    For pipx-installed users IMGEN_HOME is None — print the right command
    and return.
    """
    if IMGEN_HOME is None:
        ok("Pipx install detected — use `pipx upgrade imgen` instead")
        return

    if not (IMGEN_HOME / ".git").exists():
        warn(f"{IMGEN_HOME} is not a git checkout — self-update unavailable")
        print(f"   {C.DIM}For auto-updates, reinstall via:{C.END}")
        print(f"   {C.DIM}  rm -rf {IMGEN_HOME} && "
              f"git clone https://github.com/BloamBla/imgen "
              f"{IMGEN_HOME}{C.END}")
        return

    info(f"Pulling latest from {IMGEN_HOME}")
    try:
        before = subprocess.check_output(
            ["git", "-C", str(IMGEN_HOME), "rev-parse", "HEAD"],
            text=True, timeout=10,
        ).strip()[:7]
        result = subprocess.run(
            ["git", "-C", str(IMGEN_HOME), "pull", "--ff-only"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            warn(f"git pull failed: {result.stderr.strip().splitlines()[-1]}")
            print(f"   {C.DIM}Resolve manually: "
                  f"cd {IMGEN_HOME} && git status{C.END}")
            print(f"   {C.DIM}Continuing with mflux upgrade.{C.END}")
            return
        after = subprocess.check_output(
            ["git", "-C", str(IMGEN_HOME), "rev-parse", "HEAD"],
            text=True, timeout=10,
        ).strip()[:7]
        if before == after:
            ok("imgen already up to date")
            return
        ok(f"imgen updated: {before} → {after}")
        try:
            log = subprocess.check_output(
                ["git", "-C", str(IMGEN_HOME), "log",
                 f"{before}..{after}", "--oneline", "--no-decorate"],
                text=True, timeout=10,
            ).strip()
            if log:
                print(f"   {C.DIM}New commits:{C.END}")
                for line in log.splitlines()[:10]:
                    print(f"   {C.DIM}  • {line}{C.END}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        warn(f"git operation failed: {e}")
        return

    # Re-install the package so the venv picks up new code. Required since
    # v0.2 — before the split, `imgen` was a standalone script and git pull
    # was enough.
    pip = VENV_BIN / "pip"
    if not pip.exists():
        warn(f"pip not found at {pip} — skipping package reinstall")
        print(f"   {C.DIM}Re-run: {IMGEN_HOME / 'bootstrap.sh'}{C.END}")
        return
    try:
        subprocess.check_call(
            [str(pip), "install", "--quiet", "-e", str(IMGEN_HOME)],
            timeout=120,
        )
        ok("imgen package reinstalled into venv")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        warn(f"pip install -e failed: {e}")
        print(f"   {C.DIM}Re-run: {IMGEN_HOME / 'bootstrap.sh'}{C.END}")


def cmd_upgrade(args):
    step("Upgrading imgen (self-update)")
    _self_update()
    print()

    step("Upgrading mflux")
    if not check_venv():
        die("venv missing — run: imgen setup", code=3)

    if args.latest:
        target = "mflux"
        warn("--latest: installing newest mflux — may have breaking changes "
             f"vs pinned {MFLUX_PIN}")
    else:
        target = MFLUX_PIN
        info(f"Installing pinned {MFLUX_PIN} (use --latest to skip pin)")

    pip = VENV_BIN / "pip"
    try:
        subprocess.check_call(
            [str(pip), "install", "--upgrade", target]
        )
        ok(f"mflux is now {check_mflux()}")
    except subprocess.CalledProcessError as e:
        die(f"mflux upgrade failed: {e}", code=3)
    return 0


# ── Subcommand: history / last / replay ──────────────────────────────────

def cmd_history(args):
    entries = load_history()
    if not entries:
        dim("No history yet")
        return 0
    n = max(1, args.last or 20)
    for entry in entries[-n:]:
        status_icon = "✅" if entry.get("status") == "success" else "❌"
        ts = entry.get("ts", "?")[:16].replace("T", " ")
        eid = str(entry.get("id", "?"))
        style = entry.get("style") or "custom"
        print(f"{C.DIM}#{eid:<4}{C.END} {status_icon} "
              f"{C.BOLD}{ts}{C.END}  "
              f"{C.INFO}{style:10}{C.END}  "
              f"{Path(entry.get('input', '?')).name:30}  "
              f"→ {Path(entry.get('output', '?')).name}")
    return 0


def cmd_last(_args):
    entries = load_history()
    if not entries:
        die("No history yet", code=1)
    last = entries[-1]
    return replay_entry(last)


def cmd_replay(args):
    entries = load_history()
    target = next((e for e in entries if e.get("id") == args.id), None)
    if not target:
        die(f"No entry with id {args.id}", code=1)
    return replay_entry(target)


def replay_entry(entry: dict) -> int:
    entry_v = entry.get("v", 0)
    if entry_v > HISTORY_SCHEMA_VERSION:
        die(f"History entry #{entry.get('id', '?')} is from a newer schema "
            f"(v{entry_v} > v{HISTORY_SCHEMA_VERSION}). "
            f"Run `imgen upgrade` to pick up the new fields.", code=2)
    image = entry.get("input")
    if not image:
        die(f"History entry #{entry.get('id', '?')} has no input path — "
            f"cannot replay.", code=1)
    info(f"Replaying #{entry.get('id')}: {entry.get('style')} on "
         f"{Path(image).name}")
    args = argparse.Namespace(
        image=image,
        style=entry.get("style", "pixar") if not entry.get("custom_prompt") else None,
        custom_prompt=entry.get("custom_prompt"),
        scope=entry.get("scope"),
        preview=entry.get("preview", False),
        output=None,  # auto-generate new output name
        steps=entry.get("steps", DEFAULTS["steps"]),
        guidance=entry.get("guidance", DEFAULTS["guidance"]),
        strength=entry.get("strength", DEFAULTS["strength"]),
        seed=None,  # new random seed
        backend=entry.get("backend", DEFAULTS["backend"]),
        quantize=entry.get("quantize", DEFAULTS["quantize"]),
        width=entry.get("width"),
        height=entry.get("height"),
        no_open=False,
        dry_run=False,
        force=False,
    )
    return cmd_generate(args)


# ── Subcommand: generate (main) ──────────────────────────────────────────

def cmd_generate(args):
    # 1) Validate input
    input_path = Path(args.image).expanduser().resolve()
    if not input_path.exists():
        die(f"Image not found: {input_path}",
            code=2,
            hint="Check the path. Use absolute path if unsure.")
    if not input_path.is_file():
        die(f"Not a file: {input_path}", code=2)

    # 2) Validate style vs custom-prompt
    if args.style and args.custom_prompt:
        die("Cannot use both --style and --custom-prompt; pick one",
            code=2)

    # 3) Build prompt
    preset: dict | None = None
    if args.custom_prompt:
        prompt = args.custom_prompt
        negative = ""
        style_name = "custom"
    else:
        style_name = args.style or DEFAULTS["style"]
        try:
            preset = get_style(style_name)
        except KeyError as e:
            name = e.args[0] if e.args else str(e)
            die(f"Unknown style: {name}",
                code=2, hint="See: imgen --list-styles")
        prompt = preset["prompt"]
        negative = preset.get("negative", "")

    # 3a) Apply --scope (warn if combined with --custom-prompt)
    if args.scope:
        if args.custom_prompt:
            warn(f"--scope={args.scope} ignored when using --custom-prompt")
        else:
            prompt = apply_scope(prompt, args.scope)

    # 3b) Resolve final parameter values:
    #   CLI flag (if set) > style preset > preview override > global default
    if args.steps is not None:
        final_steps = args.steps
    elif args.preview:
        final_steps = PREVIEW_OVERRIDES["steps"]
    else:
        final_steps = DEFAULTS["steps"]

    if args.quantize is not None:
        final_quantize = args.quantize
    elif args.preview:
        final_quantize = PREVIEW_OVERRIDES["quantize"]
    else:
        final_quantize = DEFAULTS["quantize"]

    if args.guidance is not None:
        final_guidance = args.guidance
    elif preset and "guidance" in preset:
        final_guidance = preset["guidance"]
    else:
        final_guidance = DEFAULTS["guidance"]

    if args.strength is not None:
        final_strength = args.strength
    elif preset and "strength" in preset:
        final_strength = preset["strength"]
    else:
        final_strength = DEFAULTS["strength"]

    # 4) Resolution
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = detect_resolution(input_path, preview=args.preview)

    # 5) Output path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_{style_name}_{ts}.png"

    # 6) Backend & token
    backend = args.backend
    if backend == "flux":
        token = load_token()
        if not token:
            die("FLUX backend requires HuggingFace token",
                code=3,
                hint="Run: imgen setup   (or use --backend qwen)")
    else:
        token = None  # qwen-image-edit is open

    # 7) Build mflux command
    if not check_venv() or not check_mflux():
        die("mflux not installed",
            code=3,
            hint="Run: imgen setup")

    binary = VENV_BIN / BACKENDS[backend]
    if not binary.exists():
        die(f"Backend binary not found: {binary}",
            code=3,
            hint="Run: imgen upgrade")

    seed = (args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), "big"))

    cmd = [
        str(binary),
        "--quantize", str(final_quantize),
        ("--image-path" if backend == "flux" else "--image-paths"), str(input_path),
        "--prompt", prompt,
        "--steps", str(final_steps),
        "--guidance", str(final_guidance),
        "--seed", str(seed),
        "--width", str(width),
        "--height", str(height),
        "--mlx-cache-limit-gb", str(DEFAULTS["mlx_cache_gb"]),
        "--battery-percentage-stop-limit", str(DEFAULTS["battery_stop"]),
        "--metadata",
        "--output", str(output_path),
    ]
    # FLUX supports --image-strength and --negative-prompt; qwen-edit doesn't
    if backend == "flux":
        cmd += ["--image-strength", str(final_strength), "--model", "dev"]
        if negative:
            cmd += ["--negative-prompt", negative]
    else:
        cmd += ["--model", "qwen"]

    # 8) Dry run (skip resource checks — just show what would run)
    if args.dry_run:
        step("Dry run — would execute:")
        print()
        print(format_cmd(cmd))
        print()
        return 0

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
        "style": style_name if not args.custom_prompt else None,
        "custom_prompt": args.custom_prompt,
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


# ── CLI parser ───────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="imgen",
        description="Photo style transfer for Apple Silicon Macs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[1] if __doc__ else None,
    )

    # Top-level utility flags
    p.add_argument("--list-styles", action="store_true",
                   help="List style presets and exit")
    p.add_argument("--version", action="version",
                   version=f"imgen {__version__}")

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # setup / doctor / upgrade
    sub.add_parser("setup", help="First-time install & token setup")
    sub.add_parser("doctor", help="Check environment & cached models")
    u = sub.add_parser("upgrade", help=f"Upgrade mflux to pinned {MFLUX_PIN}")
    u.add_argument("--latest", action="store_true",
                   help="Install newest mflux instead of pinned version "
                        "(may have breaking changes)")

    # clean
    c = sub.add_parser("clean", help="Cleanup HuggingFace cache")
    c.add_argument("--all", action="store_true",
                   help="Also delete cached models (with confirmation)")
    c.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted without deleting")

    # history / last / replay
    h = sub.add_parser("history", help="Show generation history")
    h.add_argument("--last", type=int, default=20,
                   help="Show last N entries (default 20)")
    sub.add_parser("last", help="Repeat last generation with new seed")
    r = sub.add_parser("replay", help="Repeat generation by id")
    r.add_argument("id", type=int)

    # generate (default — no subcommand, positional image)
    g = sub.add_parser("generate",
                       help="Generate styled image (default command)")
    _add_generate_args(g)

    return p


def _add_generate_args(p):
    p.add_argument("image", help="Path to input photo")
    p.add_argument("-s", "--style", choices=list_styles(),
                   help=f"Style preset (default: {DEFAULTS['style']})")
    p.add_argument("--custom-prompt",
                   help="Custom prompt (overrides --style)")
    p.add_argument("-o", "--output", type=_safe_output_path,
                   help=f"Output path with .png/.jpg/.jpeg/.webp suffix "
                        f"(default: {DEFAULT_OUTPUT_DIR}/<auto>.png)")
    # Override args use default=None so we can tell "user set" from "use default"
    p.add_argument("--steps", type=_int_range(1, 200), default=None,
                   help=f"Inference steps 1..200 (default {DEFAULTS['steps']}, "
                        f"preview {PREVIEW_OVERRIDES['steps']})")
    p.add_argument("-g", "--guidance", type=_float_range(0.5, 15.0), default=None,
                   help=f"Guidance scale 0.5..15 (default {DEFAULTS['guidance']}, "
                        "style preset may override)")
    p.add_argument("--strength", type=_float_range(0.0, 1.0), default=None,
                   help=f"Image strength 0..1 (default {DEFAULTS['strength']}, "
                        "style preset may override)")
    p.add_argument("--seed", type=_int_range(0, 2**32 - 1),
                   help="Seed (default: random)")
    p.add_argument("--backend", choices=list(BACKENDS), default=DEFAULTS["backend"],
                   help=f"Backend (default {DEFAULTS['backend']})")
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8],
                   default=None,
                   help=f"Quantization (default {DEFAULTS['quantize']}, "
                        f"preview {PREVIEW_OVERRIDES['quantize']})")
    p.add_argument("--scope", choices=["person", "scene"],
                   help="person=transform person only (keep background); "
                        "scene=transform whole image; default=balanced subject focus")
    p.add_argument("-p", "--preview", action="store_true",
                   help="Fast preview mode: smaller resolution, fewer steps, "
                        "lower quantization (~5x faster, lower quality)")
    p.add_argument("--width", type=_int_range(64, 4096),
                   help="Override output width (64..4096)")
    p.add_argument("--height", type=_int_range(64, 4096),
                   help="Override output height (64..4096)")
    p.add_argument("--no-open", action="store_true",
                   help="Don't open result in Preview")
    p.add_argument("--dry-run", action="store_true",
                   help="Show mflux command without running")
    p.add_argument("--force", action="store_true",
                   help="Skip resource checks (RAM, parallel mflux, etc.) "
                        "and try anyway. Use at your own risk.")


def _print_styles() -> int:
    step("Available styles")
    for name in list_styles():
        preset = STYLES[name]
        print(f"  {C.BOLD}{name:10}{C.END} "
              f"{C.DIM}(guidance={preset.get('guidance')}, "
              f"strength={preset.get('strength')}){C.END}")
        print(f"             {preset['prompt'][:80]}...")
    return 0


def main():
    # If the FIRST non-flag arg isn't a known subcommand, prepend "generate".
    # Only checking the first positional avoids two prior pitfalls:
    #   - a path like "last.jpg" being mistaken for the "last" subcommand
    #   - an --option value that happens to match a subcommand name
    #     blocking the shorthand dispatch
    argv = sys.argv[1:]
    known = {"setup", "doctor", "upgrade", "clean",
             "history", "last", "replay", "generate"}
    first_positional = next((a for a in argv if not a.startswith("-")), None)
    if first_positional and first_positional not in known:
        argv = ["generate"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    # Top-level info actions: handled before subcommand dispatch
    if getattr(args, "list_styles", False):
        return _print_styles()

    handlers = {
        "setup": cmd_setup,
        "doctor": cmd_doctor,
        "upgrade": cmd_upgrade,
        "clean": cmd_clean,
        "history": cmd_history,
        "last": cmd_last,
        "replay": cmd_replay,
        "generate": cmd_generate,
    }

    if not args.command:
        parser.print_help()
        return 0

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        return 2

    # Graceful SIGINT
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        return handler(args) or 0
    except KeyboardInterrupt:
        print()
        warn("Cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

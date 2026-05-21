"""`imgen doctor` — environment + resource forecast + cached models report."""
from __future__ import annotations

import sys

from ..checks import (
    check_disk_gb,
    check_mflux,
    check_pillow,
    check_venv,
    find_running_mflux,
    get_battery,
    get_memory_gb,
)
from ..colors import C, dim, err, info, ok, step, warn
from ..config import ConfigError, load_validated_config
from ..defaults import MIN_BATTERY_PCT, MIN_DISK_GB, RAM_REQUIRED_GB
from ..paths import (
    CONFIG_FILE,
    HF_CACHE,
    IMGEN_HOME,
    STATE_DIR,
    TOKEN_FILE,
    VENV_BIN,
)
from ..styles import BUILTIN_STYLES, list_styles, load_user_styles_dir
from ..tokens import check_token_perms, load_token


def cmd_doctor(_args) -> int:
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

    # styles (built-in + user-supplied TOMLs)
    if BUILTIN_STYLES:
        ok(f"Built-in styles: {', '.join(sorted(BUILTIN_STYLES.keys()))}")
    else:
        err("Built-in styles registry is empty")
        issues += 1
    user_styles_dir = STATE_DIR / "styles.d"
    user_styles = load_user_styles_dir(user_styles_dir)
    if user_styles:
        ok(f"User styles ({user_styles_dir}): "
           f"{', '.join(sorted(user_styles.keys()))}")
    else:
        dim(f"   (no user styles in {user_styles_dir})")

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

    # User config
    print()
    info("User config")
    if not CONFIG_FILE.exists():
        dim(f"   (no {CONFIG_FILE} — using built-in defaults)")
        dim("   Run `imgen setup` to create a starter template.")
    else:
        try:
            cfg = load_validated_config(CONFIG_FILE)
            n_defaults = len(cfg["defaults"])
            n_ui = len(cfg["ui"])
            ok(f"{CONFIG_FILE}: {n_defaults} default(s), {n_ui} ui setting(s)")
            for k, v in cfg["defaults"].items():
                print(f"   {C.DIM}[defaults] {k} = {v!r}{C.END}")
            for k, v in cfg["ui"].items():
                print(f"   {C.DIM}[ui] {k} = {v!r}{C.END}")
        except ConfigError as e:
            err(f"{CONFIG_FILE}: {e}")
            issues += 1

    print()
    if issues == 0 and mflux_ver and tok:
        step("Everything ready")
        return 0
    if issues > 0:
        step(f"{issues} issue(s) found — see ❌ above")
        return 1
    step("Some setup needed (see ⚠️  above)")
    return 0

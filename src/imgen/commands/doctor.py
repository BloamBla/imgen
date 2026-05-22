"""`imgen doctor` — environment + resource forecast + cached models report."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from pathlib import Path

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
    VENV_BIN,
)
from ..backends import BUILTIN_BACKENDS, get_backend, list_backends
from ..shell_rc import ALL_RC_FILES_REL
from ..styles import BUILTIN_STYLES, list_styles, load_user_styles_dir
from ..tokens import active_token_path, check_token_perms, load_token

# Matches `alias imgen=<value>` at line start (allowing leading
# whitespace). The value is captured greedily to end-of-line and then
# unwrapped with shlex.split so quoting / trailing comments are handled
# the same way the shell would.
_ALIAS_RE = re.compile(r"^\s*alias\s+imgen\s*=\s*(\S.*?)\s*$", re.MULTILINE)


def parse_imgen_alias(rc_content: str) -> Path | None:
    """Extract the path the last ``alias imgen=...`` line points to.

    Returns None if no such alias is present, or the value can't be
    parsed (malformed quoting). The last match wins to match shell
    semantics — a later definition in the file overrides earlier ones.

    Pure function for testability; the doctor wrapper iterates rc files
    and prints. shlex.split with ``comments=True`` handles both the
    shlex.quote output that ``setup.py`` writes (single-quoted on
    paths with spaces, bare otherwise) and trailing `# comments`.
    """
    matches = _ALIAS_RE.findall(rc_content)
    if not matches:
        return None
    try:
        tokens = shlex.split(matches[-1], comments=True)
    except ValueError:
        return None
    if not tokens:
        return None
    return Path(tokens[0])


def check_alias_consistency(
    home: Path,
    imgen_home: Path | None,
) -> list[tuple[Path, Path, str]]:
    """Report each shell rc file's alias status vs the expected path.

    Pure function: returns a list of ``(rc_file, aliased_path, status)``
    tuples where status ∈ ``{"match", "mismatch", "unparsable"}``. The
    doctor printer turns these into ok/warn lines. RC files with no
    ``alias imgen=`` line are simply not included.

    Returns ``[]`` for pipx mode (``imgen_home=None``) — no alias is
    ever written there, so divergence is impossible by construction.
    (architect #1 from v0.1.x review — covers the "user moved the
    repo directory after bootstrap.sh" footgun.)
    """
    if imgen_home is None:
        return []
    expected = (imgen_home / "imgen").resolve()
    results: list[tuple[Path, Path, str]] = []
    for rel in ALL_RC_FILES_REL:
        rc = home / rel
        if not rc.exists():
            continue
        try:
            content = rc.read_text(errors="replace")
        except OSError:
            continue
        aliased = parse_imgen_alias(content)
        if aliased is None:
            continue
        # resolve() collapses /Users/foo/./imgen vs /Users/foo/imgen and
        # follows symlinks — both legitimate forms of the same target
        # are treated as a match.
        try:
            aliased_resolved = aliased.resolve()
        except OSError:
            results.append((rc, aliased, "unparsable"))
            continue
        status = "match" if aliased_resolved == expected else "mismatch"
        results.append((rc, aliased, status))
    return results


def detect_install_collision(
    home: Path,
    imgen_home: Path | None,
) -> str | None:
    """Return a warning string if both bootstrap and pipx imgen installs
    are present on the system. Pure / injectable for tests.

    A colleague who ran both `bootstrap.sh` and `pipx install imgen`
    ends up with `~/imgen/.venv/bin/imgen` (used by the alias-pointed
    shim) AND `~/.local/bin/imgen` (used by raw shell PATH). Either
    works, but they may have diverged versions if the user upgraded
    one without the other. Surface it so the user can pick one.
    """
    if imgen_home is None:
        return None  # pipx-only mode — no collision possible
    bootstrap_imgen = imgen_home / ".venv" / "bin" / "imgen"
    pipx_imgen = home / ".local" / "bin" / "imgen"
    if bootstrap_imgen.exists() and pipx_imgen.exists():
        return (
            f"Both install paths present: bootstrap ({bootstrap_imgen}) "
            f"AND pipx ({pipx_imgen}). They can diverge in version. "
            "Pick one: `pipx uninstall imgen` to drop the pipx copy, "
            "or `rm -rf ~/imgen` to drop the bootstrap one."
        )
    return None


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
        if os.environ.get("HF_TOKEN"):
            dim("   source: $HF_TOKEN env")
        else:
            active = active_token_path()
            if active is not None:
                dim(f"   source: {active}")
                # No second warn about legacy path here — _try_migrate_legacy
                # (called by load_token above) already printed its own
                # remediation hint with full context.
                if not check_token_perms():
                    warn(f"{active} permissions not 600 — run: chmod 600 {active}")
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
    collision = detect_install_collision(Path.home(), IMGEN_HOME)
    if collision:
        warn(collision)
    # Bootstrap-mode users get an alias in their shell rc that points at
    # `IMGEN_HOME/imgen`. If the user moved the repo after running
    # `bootstrap.sh`, the alias silently still points at the old location
    # and runs stale code on every invocation. Surface divergence here
    # so the fix (`imgen setup` to rewrite the alias) is obvious.
    # (architect #1 from v0.1.x review.)
    alias_results = check_alias_consistency(Path.home(), IMGEN_HOME)
    for rc, aliased, status in alias_results:
        # !r-format the path string so any C0/DEL/C1 control bytes that
        # rode along from the rc file (e.g. ANSI escapes embedded by a
        # different account writing to an NFS-shared $HOME) render as
        # \x1b literals instead of escaping into the terminal. Mirrors
        # the `_is_safe_stem` defence on style filenames in styles.py.
        # (v0.3.6 security-reviewer NIT.)
        aliased_safe = repr(str(aliased))
        if status == "match":
            ok(f"shell alias in {rc.name} matches IMGEN_HOME")
        elif status == "mismatch":
            warn(f"shell alias in {rc.name} points at {aliased_safe}, but "
                 f"IMGEN_HOME is {IMGEN_HOME} — stale alias from a "
                 f"previous install. Run `imgen setup` to refresh.")
            issues += 1
        else:  # "unparsable" — broken symlink in alias path; rare
            warn(f"shell alias in {rc.name} points at {aliased_safe} which "
                 "couldn't be resolved (broken symlink?). "
                 "Run `imgen setup` to refresh.")
            issues += 1

    # Backends (built-in + user TOMLs from ~/.imgen/backends.d/)
    # v0.4 — surfaces binary-on-disk + secret-env-var status per
    # backend so the user knows up-front whether `imgen --backend X`
    # will actually launch. Mirrors the existing per-backend RAM
    # forecast section earlier, but one rung lower (file resolution +
    # env, not memory).
    print()
    info("Backends")
    for name in list_backends():
        be = get_backend(name)
        origin = "built-in" if name in BUILTIN_BACKENDS else "custom"
        # Binary resolution — branch on shape matches the actual
        # resolution in cmd_helpers.load_backend_and_token. Absolute
        # paths used as-is, bare names live in VENV_BIN (mflux
        # convention).
        if be.binary.startswith("/"):
            binary_path = Path(be.binary)
        else:
            binary_path = VENV_BIN / be.binary
        if binary_path.exists():
            ok(f"{name} ({origin}): {binary_path}")
        else:
            warn(f"{name} ({origin}): binary not found at {binary_path}"
                 + ("" if origin == "built-in" else " — fix backends.d "
                    "TOML or install the binary"))
            issues += 1
        # Secret env var status (only declared on custom backends).
        if be.secret_env_var is not None:
            value = os.environ.get(be.secret_env_var)
            if value:
                ok(f"   secret ${be.secret_env_var} set")
            elif be.secret_required:
                warn(f"   secret ${be.secret_env_var} (required) NOT set in "
                     f"environment — `imgen --backend {name}` will die")
                issues += 1
            else:
                dim(f"   secret ${be.secret_env_var} (optional) not set — "
                    "best-effort forward, backend handles its own auth")

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

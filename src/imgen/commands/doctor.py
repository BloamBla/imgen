"""`imgen doctor` — environment + resource forecast + cached models report."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
from ..config import (
    ConfigError,
    effective_enhance,
    load_validated_config,
)
from ..defaults import MIN_BATTERY_PCT, MIN_DISK_GB, RAM_REQUIRED_GB
from ..history import load_history
from ..paths import (
    CONFIG_FILE,
    HF_CACHE,
    IMGEN_HOME,
    STATE_DIR,
    STYLES_D,
    VENV_BIN,
)
from ..backends import BUILTIN_BACKENDS, get_backend, list_backends
from ..shell_rc import ALL_RC_FILES_REL
from ..styles import BUILTIN_STYLES, list_styles, load_user_styles_dir
from ..tokens import (
    active_token_path,
    check_token_perms,
    load_token,
    safe_display_username,
)

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


@dataclass(frozen=True, slots=True)
class BackendHealth:
    """One backend's binary + secret-env status, for doctor reporting.

    Pure-data result of :func:`check_backend_health`. ``binary_ok`` and
    ``secret_present`` are the two health gates; ``cmd_doctor`` turns
    each into an ok/warn line. ``secret_present`` is None when no
    ``[secret]`` section is declared (don't report on what wasn't
    asked for).
    """
    name: str
    origin: str               # "built-in" | "custom"
    binary_path: Path
    binary_ok: bool           # is_file() at check time
    secret_env_var: str | None
    secret_present: bool | None
    secret_required: bool


def check_backend_health(
    *,
    venv_bin: Path,
    env: Mapping[str, str],
) -> list[BackendHealth]:
    """Iterate the merged backend registry, classify each entry.

    Pure (no print, no warn) — the doctor printer wraps results into
    ok/warn lines and bumps the issues counter. Extracted from inline
    cmd_doctor v0.4 code per architect IMP-5 from the v0.4 review:
    same shape as ``check_alias_consistency`` (v0.3.6), restores
    symmetry between the two diagnostic surfaces and makes the
    "binary not found" / "required env var missing" paths
    independently testable.

    Args:
        venv_bin: where bare-name binaries are resolved (production
                  passes :data:`paths.VENV_BIN`; tests pass a tmp dir).
        env:      env mapping checked for declared secret vars
                  (production passes ``os.environ``; tests pass a
                  dict).

    Returns one :class:`BackendHealth` per merged backend, in
    ``list_backends()`` order (sorted by name).
    """
    results: list[BackendHealth] = []
    for name in list_backends():
        be = get_backend(name)
        origin = "built-in" if name in BUILTIN_BACKENDS else "custom"
        if be.binary.startswith("/"):
            binary_path = Path(be.binary)
        else:
            binary_path = venv_bin / be.binary
        binary_ok = binary_path.is_file()

        secret_present: bool | None = None
        if be.secret_env_var is not None:
            secret_present = bool(env.get(be.secret_env_var))

        results.append(BackendHealth(
            name=name,
            origin=origin,
            binary_path=binary_path,
            binary_ok=binary_ok,
            secret_env_var=be.secret_env_var,
            secret_present=secret_present,
            secret_required=be.secret_required,
        ))
    return results


@dataclass(frozen=True, slots=True)
class EnhanceHealth:
    """v0.5 LLM prompt enhancer status report, for doctor.

    Pure-data result of :func:`check_enhance_health`. ``mlx_lm_importable``
    is the dependency gate (if mlx-lm isn't installed, the enhancer
    cannot run at all). ``model_cached`` + ``model_cache_size_bytes``
    tell the user whether the first-time download is still pending
    (~4 GB for Qwen2.5-7B-Instruct-4bit, ~7 minutes unauthenticated).
    ``recent_runs`` / ``recent_runs_succeeded`` summarise the last 10
    enhancer-aware history entries so a degrading invariant or
    persistent runner error surfaces in routine `imgen doctor` runs.
    """
    mlx_lm_importable: bool
    enabled_by_default: bool   # config [enhance] default = true
    model_ref: str             # configured HF repo or local path
    model_cached: bool         # ``~/.cache/huggingface/hub/`` has it
    model_cache_size_bytes: int | None
    recent_runs: int           # enhancer-aware history entries in last 10
    recent_runs_succeeded: int  # of those, how many enhanced=True


from ..hf_cache import hf_cache_dir_for


def _dir_size_bytes(p: Path) -> int:
    """Total size of regular files under ``p`` (recursive). Returns 0
    if ``p`` is missing or not a directory. Symlinks are NOT followed
    to avoid double-counting HF cache's snapshots/<sha>/* → blobs/*
    indirection (snapshots are symlinks; blobs are real files)."""
    if not p.is_dir():
        return 0
    total = 0
    for entry in p.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue  # transient race or perms — don't crash doctor
    return total


def check_enhance_health(
    *,
    enhance_cfg: dict,
    hf_cache: Path,
    history: list[dict] | None = None,
    importable: bool | None = None,
) -> EnhanceHealth:
    """Inspect the v0.5 enhancer's readiness state. Pure-ish (one
    optional ``import mlx_lm`` attempt; otherwise no I/O beyond
    stat'ing the HF cache directory) — the doctor printer wraps the
    result into ok/warn lines.

    Args:
        enhance_cfg: validated ``[enhance]`` section dict (may be
                     empty when config.toml is missing; falls back to
                     module defaults via ``effective_enhance``).
        hf_cache:    ``HF_CACHE`` in prod, tmp dir in tests.
        history:     pre-loaded history entries (production calls
                     ``load_history()``; tests inject canned lists to
                     verify the recent-success counter).
        importable:  override for the mlx-lm import probe. None →
                     attempt ``import mlx_lm`` and set based on success.
                     Tests pass True/False directly to avoid touching
                     the real mlx_lm install state.
    """
    if importable is None:
        try:
            import mlx_lm  # noqa: F401 — import is the probe
            importable = True
        except ImportError:
            importable = False

    eff = effective_enhance(cli_enable=None, config_enhance=enhance_cfg)
    model_ref = eff["model"]
    # Read the raw config "default" key directly rather than the resolved
    # "enabled" — when cli_enable=None they happen to be equal, but the
    # variable name should reflect WHICH source we're reading. Surfacing
    # it as "config says enhance is default-on" is clearer than the
    # implicit detour through effective_enhance's CLI override logic.
    # (v0.5 python-reviewer I-2.)
    enabled_default = bool(enhance_cfg.get("default", False))

    cache_dir = hf_cache_dir_for(model_ref, hf_cache)
    model_cached = cache_dir.is_dir()
    cache_size: int | None = None
    if model_cached:
        size = _dir_size_bytes(cache_dir)
        # Threshold: anything under ~1 MB is almost certainly just the
        # config.json + tokenizer files without the weights blobs (the
        # download was interrupted before the .safetensors landed). Treat
        # as not-fully-cached so the doctor surfaces the pending download.
        if size >= 1_000_000:
            cache_size = size
        else:
            model_cached = False

    # Recent enhance-success-rate. Look at last 10 history entries
    # where the user actually ATTEMPTED to enhance — exclude entries
    # whose fallback_reason is ``user_opt_out`` (those are intentional
    # `--no-enhance` runs, not failed attempts). Without this filter
    # every run made before --enhance-prompt was first used would drag
    # the success rate to 0%, which would surface as a misleading
    # warning for users who simply haven't tried the feature yet.
    history = history if history is not None else load_history()
    recent = [
        e for e in history[-10:]
        if "enhanced" in e and e.get("enhance_fallback_reason") != "user_opt_out"
    ]
    succeeded = sum(1 for e in recent if e.get("enhanced") is True)

    return EnhanceHealth(
        mlx_lm_importable=importable,
        enabled_by_default=enabled_default,
        model_ref=model_ref,
        model_cached=model_cached,
        model_cache_size_bytes=cache_size,
        recent_runs=len(recent),
        recent_runs_succeeded=succeeded,
    )


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


def _ping_hf_whoami_and_report(token: str) -> int:
    """v0.7.1: validate an HF token via the whoami endpoint and emit
    a one-line status to stdout. Returns the doctor-issue delta —
    1 if the token is invalid (revoked/expired/wrong-scope, 401 from
    HF), 0 otherwise.

    Network unreachable / DNS down / HF outage all warn but return 0
    — the user's token may be fine, we just can't verify it. Refusing
    to ship them on an air-gapped Mac would be worse than missing a
    stale-token diagnosis.

    Extracted from inline cmd_doctor for testability — the inline
    chain wraps ``HfApi().whoami`` + ``HfHubHTTPError`` handling +
    print() + issue counter, all of which need separate mocks. A
    thin helper makes the test surface a single function-call away.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        # huggingface_hub not importable — broken venv, but the
        # mflux check above would have already failed loudly.
        return 0
    try:
        user_info = HfApi().whoami(token=token)
        raw_name = user_info.get("name") or user_info.get("fullname") or "?"
        # v0.7.2 security NIT: HF account names are user-controlled —
        # strip non-printable chars so an ANSI-laden username can't
        # clear the user's terminal when doctor prints it. Symmetric
        # with the validate_token path in tokens.py.
        username = safe_display_username(raw_name)
        ok(f"   HF whoami: logged in as {username}")
        return 0
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 401:
            warn("   HF whoami: token is INVALID "
                 "(revoked / expired / wrong scope)")
            print(f"   {C.DIM}Generate a new Read token at "
                  f"https://huggingface.co/settings/tokens{C.END}")
            print(f"   {C.DIM}Then: imgen setup  to update.{C.END}")
            return 1
        # v0.7.1 (python NIT-3): non-401 HTTP (5xx, 429, etc) is
        # transient — HF outage, rate limit, intermediate failure.
        # The token may be perfectly valid; we just couldn't verify
        # right now. Symmetric with the ConnectionError/Timeout path
        # below (warn but don't bump issues — air-gapped Macs +
        # HF-down windows shouldn't fail doctor).
        warn(f"   HF whoami: HuggingFace returned HTTP {status} "
             f"(transient — try again later)")
        return 0
    except Exception as e:
        # Network unreachable / DNS failure / etc.
        warn(f"   HF whoami: could not reach HuggingFace ({type(e).__name__})")
        return 0


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
    user_styles_dir = STYLES_D
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

    # RAM forecast per backend/quant. v0.7.14 (architect NIT closure):
    # column header now says "@1MP" because pre-v0.7.14 the table
    # value was the 2K²-worst-case peak; post-v0.7.14 it is the 1 MP
    # canonical baseline (peak grows by ~5 GB/MP above 1 MP per
    # ACTIVATION_GB_PER_MP_ABOVE_BASELINE — see checks.ram_required_gb
    # for the dimension-aware estimate the actual preflight uses).
    # Colleagues reading the table would otherwise misread the 1MP
    # baseline as the operational ceiling.
    print()
    info("Will this fit in RAM? (peak at 1024² output — larger "
         "resolutions need ~5 GB more per megapixel)")
    headers = ["backend × quant", "@1MP", "have", "verdict"]
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

        # v0.7.1: ping HF whoami so a stale / revoked / wrong-scope token
        # surfaces HERE (before the user wastes 13s on a first
        # snapshot_download attempt that 401s buried inside a stack
        # trace). Burned live during the v0.7.0 smoke pre-tag — user's
        # token went stale + the first failure mode was a mflux+
        # huggingface_hub traceback rather than a clean "token invalid"
        # message at the right surface.
        issues += _ping_hf_whoami_and_report(tok)
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
                    from ..hf_cache import repo_from_cache_dir
                    name = repo_from_cache_dir(model_dir.name)
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
    # will actually launch. Pure check_backend_health() returns the
    # classification; this loop turns it into UI. Mirrors the v0.3.6
    # alias-consistency check shape.
    print()
    info("Backends")
    for health in check_backend_health(venv_bin=VENV_BIN, env=os.environ):
        # !r-format the path so any C0/DEL/C1 bytes that snuck past
        # the validator (or that ride along on a path the validator
        # never saw, e.g. via a symlinked target name) render as
        # \x1b literals instead of escaping into the terminal.
        # (v0.4 security-reviewer IMP-2.)
        binary_safe = repr(str(health.binary_path))
        if health.binary_ok:
            ok(f"{health.name} ({health.origin}): {binary_safe}")
        else:
            warn(f"{health.name} ({health.origin}): binary not found "
                 f"(or not a regular file) at {binary_safe}"
                 + ("" if health.origin == "built-in" else " — fix "
                    "backends.d TOML or install the binary"))
            issues += 1
        # Secret env var status (only declared on custom backends).
        if health.secret_env_var is not None:
            if health.secret_present:
                ok(f"   secret ${health.secret_env_var} set")
            elif health.secret_required:
                warn(f"   secret ${health.secret_env_var} (required) "
                     f"NOT set in environment — "
                     f"`imgen --backend {health.name}` will die")
                issues += 1
            else:
                dim(f"   secret ${health.secret_env_var} (optional) "
                    "not set — best-effort forward, backend handles "
                    "its own auth")

    # Enhance — LLM prompt enhancer readiness. Reports whether mlx-lm
    # is importable, whether the configured model is in HF cache,
    # total cache size for that model, and recent enhance success-rate
    # from history. Not an issue if disabled — opt-in surface, so
    # "no enhance attempts in history + model not cached" is the
    # expected state for users who haven't tried --enhance-prompt.
    print()
    info("Smart prompts")
    enhance_cfg: dict = {}
    if CONFIG_FILE.exists():
        try:
            enhance_cfg = load_validated_config(CONFIG_FILE).get("enhance", {})
        except ConfigError:
            pass  # config errors surfaced in the User config section below
    eh = check_enhance_health(enhance_cfg=enhance_cfg, hf_cache=HF_CACHE)
    if not eh.mlx_lm_importable:
        warn("mlx-lm not importable — `--enhance-prompt` will die. "
             "Run `imgen upgrade` or `pip install -e .` to install.")
        issues += 1
    else:
        ok(f"mlx-lm importable, model: {eh.model_ref}")
    if eh.model_cached:
        gb = (eh.model_cache_size_bytes or 0) / 1e9
        ok(f"   model cached ({gb:.1f} GB) at {HF_CACHE}")
    else:
        dim(f"   model NOT cached — first `--enhance-prompt` run will "
            f"download ~4 GB from huggingface.co (one-time)")
    if eh.enabled_by_default:
        dim("   [enhance] default = true in config — every run enhances "
            "unless --no-enhance passed")
    if eh.recent_runs > 0:
        pct = 100 * eh.recent_runs_succeeded // eh.recent_runs
        if eh.recent_runs_succeeded == eh.recent_runs:
            ok(f"   recent: {eh.recent_runs_succeeded}/{eh.recent_runs} "
               f"enhance attempt(s) succeeded")
        else:
            warn(f"   recent: {eh.recent_runs_succeeded}/{eh.recent_runs} "
                 f"enhance attempt(s) succeeded ({pct}%) — "
                 "check enhance_fallback_reason in history.jsonl")

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
            n_enhance = len(cfg.get("enhance", {}))
            ok(f"{CONFIG_FILE}: {n_defaults} default(s), {n_ui} ui "
               f"setting(s), {n_enhance} enhance setting(s)")
            for k, v in cfg["defaults"].items():
                print(f"   {C.DIM}[defaults] {k} = {v!r}{C.END}")
            for k, v in cfg["ui"].items():
                print(f"   {C.DIM}[ui] {k} = {v!r}{C.END}")
            for k, v in cfg.get("enhance", {}).items():
                print(f"   {C.DIM}[enhance] {k} = {v!r}{C.END}")
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

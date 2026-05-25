"""`imgen upgrade [--latest]` — `git pull --ff-only` on the checkout (if
bootstrap-installed) + `pip install -e .` so new code loads + mflux refresh.

For pipx-installed users (no IMGEN_HOME), git/pip steps are skipped with a
`pipx upgrade imgen` hint.
"""
from __future__ import annotations

import subprocess

from ..checks import check_mflux, check_venv
from ..colors import C, die, info, ok, step, warn
from ..defaults import MFLUX_PIN
from ..paths import IMGEN_HOME, VENV_BIN


def _self_update() -> None:
    """git pull --ff-only in IMGEN_HOME, then re-install the package so
    the new code is actually loaded next run.

    Warns (doesn't fail) on any issue so the mflux-upgrade step still runs.
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

    # Re-install the package so the venv picks up the new code. Required
    # since v0.2 — before the split, `imgen` was a standalone script and
    # git pull was enough.
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

    # v0.8.0 commit 6: keep .venv-diffusers/ editable-install in sync.
    # Per architect commit-6 pre-vet M2: UNCONDITIONAL pip install -e
    # if the diffusers venv exists — 5-10s cost amortises against the
    # risk of "imgen upgrade pulled new transitive deps, diffusers venv
    # didn't, ImportError at next imgen draw" UX disaster. The
    # diffusers venv has imgen editable-installed too (see
    # bootstrap.sh §5b); editable install means src/imgen/* is shared
    # via egg-link, but transitive Python deps in pyproject.toml are
    # NOT auto-pulled on the next imgen launch — pip must re-run.
    _diffusers_venv = IMGEN_HOME / ".venv-diffusers" if IMGEN_HOME else None
    if _diffusers_venv is not None and _diffusers_venv.is_dir():
        diff_pip = _diffusers_venv / "bin" / "pip"
        if diff_pip.is_file():
            try:
                subprocess.check_call(
                    [str(diff_pip), "install", "--quiet", "-e",
                     str(IMGEN_HOME)],
                    timeout=180,
                )
                ok("diffusers venv editable-install refreshed")
            except (subprocess.CalledProcessError,
                    subprocess.TimeoutExpired) as e:
                warn(f"diffusers venv pip install -e failed: {e}")
                print(f"   {C.DIM}Re-run: {IMGEN_HOME / 'bootstrap.sh'}"
                      f" (the diffusers prompt re-creates the venv)"
                      f"{C.END}")


def cmd_upgrade(args) -> int:
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
            [str(pip), "install", "--upgrade", target],
            timeout=300,
        )
        ok(f"mflux is now {check_mflux()}")
    except subprocess.CalledProcessError as e:
        die(f"mflux upgrade failed: {e}", code=3)
    except subprocess.TimeoutExpired:
        die("mflux upgrade timed out after 5 min — check your network",
            code=3)
    return 0

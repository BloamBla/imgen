"""Filesystem path constants used by imgen.

Two install modes:
  - Bootstrap mode: user cloned the repo to ~/imgen (or anywhere), runs
    bootstrap.sh, ends up with ~/imgen/.venv/ and the launcher shim sets
    IMGEN_HOME env var before exec'ing the venv's imgen entry point.
  - Pipx mode: `pipx install git+...` installs the package into a pipx-
    managed venv; there is no repo dir. IMGEN_HOME is None in this mode.

The venv that hosts mflux is always `Path(sys.executable).parent` — works
for both modes because the entry-point binary always runs under the venv
where the imgen package itself was installed.

Per-run + per-batch-log helpers (auto_run_dirname, LOGS_DIR, ...) used
to live here but were moved to ``runs.py`` in v0.2.4 — this module is
now strictly for filesystem-path constants and the ``ensure_state_dir``
bootstrap.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "BACKENDS_D",
    "CONFIG_FILE",
    "DEFAULT_OUTPUT_DIR",
    "HF_CACHE",
    "HF_CLI_TOKEN_FILE",
    "HISTORY_FILE",
    "IMGEN_HOME",
    "IMGEN_INSTALL_ROOT",
    "LEGACY_TOKEN_FILE",
    "MFLUX_LORAS_CACHE",
    "MODELS_D",
    "MODELS_D_EXAMPLE",
    "SAFE_OUTPUT_EXTS",
    "STATE_DIR",
    "STYLES_D",
    "TOKEN_FILE",
    "VENV_BIN",
    "ensure_state_dir",
]

# IMGEN_HOME: repo checkout dir. Set by the bash shim at ~/imgen/imgen
# before exec'ing the venv entry point. None for pipx-installed users —
# `_self_update` and the bootstrap-style alias path are skipped in that
# mode.
_imgen_home_env = os.environ.get("IMGEN_HOME")
IMGEN_HOME: Path | None = (
    Path(_imgen_home_env).resolve() if _imgen_home_env else None
)

# The bin/ dir of the venv hosting this package. Mflux binaries live here.
VENV_BIN = Path(sys.executable).parent


def _compute_imgen_install_root() -> Path:
    """Locate the imgen install root for diffusers_mps engine venv
    resolution (v0.8.0 §E lock-in).

    The diffusers_mps engine spawns ``.venv-diffusers/bin/python`` to
    run a static runner module. That path MUST resolve from a stable
    anchor, not from cwd — otherwise ``cd /tmp/attacker && imgen ...``
    could exec a planted python.

    Probe order:

    1. ``Path(sys.prefix).parent`` — canonical bootstrap.sh layout.
       sys.prefix is ``<imgen-root>/.venv``; its parent is the install
       root containing ``src/imgen/__init__.py``.
    2. ``Path(__file__).resolve().parents[2]`` — fallback for pipx /
       uv tool install / Homebrew Python where sys.prefix is unrelated
       (e.g. ``/opt/homebrew/...``). paths.py is at
       ``<imgen-root>/src/imgen/paths.py``, so parents[2] is the root.

    Both probes verify ``<candidate>/src/imgen/__init__.py`` exists, so
    a directory that happens to be on the path but isn't an imgen
    install is rejected.

    Dies via SystemExit if neither probe resolves — the install is
    broken in a way diffusers_mps engine couldn't recover from anyway,
    and the error message gives the user a path forward
    (``IMGEN_INSTALL_ROOT`` env var would be the v0.8.x extension if
    field reports ever need it).
    """
    # Local import to keep die() out of module-load critical path —
    # colors.die uses ANSI codes that wouldn't apply at this boot phase.
    from sys import exit as _sys_exit

    def _candidate_is_imgen_root(p: Path) -> bool:
        return (p / "src" / "imgen" / "__init__.py").is_file()

    # Probe 1 — canonical bootstrap.sh / `python -m venv .venv` layout.
    venv_layout = Path(sys.prefix).parent
    if _candidate_is_imgen_root(venv_layout):
        return venv_layout

    # Probe 2 — fallback for pipx / uv / Homebrew where sys.prefix is
    # the system Python, not a project-local venv. paths.py lives at
    # `<root>/src/imgen/paths.py`; parents[2] is the root.
    source_layout = Path(__file__).resolve().parents[2]
    if _candidate_is_imgen_root(source_layout):
        return source_layout

    # Neither resolved. Surface a diagnostic before dying — diffusers_mps
    # engine needs this; mflux-only users hit this branch only if the
    # source tree is genuinely broken.
    sys.stderr.write(
        "imgen: cannot locate install root for diffusers_mps engine.\n"
        f"  Tried sys.prefix.parent  = {venv_layout}\n"
        f"  Tried __file__-relative  = {source_layout}\n"
        "  Neither contains src/imgen/__init__.py.\n"
        "  If you installed imgen via pipx/uv, file an issue — the\n"
        "  fallback chain needs another probe arm for your layout.\n"
    )
    _sys_exit(1)


# Install root anchor for ``.venv-diffusers/`` resolution (v0.8 §E).
# Computed eagerly at module load so paths.STATE_DIR / paths.HF_CACHE /
# IMGEN_INSTALL_ROOT all stay consistent (no lazy-init footgun where the
# value depends on call-site cwd at first read).
IMGEN_INSTALL_ROOT = _compute_imgen_install_root()

# Persistent state — independent of install mode.
STATE_DIR = Path.home() / ".imgen"
HISTORY_FILE = STATE_DIR / "history.jsonl"
CONFIG_FILE = STATE_DIR / "config.toml"
# HF token moved under STATE_DIR in v0.2.2 — `~/.hf_token` was a generic
# name other HF tooling might claim. Legacy path is still read as a
# fallback and auto-migrated to TOKEN_FILE on first load. See tokens.py.
TOKEN_FILE = STATE_DIR / "hf_token"
LEGACY_TOKEN_FILE = Path.home() / ".hf_token"
# v0.7.12 (gap 9): HF CLI's own token store, separate from imgen's. mflux
# subprocess reads TOKEN_FILE via env injection; `hf` CLI + standalone
# diffusers read HF_CLI_TOKEN_FILE. Pre-v0.7.12 these drifted silently —
# fresh imgen token next to a stale HF CLI token (or vice versa) caused
# `hf download` to fail with "Invalid user token" even though imgen
# worked. ``sync_token_to_hf_cli_store`` in tokens.py writes here after
# ``save_token_atomic`` to keep both stores aligned at setup time.
HF_CLI_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
# Fallback only — runtime env ($IMGEN_OUTPUT_DIR) and ~/.imgen/config.toml
# `[defaults] output_dir` win in that order. Resolution lives in
# config.effective_output_dir, which checks env at call time so test
# monkeypatches see the patched value (previously this constant captured
# env at module import → tests had no way to override).
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "imgen"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
# mflux's LoRA download cache lives under platformdirs.user_cache_dir
# ("mflux") + "/loras" — on macOS that's ``~/Library/Caches/mflux/loras``.
# Used by ``--list-loras`` to probe whether a built-in / user LoRA's
# weights are already on disk. Same ``models--<author>--<name>``
# convention inside as the standard HF cache, so ``hf_cache_dir_for``
# returns a valid path under either root. (v0.6.4 task #21 — v0.6.3's
# --list-loras reported "not downloaded" for every LoRA because it
# only probed HF_CACHE; mflux actually writes here.) The env var
# ``MFLUX_CACHE_DIR`` overrides mflux's root; we ignore it here for
# simplicity since the override is rare and probe is read-only — if a
# user sets it AND complains about --list-loras being stale, we'll
# bridge the override.
MFLUX_LORAS_CACHE = Path.home() / "Library" / "Caches" / "mflux" / "loras"
# User-extension subdirectories under STATE_DIR. Single source of
# truth — setup.py creates these on first run, styles.py/backends.py
# scan them at startup, doctor.py reports their contents. Centralized
# here to prevent drift across modules (v0.4 architect IMP-4 — same
# rationale as shell_rc.ALL_RC_FILES_REL added in v0.3.6).
STYLES_D = STATE_DIR / "styles.d"
BACKENDS_D = STATE_DIR / "backends.d"
# v0.8.0 canonical path for user-TOML model registrations. Read alongside
# BACKENDS_D during the v0.8.x deprecation window; same-stem-in-both
# resolves with MODELS_D winning (encourages migration). The deprecation
# warn on BACKENDS_D entries lands in v0.8.0 commit 4a — not at commit 3,
# which only adds the second read path. See [[project-v080-design]] §H.
MODELS_D = STATE_DIR / "models.d"
# v0.8.0 commit 10 (§G.2): opt-in template directory. ``imgen setup``
# (and bootstrap.sh by extension) drops example TOMLs here; the user
# moves a file into ``models.d/`` to activate. NOT scanned by the
# loader — that's the whole point of the demotion per §G.2.
MODELS_D_EXAMPLE = STATE_DIR / "models.d.example"

# Output extensions allowed for --output and auto-`open`. macOS `open`
# delegates to the registered app for the extension, so .terminal /
# .command / .sh etc would auto-execute. Restrict to known-safe image
# suffixes.
SAFE_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_state_dir() -> None:
    """Create STATE_DIR with restrictive perms (history may contain prompts).

    Trust-boundary note (v0.2.6 review NIT-A + python NIT-4, doc-only):
    we do NOT verify that STATE_DIR itself is a real directory rather
    than a symlink, nor do we close the TOCTOU window between
    `LOGS_DIR.is_symlink()` and the subsequent glob/unlink in
    ``runs.prune_old_batch_logs``. Both gaps are deliberate — imgen is
    a single-user CLI on the user's own Mac, and an attacker with
    same-uid code-exec already has direct `rm -rf` and file-write, so
    symlink games gain them nothing they can't do more directly. The
    ``LOGS_DIR`` symlink guards in ``runs.py`` are defence-in-depth
    against an accidental `ln -s` over ``~/.imgen/logs/``, not against
    a compromised STATE_DIR — don't read them as full path-traversal
    hardening.
    """
    if not STATE_DIR.exists():
        STATE_DIR.mkdir(mode=0o700)
    elif (STATE_DIR.stat().st_mode & 0o777) != 0o700:
        try:
            STATE_DIR.chmod(0o700)
        except OSError:
            pass

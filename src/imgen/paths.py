"""All filesystem paths used by imgen + state-dir setup.

Two install modes:
  - Bootstrap mode: user cloned the repo to ~/imgen (or anywhere), runs
    bootstrap.sh, ends up with ~/imgen/.venv/ and the launcher shim sets
    IMGEN_HOME env var before exec'ing the venv's imgen entry point.
  - Pipx mode: `pipx install git+...` installs the package into a pipx-
    managed venv; there is no repo dir. IMGEN_HOME is None in this mode.

The venv that hosts mflux is always `Path(sys.executable).parent` — works
for both modes because the entry-point binary always runs under the venv
where the imgen package itself was installed.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

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

# Persistent state — independent of install mode.
STATE_DIR = Path.home() / ".imgen"
HISTORY_FILE = STATE_DIR / "history.jsonl"
CONFIG_FILE = STATE_DIR / "config.toml"
# Per-batch logs (v0.2.3+) — one .log file per multi-style invocation,
# named after batch_id. Single-style generations don't write here.
# Retention is enforced by `imgen clean` (30 days).
LOGS_DIR = STATE_DIR / "logs"
LOG_RETENTION_DAYS = 30
# HF token moved under STATE_DIR in v0.2.2 — `~/.hf_token` was a generic
# name other HF tooling might claim. Legacy path is still read as a
# fallback and auto-migrated to TOKEN_FILE on first load. See tokens.py.
TOKEN_FILE = STATE_DIR / "hf_token"
LEGACY_TOKEN_FILE = Path.home() / ".hf_token"
# Fallback only — runtime env ($IMGEN_OUTPUT_DIR) and ~/.imgen/config.toml
# `[defaults] output_dir` win in that order. Resolution lives in
# config.effective_output_dir, which checks env at call time so test
# monkeypatches see the patched value (previously this constant captured
# env at module import → tests had no way to override).
DEFAULT_OUTPUT_DIR = Path.home() / "Desktop" / "imgen"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Output extensions allowed for --output and auto-`open`. macOS `open`
# delegates to the registered app for the extension, so .terminal /
# .command / .sh etc would auto-execute. Restrict to known-safe image
# suffixes.
SAFE_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def auto_run_dirname(now: _dt.datetime | None = None) -> str:
    """Folder name for one CLI invocation: '2026-05-21-14-30-12'.

    All-dashes, second precision. Sortable alphabetically = sortable
    chronologically. No colons (`:`) so the path survives macOS quirks
    and copy-paste into terminals that quote-mangle `:`. We run mflux
    serially so no two generations within one invocation finish in the
    same second; folder-level collisions only arise from scripted
    double-invokes, handled by `next_available_run_dir`.
    """
    if now is None:
        now = _dt.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def next_available_run_dir(parent: Path, dirname: str) -> Path:
    """Return parent/dirname, suffixing `_2`, `_3` if it already exists.

    Pure: does not create the directory. Caller is responsible for
    `mkdir(parents=True, exist_ok=True)` on the returned path.
    """
    target = parent / dirname
    if not target.exists():
        return target
    i = 2
    while (parent / f"{dirname}_{i}").exists():
        i += 1
    return parent / f"{dirname}_{i}"


def ensure_state_dir() -> None:
    """Create STATE_DIR with restrictive perms (history may contain prompts)."""
    if not STATE_DIR.exists():
        STATE_DIR.mkdir(mode=0o700)
    elif (STATE_DIR.stat().st_mode & 0o777) != 0o700:
        try:
            STATE_DIR.chmod(0o700)
        except OSError:
            pass


def ensure_logs_dir() -> None:
    """Create LOGS_DIR (0o700) under STATE_DIR.

    Used by cmd_generate when it opens a per-batch log for multi-style
    runs. STATE_DIR is created first so a fresh user never hits ENOENT.
    """
    ensure_state_dir()
    if not LOGS_DIR.exists():
        LOGS_DIR.mkdir(mode=0o700)
    elif (LOGS_DIR.stat().st_mode & 0o777) != 0o700:
        try:
            LOGS_DIR.chmod(0o700)
        except OSError:
            pass

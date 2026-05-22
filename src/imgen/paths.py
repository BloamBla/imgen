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
    "CONFIG_FILE",
    "DEFAULT_OUTPUT_DIR",
    "HF_CACHE",
    "HISTORY_FILE",
    "IMGEN_HOME",
    "LEGACY_TOKEN_FILE",
    "SAFE_OUTPUT_EXTS",
    "STATE_DIR",
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

# Persistent state — independent of install mode.
STATE_DIR = Path.home() / ".imgen"
HISTORY_FILE = STATE_DIR / "history.jsonl"
CONFIG_FILE = STATE_DIR / "config.toml"
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

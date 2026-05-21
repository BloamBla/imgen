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
TOKEN_FILE = Path.home() / ".hf_token"
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("IMGEN_OUTPUT_DIR", Path.home() / "Desktop" / "imgen")
)
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Output extensions allowed for --output and auto-`open`. macOS `open`
# delegates to the registered app for the extension, so .terminal /
# .command / .sh etc would auto-execute. Restrict to known-safe image
# suffixes.
SAFE_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_state_dir() -> None:
    """Create STATE_DIR with restrictive perms (history may contain prompts)."""
    if not STATE_DIR.exists():
        STATE_DIR.mkdir(mode=0o700)
    elif (STATE_DIR.stat().st_mode & 0o777) != 0o700:
        try:
            STATE_DIR.chmod(0o700)
        except OSError:
            pass

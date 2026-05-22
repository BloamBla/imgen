"""Shell rc file paths — single source of truth for setup + doctor.

Pre-v0.3.6 the rc-file enumeration was duplicated between
``commands/setup.py`` (writes the alias) and ``commands/doctor.py``
(reads + checks for staleness). The two copies had drifted: doctor
read ``.bashrc`` which setup never wrote to, while setup wrote to one
file per current shell. The architect IMP-1 review caught the drift;
this module is the fix.

Both consumers import from here so the rc-file truth lives in exactly
one place. Adding shell support (e.g. nushell) is a one-line change
to ``RC_FILE_BY_SHELL`` that automatically extends the doctor's
divergence check too.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "ALL_RC_FILES_REL",
    "RC_FILE_BY_SHELL",
]

# Map ``basename($SHELL)`` → rc file path relative to ``$HOME``.
# ``setup.py`` writes the imgen alias to the file selected by the
# user's current shell; shells not in this map get a "manual setup
# needed" hint instead.
#
# macOS-specific: bash maps to ``.bash_profile`` (login-shell rc that
# Terminal.app uses by default), not ``.bashrc`` (interactive-shell
# rc more common on Linux). This is the macOS-only convention; if
# imgen ever supports Linux the mapping would need to branch on
# platform, but mflux/MLX pin it to Apple Silicon for now.
RC_FILE_BY_SHELL: dict[str, Path] = {
    "zsh": Path(".zshrc"),
    "bash": Path(".bash_profile"),
    "fish": Path(".config") / "fish" / "config.fish",
}

# All rc files imgen ever writes to, as $HOME-relative paths. Doctor
# reads every entry to detect stale aliases — a user who switched
# from bash → zsh keeps the old ``.bash_profile`` alias around, and
# we want to surface that even though their current shell is zsh.
# Tuple (not list) so consumers can't mutate the shared state.
ALL_RC_FILES_REL: tuple[Path, ...] = tuple(RC_FILE_BY_SHELL.values())

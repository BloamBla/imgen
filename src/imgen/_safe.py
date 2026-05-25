"""Shared safety helpers for terminal-display + control-byte filtering.

Per the v0.4 design decision: ``_has_control_bytes`` was duplicated
inline across ``backends.py`` / ``parser.py`` / ``history.py`` until a
4th surface emerged. That threshold was hit in v0.8.0 commit 9
(history.py made it the 4th byte-detector copy). At the same time,
``commands/migrate_toml.py`` shipped its own local ``_safe_path_display``
display-helper duplicate of the same control-byte-aware repr pattern.
v0.8.1 LOW-3 closure consolidates BOTH surfaces (byte detector + display
wrapper) into this module so the consumers share a single byte-identical
implementation + a documented contract.

Three helpers ship from this module:

* :func:`has_control_bytes` — the C0/DEL/C1 detector. Reject when True.
* :func:`safe_display` — wrap any string via ``repr()`` so control
  bytes render as ``\\xNN`` literals instead of escaping into the
  user's terminal. Use for display-only sites (logs, warns, doctor
  output). Do NOT use for argv composition or file-path resolution.
* :func:`safe_path_display` — same as ``safe_display`` for ``Path``
  objects (just stringifies first).

Keep this module dependency-free (no imports beyond stdlib pathlib)
so anything in the imgen package can import from it without cycles.
"""
from __future__ import annotations

from pathlib import Path


__all__ = ["has_control_bytes", "safe_display", "safe_path_display"]


def has_control_bytes(s: str) -> bool:
    """C0 (0x00-0x1F) / DEL (0x7F) / C1 (0x80-0x9F) byte detector.

    Reject when True at any boundary where a string flows into:
      * subprocess argv (escape sequences corrupt stderr / log files)
      * filenames (filesystem-level surprises + display hazards)
      * structured display output (terminal escape injection)

    The C1 range (0x80-0x9F) is included because some terminals
    interpret it as control regardless of UTF-8 framing — defense in
    depth against a UTF-8 string that decodes legitimately but
    contains rare control characters.

    Pure: no I/O. Inlined hot — every character compared with a few
    branches.
    """
    return any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in s
    )


def safe_display(s: str) -> str:
    """Render a string for terminal/log output with control bytes
    escaped to ``\\xNN`` literals.

    Wraps via ``repr()`` so any C0/DEL/C1 byte that snuck onto disk
    via a hand-crafted filename / TOML value renders visibly instead
    of triggering a terminal-escape sequence (e.g. clearing the
    user's terminal). The returned string is quoted (single or double
    quotes per Python's repr rules).

    Use for DISPLAY ONLY — do NOT pass the result to subprocess argv,
    file-system APIs, or string-equality checks. Those surfaces want
    the raw string (rejected upstream via ``has_control_bytes`` if
    dirty).
    """
    return repr(s)


def safe_path_display(p: Path | str) -> str:
    """Same as :func:`safe_display` for path-shaped inputs.

    Accepts ``Path`` or ``str`` (any ``__fspath__``-able object via
    ``str()``). Equivalent to ``safe_display(str(p))``.
    """
    return repr(str(p))

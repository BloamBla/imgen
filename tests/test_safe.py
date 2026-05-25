"""v0.8.1 LOW-3 closure — shared imgen._safe helpers.

Per [[project-v08x-backlog]] LOW-3: ``_has_control_bytes`` was
duplicated inline across backends.py / parser.py / history.py /
migrate_toml.py per the v0.4 design lock. v0.8.1 extracts a shared
``imgen._safe`` module exporting ``has_control_bytes`` +
``safe_display`` + ``safe_path_display``. These tests lock in the
contract.
"""
from __future__ import annotations

from pathlib import Path

from imgen._safe import has_control_bytes, safe_display, safe_path_display


# ── has_control_bytes ────────────────────────────────────────────────


def test_has_control_bytes_clean_string_returns_false():
    """Plain ASCII text, normal whitespace, accented chars — all clean."""
    assert has_control_bytes("") is False
    assert has_control_bytes("flux-kontext") is False
    assert has_control_bytes("hello world") is False
    assert has_control_bytes("café français") is False  # Latin-1 accents
    assert has_control_bytes("日本語") is False  # CJK
    assert has_control_bytes("emoji 🚀 included") is False


def test_has_control_bytes_c0_range_detected():
    """C0 control bytes (0x00-0x1F) — reject."""
    # \x00 NUL, \x07 BEL, \x09 TAB, \x0a NEWLINE, \x1b ESC, \x1f
    for byte in (0x00, 0x07, 0x09, 0x0a, 0x1b, 0x1f):
        assert has_control_bytes(f"foo{chr(byte)}bar") is True, (
            f"byte 0x{byte:02x} not detected"
        )


def test_has_control_bytes_del_detected():
    """DEL (0x7F) — reject. Sits between C0 and printable ASCII."""
    assert has_control_bytes("foo\x7fbar") is True


def test_has_control_bytes_c1_range_detected():
    """C1 control bytes (0x80-0x9F) — reject. Some terminals
    interpret these even when UTF-8 framing would suggest otherwise."""
    for byte in (0x80, 0x88, 0x9a, 0x9f):
        assert has_control_bytes(f"foo{chr(byte)}bar") is True, (
            f"byte 0x{byte:02x} not detected"
        )


def test_has_control_bytes_printable_boundary_unchanged():
    """Bytes just above C1 (0xA0 = nbsp) — accepted, not control."""
    assert has_control_bytes("foo\xa0bar") is False
    # Plain ASCII boundary: space (0x20) is the lowest non-control.
    assert has_control_bytes("foo bar") is False


# ── safe_display ─────────────────────────────────────────────────────


def test_safe_display_clean_string_wraps_in_quotes():
    """Clean string round-trips via repr — gets quotes added."""
    out = safe_display("flux-kontext")
    assert out.startswith("'") or out.startswith('"')
    assert out.endswith("'") or out.endswith('"')
    assert "flux-kontext" in out


def test_safe_display_escape_sequence_rendered_as_literal():
    """ANSI escape sequence renders as ``\\x1b`` literal, not as the
    actual escape that would clear the terminal."""
    out = safe_display("attack\x1b[2J")
    # repr() escapes \x1b → \\x1b (literal backslash + x1b in the
    # rendered string). The raw 0x1b byte must NOT be present.
    assert "\x1b" not in out, (
        f"raw escape byte leaked into display output: {out!r}"
    )
    # And the escape sequence must be visible as a literal.
    assert "\\x1b" in out


# ── safe_path_display ────────────────────────────────────────────────


def test_safe_path_display_accepts_path_object():
    """Path → repr(str(p)) round-trip."""
    p = Path("/tmp/foo.toml")
    out = safe_path_display(p)
    assert "/tmp/foo.toml" in out
    assert out.startswith("'") or out.startswith('"')


def test_safe_path_display_accepts_str():
    """str input also works — same shape as Path input."""
    out = safe_path_display("/tmp/foo.toml")
    assert "/tmp/foo.toml" in out


# ── Cross-module contract: every consumer imports the shared helper ──


def test_backends_uses_shared_has_control_bytes():
    """backends.py uses the shared helper (not a local duplicate)."""
    from imgen import _safe
    from imgen.backends import _has_control_bytes as backends_helper
    assert backends_helper is _safe.has_control_bytes


def test_history_uses_shared_has_control_bytes():
    """history.py uses the shared helper."""
    from imgen import _safe
    from imgen.history import _has_control_bytes as history_helper
    assert history_helper is _safe.has_control_bytes

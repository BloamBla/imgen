"""Confirm-gate helpers in commands/generate.py (v0.2.3).

`_estimate_one_seconds`, `_format_duration`, and `_confirm_batch` are
pure on their inputs — exercised here in isolation. The cmd_generate
integration is smoke-tested manually (interactive prompt + ANSI codes
make full automation overkill for this surface).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.cmd_helpers import (
    estimate_one_seconds as _estimate_one_seconds,
    format_duration as _format_duration,
)
from imgen.commands.generate import _confirm_batch
from imgen.runs import Iteration


# ── _estimate_one_seconds ───────────────────────────────────────────────

def _ok_entry(backend: str, quant: int, preview: bool, duration: int) -> dict:
    return {
        "status": "success",
        "backend": backend,
        "quantize": quant,
        "preview": preview,
        "duration_sec": duration,
    }


def test_estimate_one_seconds_returns_none_on_empty_history():
    assert _estimate_one_seconds([], "flux", 4, False) is None


def test_estimate_one_seconds_returns_none_when_no_match():
    """History has entries, but none matching the backend/quant combo."""
    entries = [_ok_entry("flux", 8, False, 3000)]
    assert _estimate_one_seconds(entries, "flux", 4, False) is None


def test_estimate_one_seconds_uses_average_of_matching_successes():
    entries = [
        _ok_entry("flux", 4, False, 300),
        _ok_entry("flux", 4, False, 360),
        _ok_entry("flux", 4, False, 420),
    ]
    # avg(300, 360, 420) = 360
    assert _estimate_one_seconds(entries, "flux", 4, False) == 360


def test_estimate_one_seconds_caps_at_last_five():
    """A long history should only average the 5 most-recent matching successes."""
    entries = [_ok_entry("flux", 4, False, 1000)] * 10
    entries += [_ok_entry("flux", 4, False, 60)] * 5
    # Last 5 are all 60s
    assert _estimate_one_seconds(entries, "flux", 4, False) == 60


def test_estimate_one_seconds_ignores_failed_runs():
    """A failed run with duration_sec must not skew the estimate."""
    entries = [
        {"status": "failed", "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 999999},
        _ok_entry("flux", 4, False, 300),
    ]
    assert _estimate_one_seconds(entries, "flux", 4, False) == 300


def test_estimate_one_seconds_ignores_cancelled_runs():
    entries = [
        {"status": "cancelled", "backend": "flux", "quantize": 4,
         "preview": False, "duration_sec": 50},
        _ok_entry("flux", 4, False, 300),
    ]
    assert _estimate_one_seconds(entries, "flux", 4, False) == 300


def test_estimate_one_seconds_distinguishes_preview_mode():
    """--preview generations are 5-10x faster — don't conflate."""
    entries = [
        _ok_entry("flux", 4, True, 180),   # preview run
        _ok_entry("flux", 4, False, 1800),
    ]
    assert _estimate_one_seconds(entries, "flux", 4, True) == 180
    assert _estimate_one_seconds(entries, "flux", 4, False) == 1800


def test_estimate_one_seconds_ignores_zero_duration_runs():
    """A `duration_sec=0` entry (cancelled-in-same-second, or weird
    mflux instant-exit) must not pull the average to zero — ETA would
    print '0s per image' nonsense.
    (python I4 from v0.2.3 review)"""
    entries = [
        _ok_entry("flux", 4, False, 0),    # 0-duration garbage
        _ok_entry("flux", 4, False, 300),
        _ok_entry("flux", 4, False, 360),
    ]
    # avg(300, 360) = 330, not avg(0, 300, 360) = 220
    assert _estimate_one_seconds(entries, "flux", 4, False) == 330


def test_estimate_one_seconds_returns_none_when_all_zero():
    """If the only successes are 0-duration, treat as 'no data' — don't
    show a misleading 0s/image ETA."""
    entries = [_ok_entry("flux", 4, False, 0)] * 3
    assert _estimate_one_seconds(entries, "flux", 4, False) is None


# ── _format_duration ────────────────────────────────────────────────────

def test_format_duration_short_uses_seconds():
    assert _format_duration(45) == "45s"


def test_format_duration_at_boundary_one_minute():
    assert _format_duration(60) == "~1 min"


def test_format_duration_minutes():
    assert _format_duration(300) == "~5 min"
    assert _format_duration(1830) == "~30 min"


# ── _confirm_batch ──────────────────────────────────────────────────────

def _iters(*style_names: str) -> list[Iteration]:
    """Build minimal Iteration objects — _confirm_batch only reads
    style_name; the other 8 fields can be any valid value."""
    return [
        Iteration(
            style_name=s,
            prompt="",
            negative="",
            final_steps=14,
            final_quantize=8,
            final_guidance=2.5,
            final_strength=0.6,
            output_path=Path("/tmp/dummy.png"),
            cmd=[],
        )
        for s in style_names
    ]


def test_confirm_batch_yes_proceeds(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    result = _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="photo.jpg",
        output_root=Path("/tmp/run"),
        one_eta_seconds=300,
    )
    assert result is True
    out = capsys.readouterr().out
    assert "About to generate 2 images" in out
    assert "anime" in out and "ghibli" in out
    assert "photo.jpg" in out
    assert "/tmp/run" in out
    # ETA shown
    assert "min" in out and "per image" in out


def test_confirm_batch_n_cancels(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="photo.jpg",
        output_root=Path("/tmp/run"),
        one_eta_seconds=None,
    ) is False


def test_confirm_batch_empty_answer_cancels(monkeypatch):
    """Pressing Enter on `[y/N]` → default is No."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=None,
    ) is False


def test_confirm_batch_eof_cancels(monkeypatch):
    """Piped stdin closed (EOF) → cancel, don't crash."""
    def _eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", _eof)
    assert _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=None,
    ) is False


def test_confirm_batch_ctrl_c_cancels(monkeypatch):
    def _ctrl_c(_):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", _ctrl_c)
    assert _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=None,
    ) is False


def test_confirm_batch_uppercase_yes_accepted(monkeypatch):
    """`Y` and `YES` are also acceptances — match common shell habit."""
    monkeypatch.setattr("builtins.input", lambda _: "YES")
    assert _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=None,
    ) is True


def test_confirm_batch_eta_hidden_when_no_history(monkeypatch, capsys):
    """one_eta_seconds=None → no ETA line printed (don't guess)."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    _confirm_batch(
        iterations=_iters("anime", "ghibli"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=None,
    )
    out = capsys.readouterr().out
    assert "eta:" not in out.lower()
    assert "min" not in out.lower()


def test_confirm_batch_eta_shown_with_per_image_breakdown(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    _confirm_batch(
        iterations=_iters("anime", "ghibli", "pixar"),
        input_name="x.jpg",
        output_root=Path("/tmp"),
        one_eta_seconds=300,
    )
    out = capsys.readouterr().out
    # Total = 3 × 300s = 15 min, per image = ~5 min
    assert "~15 min total" in out
    assert "~5 min per image" in out

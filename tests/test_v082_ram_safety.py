"""v0.8.2 safety net — hard-floor RAM check in
``subprocess_helpers.run_with_stderr_redaction``.

Defence-in-depth against the v0.8.2 M-1C ops post-mortem (2026-05-26)
where a test-infrastructure gap caused real mflux subprocesses to
spawn into the user's machine memory in parallel. The hard floor
refuses to spawn ML subprocesses when available RAM is critically low,
covering 6+ preflight-bypass scenarios:

  1. ``--force`` flag at CLI entry skips preflight_resources
  2. System state changed between preflight and the actual spawn
  3. Race between parallel ``imgen`` invocations
  4. User TOML lying about ``ram_baseline_gb``
  5. External code calling ``Engine.run`` directly
  6. Test infrastructure bugs that miss the engine path

The check fires BEFORE any ``subprocess.Popen``, so no ML weights ever
get loaded into a dangerous memory state. ``InsufficientRAMError``
propagates up to ``cmd_helpers.run_one_iteration`` which catches it
alongside KeyboardInterrupt and writes a failure history entry.
"""
from __future__ import annotations

import os

import pytest

from imgen.subprocess_helpers import (
    InsufficientRAMError,
    _MIN_SAFE_AVAILABLE_RAM_GB,
    _assert_safe_ram_or_raise,
    run_with_stderr_redaction,
)


# ── _assert_safe_ram_or_raise unit tests ─────────────────────────────


def test_safety_net_raises_when_available_ram_below_floor(monkeypatch):
    """Lock-in: with available RAM artificially constrained to 2 GB
    (below the 4 GB floor), the helper raises InsufficientRAMError.
    Pure unit test — no subprocess spawn at all."""
    import imgen.checks as checks_mod
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 2.0),
    )
    with pytest.raises(InsufficientRAMError) as exc:
        _assert_safe_ram_or_raise()
    msg = str(exc.value)
    assert "2.0 GB" in msg
    assert "4.0 GB" in msg
    assert "IMGEN_BYPASS_RAM_FLOOR" in msg


def test_safety_net_passes_when_available_ram_above_floor(monkeypatch):
    """Negative test: with healthy available RAM (20 GB free of 32 GB),
    the check is silent and returns None. Guards against an overeager
    "raise always" regression."""
    import imgen.checks as checks_mod
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 20.0),
    )
    # Should NOT raise
    _assert_safe_ram_or_raise()


def test_safety_net_bypass_env_var_skips_check(monkeypatch):
    """Escape hatch: IMGEN_BYPASS_RAM_FLOOR=1 unconditionally bypasses
    the check, even when RAM is at 0.5 GB. For CI / power users who
    knowingly accept OOM risk. The error message itself documents this
    opt-out so end users always know the escape hatch."""
    import imgen.checks as checks_mod
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 0.5),
    )
    monkeypatch.setenv("IMGEN_BYPASS_RAM_FLOOR", "1")
    # Should NOT raise despite catastrophically-low RAM
    _assert_safe_ram_or_raise()


def test_safety_net_parse_failure_does_not_false_positive(monkeypatch):
    """``get_memory_gb()`` returns (0, 0) on parse failure (non-Darwin,
    sysctl unavailable, malformed vm_stat output). We treat that as
    "unknown — allow" rather than blocking legit CI / Linux smokes.
    The production target is Apple Silicon Macs where get_memory_gb is
    well-tested; this is defence against false positives on the edges."""
    import imgen.checks as checks_mod
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (0.0, 0.0),
    )
    # Should NOT raise despite the apparent "0 GB" reading
    _assert_safe_ram_or_raise()


def test_safety_net_floor_value_is_documented_constant():
    """Lock-in: the floor lives in a named module constant
    (_MIN_SAFE_AVAILABLE_RAM_GB) not a magic number sprinkled in the
    code. Defence against future regression where someone might
    accidentally tighten or loosen the floor without realizing it's a
    safety surface."""
    assert _MIN_SAFE_AVAILABLE_RAM_GB == 4.0


# ── run_with_stderr_redaction integration: safety net fires BEFORE Popen ──


def test_safety_net_fires_before_popen_in_wrapper(monkeypatch):
    """When the safety net trips, ``run_with_stderr_redaction`` raises
    BEFORE any ``subprocess.Popen`` call. Critical invariant: no ML
    process EVER spawns into a low-RAM state — that's the whole point
    of the safety net. Pre-fix the wrapper would have called
    subprocess.Popen first, loaded FLUX weights, THEN crashed somewhere
    downstream.
    """
    import imgen.checks as checks_mod
    import imgen.subprocess_helpers as sh

    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 1.0),
    )

    popen_calls = []

    class FakePopen:
        def __init__(self, *a, **kw):
            popen_calls.append((a, kw))

    monkeypatch.setattr(sh.subprocess, "Popen", FakePopen)

    with pytest.raises(InsufficientRAMError):
        run_with_stderr_redaction(
            ["echo", "hello"], env={"PATH": "/usr/bin"},
        )
    # CRITICAL invariant: Popen NEVER called
    assert popen_calls == [], (
        "subprocess.Popen was called despite RAM safety net raising — "
        "ML weights could have been loaded into dangerous memory state"
    )


# ── cmd_helpers.run_one_iteration catches InsufficientRAMError ───────


def test_run_one_iteration_catches_ram_safety_failure(
    monkeypatch, tmp_path,
):
    """When the safety net trips inside run_with_stderr_redaction
    (called from inside run_one_iteration), the orchestrator catches
    InsufficientRAMError, writes a status="failed" history entry, and
    returns True (continue batch). Mirrors the KeyboardInterrupt catch
    pattern so a low-RAM situation gets the same UX treatment as user
    cancellation — visible, recorded, doesn't kill the parent."""
    import imgen.checks as checks_mod
    import imgen.cmd_helpers as ch
    import imgen.subprocess_helpers as sh

    # Force the safety net to trip.
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 1.0),
    )

    # Prevent Popen even if the safety net somehow doesn't fire.
    class FakePopen:
        def __init__(self, *a, **kw):
            raise AssertionError(
                "Popen called despite safety net — invariant violated"
            )
    monkeypatch.setattr(sh.subprocess, "Popen", FakePopen)

    from pathlib import Path
    from imgen.engines.base import GenParams
    from imgen.models import BUILTIN_MODELS
    from imgen.runs import BatchContext, Iteration

    # v0.8.3 M-NEW-C: run_one_iteration now hard-requires Iteration
    # to carry both ``model`` and ``params`` (legacy fallback was
    # retired). Populate with BUILTIN_MODELS["flux-kontext"] so the
    # MfluxEngine.run path is reached — the safety net inside
    # ``subprocess_helpers.run_with_stderr_redaction`` trips before
    # the (stubbed-to-explode) Popen ever runs.
    model = BUILTIN_MODELS["flux-kontext"]
    params = GenParams(
        prompt="test", negative="",
        width=1024, height=1024,
        steps=20, guidance=3.5, seed=42, quantize=4, strength=0.55,
        input_path=tmp_path / "in.jpg",
        output_path=tmp_path / "out.png",
        loras=(),
    )
    iteration = Iteration(
        style_name="anime",
        prompt="test", negative="",
        final_steps=20, final_quantize=4,
        final_guidance=3.5, final_strength=0.55,
        output_path=tmp_path / "out.png",
        cmd=["echo", "hello"],
        model=model, params=params,
    )

    from types import SimpleNamespace
    ctx = BatchContext(
        model="flux", seed=42, width=1024, height=1024,
        input_path=tmp_path / "in.jpg",
        effective_custom_prompt=None,
        args=SimpleNamespace(scope=None, preview=False),
        batch_id=None, env={"PATH": "/usr/bin"},
    )

    succeeded: list = []
    failed: list = []
    keep_going = ch.run_one_iteration(
        it=iteration, idx=1, total=1, is_batch=False,
        ctx=ctx, logger=None,
        succeeded=succeeded, failed=failed,
    )
    assert keep_going is True, (
        "must continue batch loop after RAM safety failure "
        "(matches KeyboardInterrupt UX shape)"
    )
    assert len(failed) == 1
    assert succeeded == []

    # History entry recorded the refusal
    from imgen.history import load_history
    entries = load_history()
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"


# ── M-NEW-A: enhance subprocess also covered by safety net ───────────


def test_safety_net_covers_enhance_subprocess(monkeypatch):
    """v0.8.2 §R.4 M-NEW-A closure: ``enhance_runtime.run_with_mlx_lm``
    uses ``subprocess.run`` directly (synchronous + small payload), not
    the ``run_with_stderr_redaction`` wrapper. Pre-fix the safety net
    missed it — Qwen2.5-7B (~4 GB) could load into <4 GB available RAM
    and swap-thrash. Fix: call ``_assert_safe_ram_or_raise()`` at the
    enhance call site, lifting the same hard-floor check into the
    enhance path.

    Lock-in: with RAM artificially constrained, ``run_with_mlx_lm``
    raises InsufficientRAMError BEFORE any ``subprocess.run`` call.
    The orchestrator catches via the existing RunnerError flow
    (EnhanceResult.fallback_reason="runner_error") so user sees the
    cause + falls back to original prompt — same UX as a timeout."""
    import imgen.checks as checks_mod
    from imgen.enhance_runtime import run_with_mlx_lm
    import imgen.enhance_runtime as er_mod

    # Force the safety net to trip
    monkeypatch.setattr(
        checks_mod, "get_memory_gb", lambda: (32.0, 1.0),
    )

    # Prevent subprocess.run even if the safety net somehow misses
    def fake_subprocess_run(*a, **kw):
        raise AssertionError(
            "subprocess.run called despite safety net — invariant violated"
        )
    monkeypatch.setattr(er_mod.subprocess, "run", fake_subprocess_run)

    with pytest.raises(InsufficientRAMError):
        run_with_mlx_lm(
            items=[{"system": "enhance this", "user": "samurai"}],
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            temperature=0.3,
            max_tokens=128,
            timeout=120,
        )

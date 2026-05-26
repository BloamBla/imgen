"""End-to-end integration for the v0.5 LLM prompt enhancer in cmd_batch.

Mirrors the single-file path tests in tests/test_generate_enhance.py
but exercises the N×M batch flow. Stubs the orchestrator at the
``enhance_iteration_prompts`` seam so no real mlx_lm load happens
during the suite — manual smoke on real Qwen2.5-7B-4bit is the
mandatory pre-tag step.

Verifies the WIRING:

* ``--enhance-prompt`` runs the LLM ONCE for the whole N×M batch
  (single mlx_lm.load amortised across all prompts).
* Every iteration's mflux invocation receives the LLM-enhanced
  prompt for THAT (input, style) pair — the per-iteration history
  entry records ``prompt_original`` + ``enhanced`` + ``enhance_model``
  + ``enhance_fallback_reason`` aligned with the iteration's slot.
* ``--no-enhance`` bypasses the LLM entirely (orchestrator never
  called) — every history entry still records ``enhanced=False`` +
  ``user_opt_out``.
* Per-iteration fallback (mock returns ``invariant_violated`` for
  a subset) keeps the batch alive; mflux runs on the originals;
  history records the per-row reason.
* Runner-level failure (all-or-nothing fallback) — every iteration
  falls back to original; history rows uniformly record
  ``runner_error``.
"""
from __future__ import annotations

import subprocess
from dataclasses import FrozenInstanceError as dataclasses_FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.backends import BACKENDS
from imgen.commands.batch import cmd_batch
from imgen.defaults import DEFAULTS, HISTORY_SCHEMA_VERSION
from imgen.enhance import EnhanceResult


# ── Stubs (mirror tests/test_batch.py shape) ───────────────────────────


@pytest.fixture
def stub_mflux(monkeypatch):
    state: dict = {"returncode": 0, "calls": []}

    def fake_run(cmd, env, log_file=None):
        state["calls"].append({"cmd": cmd, "env": env, "log_file": log_file})
        return state["returncode"]

    # v0.8.3 M-NEW-C: single-patch — cmd_helpers no longer imports
    # run_with_stderr_redaction after the legacy fallback retirement.
    monkeypatch.setattr(
        "imgen.subprocess_helpers.run_with_stderr_redaction", fake_run
    )
    return state


@pytest.fixture
def stub_backend(monkeypatch, tmp_path):
    fake_binary = tmp_path / "fake-mflux"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)

    def fake_load(args):
        return ("flux", BACKENDS["flux"], "hf_faketoken", fake_binary, None)

    monkeypatch.setattr(
        "imgen.commands.batch.load_backend_and_token", fake_load
    )


@pytest.fixture
def stub_dims(monkeypatch):
    monkeypatch.setattr(
        "imgen.commands.batch.detect_resolution",
        lambda path, preview=False: (1024, 1024),
    )


@pytest.fixture
def stub_finder(monkeypatch):
    monkeypatch.setattr(
        "imgen.commands.batch.open_results", lambda **k: None
    )


@pytest.fixture
def stub_sips(monkeypatch):
    """Defensive: tests use .jpg inputs so sips shouldn't fire."""
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    return calls


def _args(*, directory: Path, output_dir: Path, **overrides) -> SimpleNamespace:
    """Default args namespace mimicking parser output + v0.5 enhance
    fields. Identical to test_generate_enhance._gen_args but adapted
    for the batch subcommand surface."""
    defaults: dict = dict(
        directory=str(directory),
        output_dir=str(output_dir),
        style=None,
        custom_prompt=None,
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        seed=42,
        preview=False,
        model="flux",
        scope="scene",
        width=None, height=None,
        force=True,
        yes=True,
        no_open=True,
        dry_run=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        # v0.5 enhance fields.
        enhance=None,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_input_dir(tmp_path: Path, *names: str) -> Path:
    """Create a directory with N stub .jpg inputs. Each file is just
    a tiny byte payload — discover_inputs glob-filters by extension,
    not by content."""
    d = tmp_path / "inputs"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"jpg-bytes")
    return d


# ── Orchestrator stub helpers ─────────────────────────────────────────


def _make_stub_orchestrator(
    monkeypatch,
    *,
    transform=None,
    fallback_reason=None,
    fallback_detail=None,
):
    """Patch ``imgen.cmd_helpers.enhance_iteration_prompts`` at the
    import site that ``maybe_enhance_for_command`` uses. Returns the
    calls list for inspection. Mirror of the helper in
    tests/test_generate_enhance.py — keeps both surfaces aligned.

    v0.6.5 (architect IMP-1 lock-in): ``fallback_detail`` optional —
    forwarded to the produced EnhanceResult so batch-side tests can
    exercise the runner_error + detail wire-up symmetrically with
    test_generate_enhance.py."""
    if transform is None:
        def transform(p):  # noqa: E306
            return f"ENH: {p}"

    calls: list = []

    def fake_orchestrator(*, iteration_prompts, system_prompt, invariants,
                          model, temperature, max_tokens, timeout_s):
        calls.append({
            "prompts": iteration_prompts,
            "system_prompt": system_prompt,
            "invariants": invariants,
            "model": model,
            "count": len(iteration_prompts),
        })
        results = []
        for p in iteration_prompts:
            if fallback_reason is not None:
                results.append(EnhanceResult(
                    final_prompt=p,
                    original_prompt=p,
                    was_enhanced=False,
                    fallback_reason=fallback_reason,
                    was_truncated=False,
                    raw_llm_output=None,
                    fallback_detail=fallback_detail,
                ))
            else:
                results.append(EnhanceResult(
                    final_prompt=transform(p),
                    original_prompt=p,
                    was_enhanced=True,
                    fallback_reason=None,
                    was_truncated=False,
                    raw_llm_output=transform(p),
                ))
        return results

    monkeypatch.setattr(
        "imgen.cmd_helpers.enhance_iteration_prompts", fake_orchestrator
    )
    return calls


# ── Happy path: --enhance-prompt feeds enhanced prompts to every mflux ──


def test_enhance_prompt_runs_once_for_whole_N_by_M_batch(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """Single mlx_lm.load amortises across the whole batch: 3 inputs
    × 2 styles = 6 prompts, but ONE orchestrator call. This is the
    core batch-mode efficiency claim — cold-load cost paid once,
    inference paid per-prompt."""
    orchestrator_calls = _make_stub_orchestrator(monkeypatch)

    input_dir = _make_input_dir(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
        enhance=True,
    )

    rc = cmd_batch(args)

    assert rc == 0
    # Exactly ONE orchestrator call — for the entire N×M batch.
    assert len(orchestrator_calls) == 1
    # That one call carried all 3 × 2 = 6 prompts.
    assert orchestrator_calls[0]["count"] == 6
    # And mflux was invoked 6 times.
    assert len(stub_mflux["calls"]) == 6


def test_enhance_feeds_enhanced_prompt_to_every_iteration(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """Every mflux invocation in the batch must receive the LLM-
    enhanced version (not the pre-enhance original). Alignment matters:
    iteration N gets enhance_results[N-1], not some other index."""
    _make_stub_orchestrator(monkeypatch)

    input_dir = _make_input_dir(tmp_path, "x.jpg", "y.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime", "ghibli", "pixar"],
        enhance=True,
    )

    cmd_batch(args)

    # 2 inputs × 3 styles = 6 mflux invocations, every prompt enhanced.
    assert len(stub_mflux["calls"]) == 6
    for call in stub_mflux["calls"]:
        cmd = call["cmd"]
        prompt_idx = cmd.index("--prompt") + 1
        assert cmd[prompt_idx].startswith("ENH: "), (
            f"non-enhanced prompt reached mflux: {cmd[prompt_idx][:80]!r}"
        )


def test_enhance_history_v2_fields_recorded_per_iteration(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """v=2 history entries — one per N×M iteration — each carry the
    enhance fields aligned with their own (input, style) prompt."""
    _make_stub_orchestrator(monkeypatch)
    input_dir = _make_input_dir(tmp_path, "p.jpg", "q.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=True,
    )

    cmd_batch(args)

    from imgen.history import load_history
    entries = load_history()
    assert len(entries) == 2  # 2 inputs × 1 style
    for e in entries:
        assert e["v"] == HISTORY_SCHEMA_VERSION
        assert e["enhanced"] is True
        assert e["enhance_model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert e["enhance_fallback_reason"] is None
        # The stored prompt is the enhanced version; prompt_original
        # is the pre-LLM construction.
        assert e["prompt"].startswith("ENH: ")
        assert e["prompt_original"] != e["prompt"]
        assert not e["prompt_original"].startswith("ENH: ")


# ── Opt-out paths ─────────────────────────────────────────────────────


def test_no_enhance_skips_llm_for_whole_batch(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """--no-enhance: orchestrator never called. mflux sees pre-enhance
    prompts. History records ``user_opt_out`` on every entry."""
    orchestrator_calls = _make_stub_orchestrator(monkeypatch)
    input_dir = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=False,
    )

    cmd_batch(args)

    assert orchestrator_calls == []  # LLM never invoked
    for call in stub_mflux["calls"]:
        cmd = call["cmd"]
        prompt_idx = cmd.index("--prompt") + 1
        assert not cmd[prompt_idx].startswith("ENH:")

    from imgen.history import load_history
    for e in load_history():
        assert e["v"] == HISTORY_SCHEMA_VERSION
        assert e["enhanced"] is False
        assert e["enhance_fallback_reason"] == "user_opt_out"
        assert e["enhance_model"] is None


def test_default_off_skips_llm(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """No flag = opt-in default off. Identical observable behaviour to
    --no-enhance from the LLM-not-called perspective."""
    orchestrator_calls = _make_stub_orchestrator(monkeypatch)
    input_dir = _make_input_dir(tmp_path, "a.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=None,
    )

    cmd_batch(args)

    assert orchestrator_calls == []
    cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = cmd.index("--prompt") + 1
    assert not cmd[prompt_idx].startswith("ENH:")


# ── Fallback paths ────────────────────────────────────────────────────


def test_per_iteration_invariant_violation_falls_back(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """Orchestrator returns ``invariant_violated`` for every prompt.
    Batch keeps running; every mflux sees the ORIGINAL prompt; every
    history entry records the diagnostic reason."""
    _make_stub_orchestrator(
        monkeypatch, fallback_reason="invariant_violated"
    )
    input_dir = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=True,
    )

    rc = cmd_batch(args)

    assert rc == 0  # batch still succeeded
    for call in stub_mflux["calls"]:
        cmd = call["cmd"]
        prompt_idx = cmd.index("--prompt") + 1
        assert not cmd[prompt_idx].startswith("ENH:")

    from imgen.history import load_history
    for e in load_history():
        assert e["enhanced"] is False
        assert e["enhance_fallback_reason"] == "invariant_violated"
        assert e["enhance_model"] is None


def test_runner_error_falls_back_consistently_across_batch(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch,
):
    """Orchestrator-level RunnerError (mlx_lm load failed, timeout,
    crash) → every iteration in the batch records ``runner_error``.
    mflux still runs on originals. The "all-or-nothing" nature
    matches user expectation: you get either consistent enhancement
    across the whole batch or none at all, never partial-by-chance."""
    _make_stub_orchestrator(monkeypatch, fallback_reason="runner_error")
    input_dir = _make_input_dir(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
        enhance=True,
    )

    rc = cmd_batch(args)

    assert rc == 0
    # 3 inputs × 2 styles = 6 mflux invocations, all on originals.
    assert len(stub_mflux["calls"]) == 6
    for call in stub_mflux["calls"]:
        cmd = call["cmd"]
        prompt_idx = cmd.index("--prompt") + 1
        assert not cmd[prompt_idx].startswith("ENH:")

    from imgen.history import load_history
    entries = load_history()
    assert len(entries) == 6
    assert {e["enhance_fallback_reason"] for e in entries} == {"runner_error"}


def test_runner_error_warn_reads_fallback_detail_in_batch(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch, capsys,
):
    """v0.6.5 (architect IMP-1) batch-side mirror of the
    test_generate_enhance.py lock-in: the shared
    ``maybe_enhance_for_command`` warn line reads ``fallback_detail``
    (the verbose diagnostic) — including via the cmd_batch entry path.
    Locks the wire-up at the batch surface so a future regression
    that's only exercised under N×M wouldn't sneak through the
    cmd_generate-only test."""
    _make_stub_orchestrator(
        monkeypatch,
        fallback_reason="runner_error",
        fallback_detail="mlx_lm.load: model 'fake/model' not found",
    )
    input_dir = _make_input_dir(tmp_path, "a.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=True,
    )

    rc = cmd_batch(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Enhance runner failed" in out
    # The diagnostic from fallback_detail surfaces in the warn line —
    # !r-escaped (security IMP-2 pattern). Checks the unquoted substring
    # so the assertion stays robust to repr()'s quote-style choice.
    assert "mlx_lm.load: model" in out
    assert "not found" in out


# ── Dry-run + enhance ─────────────────────────────────────────────────


def test_dry_run_with_enhance_shows_enhanced_prompts(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_finder, stub_sips, monkeypatch, capsys,
):
    """--dry-run + --enhance-prompt: orchestrator IS called; dry-run
    output contains the enhanced prompts (matches what mflux would
    actually receive). No real mflux invocation, no history written."""
    _make_stub_orchestrator(monkeypatch)
    input_dir = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=input_dir, output_dir=tmp_path / "out",
        style=["anime"],
        enhance=True,
        dry_run=True,
    )

    rc = cmd_batch(args)
    out = capsys.readouterr().out

    assert rc == 0
    # Both iterations' enhanced prompts visible in dry-run output.
    assert out.count("ENH:") >= 2
    # No real mflux + no history.
    assert stub_mflux["calls"] == []
    from imgen.history import load_history
    assert load_history() == []


# ── apply_enhance_results_to_groups — pure (v0.6.4 IMP #2) ──────────


def test_old_alias_removed():
    """v0.7.1 dropped the temporary `apply_enhance_results_to_per_input`
    backward-compat alias; v0.7.4 lost the lock-in test when the
    test_iteration_group.py file was deleted alongside the Protocol/
    DrawIterationGroup retirement. Re-establish here so a future
    copy-paste from a pre-v0.7.1 doc doesn't silently resurrect the
    legacy name."""
    import imgen.cmd_helpers as ch
    assert not hasattr(ch, "apply_enhance_results_to_per_input")


class TestApplyEnhanceResultsToPerInput:
    """v0.6.4 v0.5 architect IMP #2: the cmd_batch sliding-cursor block
    that re-bucketed a flat enhance_results list back into the per-
    input-tuple shape moved into a dedicated helper. These tests cover
    the pure function in isolation (no cmd_batch / no subprocess).

    v0.6.5 (architect IMP-3): the per-input shape was promoted from a
    bare 5-tuple to :class:`~imgen.runs.PerInputBatch`. Tests construct
    PerInputBatch instances explicitly and assert via named-field
    access (``pib.input_path``, ``pib.iters``)."""

    def _make_iter(self, prompt: str, style: str = "anime") -> "Iteration":
        from imgen.runs import Iteration
        return Iteration(
            style_name=style,
            prompt=prompt,
            negative="",
            final_steps=20,
            final_quantize=4,
            final_guidance=4.0,
            final_strength=0.6,
            output_path=Path(f"/tmp/out-{style}.png"),
            cmd=["fake-mflux", "--prompt", prompt],
        )

    def _make_result(self, original: str, enhanced: str | None = None) -> EnhanceResult:
        if enhanced is None:
            return EnhanceResult(
                final_prompt=original,
                original_prompt=original,
                was_enhanced=False,
                fallback_reason="user_opt_out",
                was_truncated=False,
                raw_llm_output=None,
            )
        return EnhanceResult(
            final_prompt=enhanced,
            original_prompt=original,
            was_enhanced=True,
            fallback_reason=None,
            was_truncated=False,
            raw_llm_output=enhanced,
        )

    def _make_pib(self, input_name, *iters_pairs):
        """Build a PerInputBatch with the given (prompt, style) iters."""
        from imgen.runs import PerInputBatch
        return PerInputBatch(
            input_path=Path(f"/tmp/{input_name}.jpg"),
            mflux_input=Path(f"/tmp/conv-{input_name}.jpg"),
            width=640,
            height=896,
            iters=tuple(
                self._make_iter(prompt, style) for prompt, style in iters_pairs
            ),
        )

    def test_round_trips_uniform_2x3(self):
        """2 inputs × 3 styles = 6 flat results. Result re-buckets to
        per-input shape with 3 iterations each. Prompts spliced in."""
        from imgen.cmd_helpers import apply_enhance_results_to_groups
        per_input = [
            self._make_pib(
                "in1",
                ("p1-anime", "anime"),
                ("p1-ghibli", "ghibli"),
                ("p1-pixar", "pixar"),
            ),
            self._make_pib(
                "in2",
                ("p2-anime", "anime"),
                ("p2-ghibli", "ghibli"),
                ("p2-pixar", "pixar"),
            ),
        ]
        results = [
            self._make_result("p1-anime", "ENH p1-anime"),
            self._make_result("p1-ghibli", "ENH p1-ghibli"),
            self._make_result("p1-pixar", "ENH p1-pixar"),
            self._make_result("p2-anime"),  # fallback, no enhancement
            self._make_result("p2-ghibli", "ENH p2-ghibli"),
            self._make_result("p2-pixar", "ENH p2-pixar"),
        ]
        out = apply_enhance_results_to_groups(per_input, results)
        assert len(out) == 2
        # Path + dim metadata preserved on each PerInputBatch.
        for i in range(2):
            assert out[i].input_path == per_input[i].input_path
            assert out[i].mflux_input == per_input[i].mflux_input
            assert out[i].width == per_input[i].width
            assert out[i].height == per_input[i].height
        # Iteration groups have the right length.
        assert [len(pib.iters) for pib in out] == [3, 3]
        # Enhanced prompts spliced through; fallback iteration unchanged.
        assert out[0].iters[0].prompt == "ENH p1-anime"
        assert out[0].iters[1].prompt == "ENH p1-ghibli"
        assert out[0].iters[2].prompt == "ENH p1-pixar"
        assert out[1].iters[0].prompt == "p2-anime"  # fallback original
        assert out[1].iters[1].prompt == "ENH p2-ghibli"
        assert out[1].iters[2].prompt == "ENH p2-pixar"

    def test_ragged_groups_preserved(self):
        """Future per-style skip logic could produce ragged group
        lengths (input #1 has 3 iters, input #2 only 1). The cursor
        inside the helper handles this transparently."""
        from imgen.cmd_helpers import apply_enhance_results_to_groups
        per_input = [
            self._make_pib(
                "in1", ("p1-a", "anime"), ("p1-b", "anime"), ("p1-c", "anime"),
            ),
            self._make_pib("in2", ("p2-a", "anime")),
        ]
        results = [
            self._make_result("p1-a", "ENH p1-a"),
            self._make_result("p1-b", "ENH p1-b"),
            self._make_result("p1-c", "ENH p1-c"),
            self._make_result("p2-a", "ENH p2-a"),
        ]
        out = apply_enhance_results_to_groups(per_input, results)
        assert [len(pib.iters) for pib in out] == [3, 1]
        assert out[1].iters[0].prompt == "ENH p2-a"

    def test_empty_input_passes_through(self):
        """v0.6.4 python NIT-2: zero inputs with zero results is a
        defined edge of the contract (helper is documented as pure).
        ``sum([])==0==len([])`` so the guard passes; result is ``[]``.
        Lock-in so a future contract narrowing doesn't silently
        reject the empty case."""
        from imgen.cmd_helpers import apply_enhance_results_to_groups
        out = apply_enhance_results_to_groups([], [])
        assert out == []

    def test_count_mismatch_raises(self):
        """Misalignment between sum-of-group-lengths and flat results
        is loud, not silent — silent miswire would assign wrong
        prompts to wrong iterations."""
        from imgen.cmd_helpers import apply_enhance_results_to_groups
        per_input = [
            self._make_pib("in", ("a", "anime"), ("b", "anime")),
        ]
        # Only 1 result for 2 iterations.
        with pytest.raises(ValueError, match="enhance-result count mismatch"):
            apply_enhance_results_to_groups(
                per_input, [self._make_result("a")],
            )

    def test_returns_per_input_batch_instances(self):
        """v0.6.5 (architect IMP-3) lock-in: the helper's return shape
        is ``list[PerInputBatch]``, not the legacy ``list[tuple]``.
        Locks the dataclass-promotion so a future regression to tuple
        return wouldn't sneak past the per-prompt assertions in the
        round-trip tests above."""
        from imgen.cmd_helpers import apply_enhance_results_to_groups
        from imgen.runs import PerInputBatch
        per_input = [self._make_pib("in", ("a", "anime"))]
        results = [self._make_result("a", "ENH a")]
        out = apply_enhance_results_to_groups(per_input, results)
        assert len(out) == 1
        assert isinstance(out[0], PerInputBatch)
        # Frozen — attribute assignment raises.
        with pytest.raises((AttributeError, dataclasses_FrozenInstanceError)):
            out[0].width = 999  # type: ignore[misc]
        # iters is a tuple (immutable container).
        assert isinstance(out[0].iters, tuple)

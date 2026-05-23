"""End-to-end integration for the v0.5 LLM prompt enhancer in cmd_generate.

Stubs the LLM at the orchestrator seam (``enhance_iteration_prompts``)
so no real mlx_lm load / 4 GB download / 10-second cold start hits the
suite. Verifies the WIRING:

* ``--enhance-prompt`` causes mflux to receive the LLM-expanded prompt
  (not the pre-enhance one).
* History entry records ``prompt_original`` + ``enhanced`` +
  ``enhance_model`` + ``enhance_fallback_reason`` at the v=2 schema.
* ``--no-enhance`` bypasses the LLM entirely — assertion via mocked
  orchestrator that the stub was never called.
* Per-prompt fallback (invariant violation): mflux receives the
  ORIGINAL prompt, history records ``enhanced=False`` plus the
  diagnostic reason. The user still gets an image.
* Runner-level failure (RunnerError): all iterations fall back to
  originals consistently, history records ``runner_error``.

Mirrors the fixture style of ``tests/test_generate_heic.py``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.backends import BACKENDS
from imgen.commands.generate import cmd_generate
from imgen.defaults import DEFAULTS, HISTORY_SCHEMA_VERSION
from imgen.enhance import EnhanceResult


# ── Fixtures (mirror tests/test_generate_heic.py) ──────────────────────


@pytest.fixture
def stub_mflux(monkeypatch):
    state: dict = {"returncode": 0, "calls": []}

    def fake_run(cmd, env, log_file=None):
        state["calls"].append({"cmd": cmd, "env": env, "log_file": log_file})
        return state["returncode"]

    monkeypatch.setattr(
        "imgen.cmd_helpers.run_with_stderr_redaction", fake_run
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
        "imgen.commands.generate.load_backend_and_token", fake_load
    )


@pytest.fixture
def stub_dims(monkeypatch):
    monkeypatch.setattr(
        "imgen.commands.generate.detect_resolution",
        lambda path, preview=False: (1024, 1024),
    )


@pytest.fixture
def stub_open(monkeypatch):
    monkeypatch.setattr(
        "imgen.commands.generate.open_results", lambda **k: None
    )


@pytest.fixture
def stub_sips(monkeypatch):
    """Defensive: tests use .jpg inputs so sips shouldn't fire, but
    stubbed for symmetry with the HEIC fixture pattern."""
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    return calls


def _gen_args(*, image: Path, **overrides) -> SimpleNamespace:
    """Mirror of tests/test_generate_heic.py _gen_args + v0.5 enhance fields."""
    defaults = dict(
        image=str(image),
        style=["anime"],
        custom_prompt=None,
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        seed=42,
        preview=False,
        backend="flux",
        scope="scene",
        width=None, height=None,
        output=None,
        output_dir=None,
        force=True,
        yes=True,
        no_open=True,
        dry_run=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
        # v0.5 enhance CLI surface — defaults match "no flag passed".
        enhance=None,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Enhance orchestrator stub helpers ──────────────────────────────────


def _make_stub_orchestrator(
    monkeypatch,
    *,
    transform=None,
    fallback_reason=None,
    fallback_detail=None,
    raise_runner_error=False,
):
    """Patch the enhance orchestrator (``enhance_iteration_prompts``) at
    the import site cmd_helpers uses. ``transform`` is a callable that
    receives an original prompt and returns the LLM "enhancement"; if
    omitted, defaults to ``f"ENH: {original}"``. ``fallback_reason`` set
    forces every result to be a fallback (was_enhanced=False) with that
    reason. ``fallback_detail`` (v0.6.5) optionally supplies the verbose
    diagnostic string — primarily used to exercise the runner_error +
    detail wire-up. ``raise_runner_error`` instead raises RunnerError so
    we can exercise the all-or-nothing fallback path.

    Returns the calls list for inspection.
    """
    from imgen.enhance import RunnerError

    if transform is None:
        def transform(p):  # noqa: E306 — small inline default
            return f"ENH: {p}"

    calls: list = []

    def fake_orchestrator(*, iteration_prompts, system_prompt, invariants,
                          model, temperature, max_tokens, timeout_s):
        calls.append({
            "prompts": iteration_prompts,
            "system_prompt": system_prompt,
            "invariants": invariants,
            "model": model,
        })
        if raise_runner_error:
            raise RunnerError("simulated runner failure")
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


# ── Happy path: --enhance-prompt feeds enhanced prompt to mflux ─────────


def test_enhance_prompt_reaches_mflux(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """--enhance-prompt: the orchestrator transforms the prompt, mflux
    sees ONLY the transformed version."""
    transform = lambda p: f"Restyle this person preserving identity (HD anime detail). [from: {p[:30]}...]"  # noqa: E731
    orchestrator_calls = _make_stub_orchestrator(monkeypatch, transform=transform)

    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True)

    rc = cmd_generate(args)

    assert rc == 0
    # Orchestrator was called once with the iteration's pre-enhance
    # prompt.
    assert len(orchestrator_calls) == 1
    pre = orchestrator_calls[0]["prompts"][0]
    assert "Restyle" in pre  # from styles.py anime preset
    # mflux saw the enhanced version.
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = mflux_cmd.index("--prompt") + 1
    assert mflux_cmd[prompt_idx].startswith("Restyle this person preserving identity (HD")
    assert "[from:" in mflux_cmd[prompt_idx]


def test_enhance_records_v2_history_fields(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    _make_stub_orchestrator(monkeypatch)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True)

    cmd_generate(args)

    from imgen.history import load_history
    entries = load_history()
    assert len(entries) == 1
    e = entries[0]
    assert e["v"] == HISTORY_SCHEMA_VERSION
    assert e["enhanced"] is True
    assert e["enhance_model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert e["enhance_fallback_reason"] is None
    # prompt_original is the pre-enhance prompt; prompt is the post.
    assert e["prompt_original"] != e["prompt"]
    assert e["prompt"].startswith("ENH: ")
    assert "Restyle" in e["prompt_original"]


# ── Opt-out paths ──────────────────────────────────────────────────────


def test_no_enhance_skips_llm(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """--no-enhance: orchestrator must never be called; mflux sees the
    pre-enhance prompt unchanged."""
    orchestrator_calls = _make_stub_orchestrator(monkeypatch)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=False)

    cmd_generate(args)

    assert orchestrator_calls == []  # LLM never called
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = mflux_cmd.index("--prompt") + 1
    assert not mflux_cmd[prompt_idx].startswith("ENH:")


def test_default_off_skips_llm(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """No flag = no enhancement (opt-in default). Same observable
    behaviour as --no-enhance from the LLM-not-called perspective."""
    orchestrator_calls = _make_stub_orchestrator(monkeypatch)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=None)

    cmd_generate(args)

    assert orchestrator_calls == []
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = mflux_cmd.index("--prompt") + 1
    assert not mflux_cmd[prompt_idx].startswith("ENH:")


def test_opt_out_history_records_reason(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """When enhancer is off, every history entry records the opt-out so
    forensic readers can tell "did the user know about the enhancer?"
    apart from "this is a pre-v0.5 entry"."""
    _make_stub_orchestrator(monkeypatch)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=False)

    cmd_generate(args)

    from imgen.history import load_history
    e = load_history()[0]
    assert e["v"] == HISTORY_SCHEMA_VERSION
    assert e["enhanced"] is False
    assert e["enhance_model"] is None  # nothing was used
    assert e["enhance_fallback_reason"] == "user_opt_out"
    assert e["prompt_original"] == e["prompt"]


# ── Fallback paths ─────────────────────────────────────────────────────


def test_per_prompt_invariant_violation_falls_back(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """LLM returned an output that dropped 'preserving' — orchestrator
    falls back per-prompt. mflux runs on the ORIGINAL prompt; history
    records the diagnostic reason."""
    _make_stub_orchestrator(monkeypatch, fallback_reason="invariant_violated")
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True)

    rc = cmd_generate(args)

    assert rc == 0  # image still generated
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = mflux_cmd.index("--prompt") + 1
    assert not mflux_cmd[prompt_idx].startswith("ENH:")
    # History records the fallback.
    from imgen.history import load_history
    e = load_history()[0]
    assert e["enhanced"] is False
    assert e["enhance_fallback_reason"] == "invariant_violated"
    assert e["enhance_model"] is None


def test_runner_error_falls_back_all_with_diagnostic(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """Runner crashed (mlx_lm failed to load): all iterations fall back
    to originals consistently. mflux still runs.

    The real ``enhance_iteration_prompts`` catches RunnerError internally
    and produces ``runner_error`` fallback results — this test mocks
    the orchestrator one level up and produces the same shape directly
    so cmd_generate's integration with that fallback path is exercised.
    """
    _make_stub_orchestrator(
        monkeypatch, fallback_reason="runner_error",
    )

    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True)

    rc = cmd_generate(args)

    assert rc == 0
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    prompt_idx = mflux_cmd.index("--prompt") + 1
    assert not mflux_cmd[prompt_idx].startswith("ENH:")
    from imgen.history import load_history
    e = load_history()[0]
    assert e["enhanced"] is False
    assert e["enhance_fallback_reason"] == "runner_error"


def test_args_without_scope_attr_passes_through_run_one_iteration(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch,
):
    """v0.6.5 (architect IMP-A) end-to-end lock-in for the FL-3 closure:
    a Namespace WITHOUT a ``scope`` attribute (mirroring what the future
    ``imgen draw`` parser will produce) passes through cmd_generate
    cleanly — no AttributeError at:

      * ``_resolve_iteration_prompt`` (the FL-3 helper)
      * ``logger.write_header(scope=...)`` (architect IMP-A site #2)
      * ``run_one_iteration`` history entry (architect IMP-A site #3)

    The history row records ``scope=None`` — readers already use
    ``entry.get`` so absence-as-None lands cleanly. ``preview`` is NOT
    stripped because it's declared on both i2i and t2i parsers (initial
    image dimension shorthand vs t2i initial canvas size); only scope
    is i2i-only."""
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     style=["anime"], enhance=False)
    # Strip scope only — preview stays universal.
    del args.scope
    assert not hasattr(args, "scope")
    assert hasattr(args, "preview")

    rc = cmd_generate(args)
    assert rc == 0
    from imgen.history import load_history
    e = load_history()[0]
    assert e["scope"] is None
    assert e["preview"] is False


def test_runner_error_warn_reads_fallback_detail(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch, capsys,
):
    """v0.6.5 (architect IMP-1): the all-runner-error warn line in
    ``maybe_enhance_for_command`` reads ``fallback_detail`` (the verbose
    diagnostic) rather than ``raw_llm_output`` (which is now None for
    this path — the runner crashed before producing any LLM output).
    Locks the producer/consumer wire-up: the detail string the stub
    sets here surfaces in the warn line the user sees on real runner
    crashes."""
    _make_stub_orchestrator(
        monkeypatch,
        fallback_reason="runner_error",
        fallback_detail="mlx_lm.load: model 'fake/model' not found",
    )

    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True)

    rc = cmd_generate(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Enhance runner failed" in out
    # The diagnostic from fallback_detail surfaces in the warn line —
    # !r-escaped (security IMP-2 pattern: any control bytes coming from
    # mlx_lm / HF tracebacks become literal \x1b instead of clearing
    # the user's screen). repr() of a str-with-quotes uses the outer
    # quote that doesn't collide, so checking the unquoted substring
    # stays robust to that choice.
    assert "mlx_lm.load: model" in out
    assert "not found" in out


# ── Dry-run + enhance ──────────────────────────────────────────────────


def test_dry_run_with_enhance_shows_enhanced_prompt(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips, monkeypatch, capsys,
):
    """--dry-run + --enhance-prompt: the LLM IS called and the displayed
    cmd contains the enhanced prompt (matches what mflux would receive
    if --dry-run weren't set). Honest behaviour — surprising users
    with "dry-run shows pre-enhance" would be worse."""
    _make_stub_orchestrator(monkeypatch)
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    args = _gen_args(image=photo, output_dir=str(tmp_path / "out"),
                     enhance=True, dry_run=True)

    rc = cmd_generate(args)
    out = capsys.readouterr().out

    assert rc == 0
    assert "ENH:" in out  # enhanced prompt visible in dry-run cmd
    # No actual mflux invocation in dry-run.
    assert stub_mflux["calls"] == []
    # No history entry either (dry-run never writes).
    from imgen.history import load_history
    assert load_history() == []

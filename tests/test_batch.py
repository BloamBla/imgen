"""Integration tests for cmd_batch — v0.3.0's `imgen batch <dir>`.

cmd_batch composes existing single-input helpers (build_iterations,
preflight, _run_one_iteration, BatchLogger, _open_results, _exit_code)
around an outer loop over discovered inputs + a HEIC sips cache.
These tests exercise the orchestration end-to-end with stubbed mflux
and stubbed backend-loading so the suite stays GPU-free / network-free
/ subprocess-light (<2s target).

Stubbing surface (v0.3.1 post-cmd_helpers extraction):
  * ``imgen.cmd_helpers.run_with_stderr_redaction`` — fake mflux
    (cmd_helpers.run_one_iteration looks it up there now)
  * ``imgen.commands.batch.load_backend_and_token`` — bypass venv +
    binary existence checks (patched at batch.py call site since it
    imports by name)
  * ``imgen.images.detect_resolution`` — return fixed dims (no PIL)
  * ``imgen.commands.batch.open_results`` — skip macOS `open`
  * ``imgen.inputs.subprocess.run`` — fake sips for HEIC tests

State redirect via ``tmp_state_dir`` fixture so HISTORY_FILE + LOGS_DIR
+ STATE_DIR point inside the test's tmp_path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.backends import BACKENDS
from imgen.commands.batch import cmd_batch
from imgen.defaults import DEFAULTS


# ── Test fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def stub_mflux(monkeypatch):
    """Patch the real mflux invocation. State controls returncode and
    captures every call so tests can assert on cmd / env / log_file."""
    state: dict = {"returncode": 0, "raise": None, "calls": []}

    def fake_run(cmd, env, log_file=None):
        state["calls"].append({"cmd": cmd, "env": env, "log_file": log_file})
        if state["raise"] is not None:
            raise state["raise"]
        return state["returncode"]

    # v0.8.2 M-1C-prep: dual-patch both layers (legacy cmd_helpers
    # binding + canonical subprocess_helpers) so the fixture covers
    # both run_one_iteration paths post-M-1C-flip. Without the second
    # patch, engine.run iterations would slip past the stub and spawn
    # real mflux subprocesses — v0.8.2 ops post-mortem 2026-05-26.
    monkeypatch.setattr(
        "imgen.cmd_helpers.run_with_stderr_redaction", fake_run
    )
    monkeypatch.setattr(
        "imgen.subprocess_helpers.run_with_stderr_redaction", fake_run
    )
    return state


@pytest.fixture
def stub_backend(monkeypatch, tmp_path):
    """Bypass _load_backend_and_token — no venv, no mflux binary on disk,
    no HF token in the test env. Returns the production Backend dataclass
    (so _build_iterations gets a valid `be` shape) with a fake binary
    path that the test will never actually exec."""
    fake_binary = tmp_path / "fake-mflux"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)

    def fake_load(args) -> tuple[str, object, str | None, Path, tuple[str, str] | None]:
        be = BACKENDS["flux"]
        # v0.4: 5th element is the custom-backend secret tuple (None for FLUX).
        return ("flux", be, "hf_faketoken", fake_binary, None)

    monkeypatch.setattr(
        "imgen.commands.batch.load_backend_and_token", fake_load
    )
    return fake_binary


@pytest.fixture
def stub_dims(monkeypatch):
    """detect_resolution would shell out to venv python+PIL; pin to a
    constant so the suite never touches a real image."""
    monkeypatch.setattr(
        "imgen.commands.batch.detect_resolution",
        lambda path, preview=False: (1024, 1024),
    )


@pytest.fixture
def stub_finder(monkeypatch):
    """Don't open Finder during tests."""
    monkeypatch.setattr("imgen.commands.batch.open_results", lambda **k: None)


@pytest.fixture
def _batch_env(
    tmp_state_dir, stub_mflux, stub_backend, stub_dims, stub_finder
):
    """Compose all the stubs needed for a baseline cmd_batch run.
    Returns the stub_mflux state dict for assertions."""
    return stub_mflux


def _make_input_dir(tmp_path: Path, *names: str) -> Path:
    """Create a directory with named (zero-content) image files. Names
    drive the test scenario — content is never read since mflux is
    stubbed."""
    d = tmp_path / "inputs"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"\x00")
    return d


def _args(*, directory: Path, output_dir: Path, **overrides) -> SimpleNamespace:
    """Default args namespace mimicking what the parser would produce
    for `imgen batch <dir>` with all knobs at their defaults."""
    defaults: dict = dict(
        directory=str(directory),
        output_dir=str(output_dir),
        style=None,            # → falls back to merged_defaults["style"]
        custom_prompt=None,
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        strength=None,
        seed=42,
        preview=False,
        model="flux",
        scope=None,
        width=None, height=None,
        force=True,            # skip resource preflight (no real RAM check)
        yes=True,              # skip confirm gate
        no_open=True,          # _open_results stubbed anyway
        dry_run=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Error paths ─────────────────────────────────────────────────────────


def test_cmd_batch_non_existent_dir_exits_2(tmp_path, _batch_env):
    args = _args(directory=tmp_path / "missing", output_dir=tmp_path / "out")
    with pytest.raises(SystemExit) as exc:
        cmd_batch(args)
    assert exc.value.code == 2


def test_cmd_batch_not_a_dir_exits_2(tmp_path, _batch_env):
    """User typed a single file path — batch is dir-only; clean error
    instead of treating it as a one-image batch."""
    f = tmp_path / "single.jpg"
    f.write_bytes(b"")
    args = _args(directory=f, output_dir=tmp_path / "out")
    with pytest.raises(SystemExit) as exc:
        cmd_batch(args)
    assert exc.value.code == 2


def test_cmd_batch_empty_dir_exits_2_with_hint(tmp_path, _batch_env, capsys):
    """Directory exists but holds no supported images — surface the
    user-facing "0 supported images" message + extension hint."""
    d = tmp_path / "empty"
    d.mkdir()
    args = _args(directory=d, output_dir=tmp_path / "out")
    with pytest.raises(SystemExit) as exc:
        cmd_batch(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "0 supported images" in err


def test_cmd_batch_stem_collision_exits_2(tmp_path, _batch_env, capsys):
    """`IMG_1234.heic` + `IMG_1234.jpg` would overwrite under the flat
    output layout — caught in preflight, no mflux invocations.

    No sips stub needed: check_input_stems dies before
    resolve_to_mflux_input is reached."""
    d = _make_input_dir(tmp_path, "IMG_1234.heic", "IMG_1234.jpg")
    args = _args(directory=d, output_dir=tmp_path / "out")
    with pytest.raises(SystemExit) as exc:
        cmd_batch(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "IMG_1234" in err


# ── Happy paths ─────────────────────────────────────────────────────────


def test_cmd_batch_two_inputs_one_style_returns_zero(
    tmp_path, _batch_env
):
    """N=2, M=1 → 2 mflux invocations, exit 0, both outputs land in
    run_dir, history has 2 success entries."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert len(_batch_env["calls"]) == 2


# ── v0.7.13 (gap 8): bare mode behaviour pivot ──────────────────────────


def test_cmd_batch_no_style_no_prompt_dies_with_hint(
    tmp_path, _batch_env, capsys
):
    """v0.7.13 (gap 8 die path): bare `imgen batch <dir>` with no
    --style AND no --custom-prompt / --prompt-file → die code 2 + hint
    mentioning both opt-ins. Pre-v0.7.13 this fell back to the default
    style (pixar) silently."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(directory=d, output_dir=tmp_path / "out")  # style=None default
    with pytest.raises(SystemExit) as exc:
        cmd_batch(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--style" in err
    assert "--custom-prompt" in err
    # No mflux invocations — we died before backend resolution.
    assert len(_batch_env["calls"]) == 0


def test_cmd_batch_no_style_with_custom_prompt_uses_bare_path(
    tmp_path, _batch_env
):
    """v0.7.13 (gap 8 happy path): bare mode produces one iteration
    per input with style_name="bare", output named <stem>-bare.png,
    and no preset negative_prompt leaks into argv. Closes the silent
    pixar-default-fallback footgun for i2i."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        custom_prompt="a samurai portrait, dramatic lighting",
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert len(_batch_env["calls"]) == 2
    # Argv carries the user's bare prompt verbatim — no preset prefix /
    # scope rewrite / negative-prompt leak. flux backend supports
    # negatives, but with no preset there's no negative to emit.
    for call in _batch_env["calls"]:
        cmd = call["cmd"]
        prompt_idx = cmd.index("--prompt") + 1
        assert cmd[prompt_idx] == "a samurai portrait, dramatic lighting"
        assert "--negative-prompt" not in cmd
    # History records style=None (v0.3.5 semantics: when custom_prompt
    # is set, the per-iteration style is recorded as None because the
    # user's text drives the prompt content). v0.7.13 keeps this shape
    # — replay routes None-style-with-custom-prompt through bare mode
    # anyway (cleaner: schema v=3 needs no bump). custom_prompt field
    # carries the actual prompt verbatim for replay correctness.
    from imgen.history import load_history
    entries = load_history()
    assert len(entries) == 2
    assert all(e["style"] is None for e in entries)
    assert all(
        e["custom_prompt"] == "a samurai portrait, dramatic lighting"
        for e in entries
    )
    # All in same batch — bare mode still uses batch_id when N>1 inputs.
    batch_ids = {e["batch_id"] for e in entries}
    assert len(batch_ids) == 1


def test_cmd_batch_two_inputs_two_styles_exit_0_and_4_invocations(
    tmp_path, _batch_env
):
    """N×M = 2×2 → 4 mflux invocations in input-major order
    (a×anime, a×ghibli, b×anime, b×ghibli)."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert len(_batch_env["calls"]) == 4


def test_cmd_batch_writes_history_entries_for_every_iteration(
    tmp_path, _batch_env
):
    """Every (input, style) pair appends a history entry — load_history
    sees N×M = 4 success rows with the shared batch_id."""
    from imgen.history import load_history
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    cmd_batch(args)
    entries = load_history()
    assert len(entries) == 4
    batch_ids = {e["batch_id"] for e in entries}
    assert len(batch_ids) == 1, \
        "all entries must share one batch_id"
    assert all(e["status"] == "success" for e in entries)


def test_cmd_batch_log_has_input_sections(tmp_path, _batch_env):
    """BatchLogger writes `=== INPUT <name> ===` open + close markers
    around each input's M iterations. Check both inputs got bracketed
    in the persistent log file."""
    from imgen.runs import LOGS_DIR
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime"],
    )
    cmd_batch(args)
    # Find the one log file written this batch.
    logs = list(LOGS_DIR.glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_bytes().decode()
    assert "=== INPUT a.jpg (1/2) ===" in content
    assert "=== INPUT b.jpg (2/2) ===" in content
    # closing markers list per-input ok/fail counts (1/1 ok each).
    assert "INPUT a.jpg → 1/1 ok" in content
    assert "INPUT b.jpg → 1/1 ok" in content


def test_cmd_batch_log_global_iteration_numbering(tmp_path, _batch_env):
    """N×M = 2×2 → 4 iterations numbered 1..4 (not 1..2 per input).
    Lets users grep `[3/4]` to find one specific generation."""
    from imgen.runs import LOGS_DIR
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    cmd_batch(args)
    content = list(LOGS_DIR.glob("*.log"))[0].read_bytes().decode()
    assert "[1/4]" in content
    assert "[2/4]" in content
    assert "[3/4]" in content
    assert "[4/4]" in content


def test_cmd_batch_global_idx_uses_flat_counter_not_formula(
    tmp_path, _batch_env, monkeypatch
):
    """v0.3.0 python review IMP-2: `global_idx` must be a flat counter,
    not `(n-1)*len(styles_list) + m`, so it stays correct if a future
    change ever makes _build_iterations return a short group for some
    input. We can't easily simulate "short group" without rebuilding
    _build_iterations, but we CAN assert that the indices passed to
    _run_one_iteration are a clean 1, 2, 3, … sequence across the
    whole batch — locking the contract independent of the formula."""
    received_idx: list[int] = []
    received_total: list[int] = []

    from imgen.cmd_helpers import run_one_iteration as real_run_one

    def spy(*args, **kwargs):
        received_idx.append(kwargs["idx"])
        received_total.append(kwargs["total"])
        return real_run_one(*args, **kwargs)

    monkeypatch.setattr("imgen.commands.batch.run_one_iteration", spy)

    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    cmd_batch(args)
    # 3 inputs × 2 styles = 6 iterations, numbered exactly 1..6.
    assert received_idx == [1, 2, 3, 4, 5, 6]
    # `total` is constant — the full N×M grid size.
    assert received_total == [6, 6, 6, 6, 6, 6]


def test_cmd_batch_output_files_use_flat_layout(tmp_path, _batch_env):
    """Iteration.cmd's `--output PATH` (built by _build_iterations) is
    `<run_dir>/<stem>-<style>.png` — no per-input subdir."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    cmd_batch(args)
    # Extract --output value from each captured mflux cmd.
    outputs = []
    for call in _batch_env["calls"]:
        cmd = call["cmd"]
        i = cmd.index("--output")
        outputs.append(Path(cmd[i + 1]).name)
    assert set(outputs) == {
        "a-anime.png", "a-ghibli.png", "b-anime.png", "b-ghibli.png",
    }


def test_cmd_batch_partial_failure_returns_exit_5(
    tmp_path, _batch_env, monkeypatch,
):
    """N×M=4 with first 2 succeeding + last 2 failing → mixed batch
    → exit code 5 (partial). Lets calling scripts distinguish all-ok
    (0) / all-failed (1) / partial (5) without parsing output."""
    returncodes = iter([0, 0, 7, 7])
    real_calls = _batch_env["calls"]

    def varying_run(cmd, env, log_file=None):
        rc = next(returncodes)
        real_calls.append({"cmd": cmd, "env": env, "log_file": log_file})
        return rc

    # Re-override the stub via monkeypatch (not bare module-attr
    # assignment) so teardown restores the original — otherwise an
    # assertion failure here would clobber run_with_stderr_redaction
    # for the rest of the suite.
    monkeypatch.setattr(
        "imgen.cmd_helpers.run_with_stderr_redaction", varying_run
    )

    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"],
    )
    rc = cmd_batch(args)
    assert rc == 5


def test_cmd_batch_all_failed_returns_exit_1(tmp_path, _batch_env):
    """All-failed batch → exit 1 (not 5)."""
    _batch_env["returncode"] = 9
    d = _make_input_dir(tmp_path, "a.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    assert cmd_batch(args) == 1


# ── Dry-run ─────────────────────────────────────────────────────────────


def test_cmd_batch_dry_run_lists_all_iterations_no_mflux(
    tmp_path, _batch_env, capsys
):
    """--dry-run prints every cmd, exits 0 cleanly, never invokes mflux,
    no log file, no history. Lets users sanity-check N×M before paying
    the wall-clock cost."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"], dry_run=True,
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert _batch_env["calls"] == []
    out = capsys.readouterr().out
    # 4 iterations listed.
    assert out.count("Dry run [") == 4
    assert "[1/4]" in out and "[4/4]" in out


# ── HEIC support ────────────────────────────────────────────────────────


@pytest.fixture
def stub_sips(monkeypatch):
    """Fake sips for HEIC tests. Touches the --out target so subsequent
    code that may stat() the converted file sees it exist."""
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    return calls


def test_cmd_batch_heic_input_runs_sips_before_mflux(
    tmp_path, _batch_env, stub_sips
):
    """HEIC input → sips invoked once to produce a JPEG → mflux gets
    the JPEG path in its --image-path arg (not the raw HEIC)."""
    d = _make_input_dir(tmp_path, "IMG_1234.heic")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert len(stub_sips) == 1
    # sips converted IMG_1234.heic → <cache>/IMG_1234.jpg
    sips_cmd = stub_sips[0]
    assert sips_cmd[:5] == ["sips", "-s", "format", "jpeg", str(d / "IMG_1234.heic")]
    out_path = Path(sips_cmd[sips_cmd.index("--out") + 1])
    assert out_path.name == "IMG_1234.jpg"
    # mflux was passed the converted path, not the original HEIC.
    mflux_cmd = _batch_env["calls"][0]["cmd"]
    image_arg = mflux_cmd[mflux_cmd.index("--image-path") + 1]
    assert image_arg.endswith("IMG_1234.jpg")
    assert ".heic" not in image_arg


def test_cmd_batch_heic_history_records_original_path(
    tmp_path, _batch_env, stub_sips
):
    """history.input must record the user's ORIGINAL HEIC path — not the
    transient cache jpeg, which would be unusable for `imgen replay <id>`.
    The output-file naming uses the original stem too (IMG_1234-anime.png,
    not the cache stem)."""
    from imgen.history import load_history
    d = _make_input_dir(tmp_path, "IMG_1234.heic")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    cmd_batch(args)
    entries = load_history()
    assert len(entries) == 1
    assert entries[0]["input"] == str(d / "IMG_1234.heic")
    # Output stays IMG_1234-anime.png (original stem, not the cache one
    # which would also be IMG_1234 but that's coincidence here — what we
    # care about is the stem is the original's).
    assert Path(entries[0]["output"]).name == "IMG_1234-anime.png"


def test_cmd_batch_heic_cache_cleaned_after_run(
    tmp_path, _batch_env, stub_sips
):
    """tempfile.TemporaryDirectory wipes the converted JPEGs on
    cmd_batch exit (success or failure). Verify no `imgen-heic-*`
    directory survives under the default TMPDIR."""
    import tempfile
    tmpdir_before = set(Path(tempfile.gettempdir()).glob("imgen-heic-*"))
    d = _make_input_dir(tmp_path, "IMG_1.heic", "IMG_2.heic")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    cmd_batch(args)
    tmpdir_after = set(Path(tempfile.gettempdir()).glob("imgen-heic-*"))
    # No imgen-heic-* directory persists past cmd_batch.
    assert tmpdir_after == tmpdir_before


def test_cmd_batch_heic_and_jpg_mixed(tmp_path, _batch_env, stub_sips):
    """Mixed dir — HEIC gets converted, JPG passes through directly."""
    d = _make_input_dir(tmp_path, "a.jpg", "b.heic")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
    )
    rc = cmd_batch(args)
    assert rc == 0
    # sips ran exactly once (for b.heic) — a.jpg passed through.
    assert len(stub_sips) == 1
    # Both inputs reach mflux.
    assert len(_batch_env["calls"]) == 2


# ── Confirm gate ────────────────────────────────────────────────────────


def test_cmd_batch_confirm_cancel_returns_zero_no_mflux(
    tmp_path, _batch_env, monkeypatch, capsys
):
    """User answers `n` at the confirm gate → no mflux runs, no run_dir
    created, no history entries, exit 0 (graceful cancel, not error)."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    d = _make_input_dir(tmp_path, "a.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
        yes=False,  # force confirm gate
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert _batch_env["calls"] == []
    # No run dir created because user cancelled.
    assert not any((tmp_path / "out").glob("*"))


def test_cmd_batch_confirm_yes_proceeds(
    tmp_path, _batch_env, monkeypatch
):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    d = _make_input_dir(tmp_path, "a.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out", style=["anime"],
        yes=False,
    )
    rc = cmd_batch(args)
    assert rc == 0
    assert len(_batch_env["calls"]) == 1


def test_cmd_batch_confirm_shows_n_times_m_counts(
    tmp_path, _batch_env, monkeypatch, capsys
):
    """Confirm gate surfaces both N (inputs) and M (styles) so the user
    can sanity-check the total without doing the math themselves."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    d = _make_input_dir(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    args = _args(
        directory=d, output_dir=tmp_path / "out",
        style=["anime", "ghibli"], yes=False,
    )
    cmd_batch(args)
    out = capsys.readouterr().out
    assert "6 images" in out
    assert "3 inputs" in out
    assert "2 styles" in out

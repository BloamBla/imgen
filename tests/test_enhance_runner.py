"""Subprocess wrapper tests for the LLM prompt enhancer (v0.5 Phase C).

The actual mlx_lm load + generate happens inside ``python -m
imgen.enhance_runner`` (a separate subprocess). This file tests:

* The JSON wire protocol between imgen and the runner (request +
  response shape; one-shot batch in, one-shot batch out).
* :func:`imgen.enhance.run_with_mlx_lm` — the spawn-and-read wrapper.
* Failure paths: runner crashes / times out / returns malformed JSON
  / item count mismatch — every one degrades to a per-item ``None``
  so ``decide_final_prompt`` falls back to original cleanly.

The mlx_lm.load itself is NOT exercised here (would download a 4 GB
Qwen model and burn ~10 s every test run). Manual smoke covers the
real-mlx-lm path. These tests use a tiny fake runner script that
acts as drop-in stdin→stdout JSON echo.
"""
from __future__ import annotations

import json
import sys

import pytest

from imgen.enhance import (
    RunnerError,
    build_runner_payload,
    parse_runner_response,
    run_with_mlx_lm,
)


# ── Wire protocol ───────────────────────────────────────────────────────


class TestBuildRunnerPayload:
    def test_minimal_shape(self):
        payload = build_runner_payload(
            items=[{"system": "sys A", "user": "usr A"}],
            model="mlx-community/Qwen2.5-7B-Instruct-4bit",
            temperature=0.0,
            max_tokens=200,
        )
        assert payload == {
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "temperature": 0.0,
            "max_tokens": 200,
            "items": [{"system": "sys A", "user": "usr A"}],
        }

    def test_multi_item(self):
        payload = build_runner_payload(
            items=[
                {"system": "S", "user": "first"},
                {"system": "S", "user": "second"},
                {"system": "S", "user": "third"},
            ],
            model="m",
            temperature=0.5,
            max_tokens=100,
        )
        assert len(payload["items"]) == 3
        assert payload["items"][2]["user"] == "third"

    def test_returns_json_serializable(self):
        # Any callable that builds the runner payload must produce a
        # dict that json.dumps accepts — wire protocol is JSON, not
        # pickle. Lock-in test so a future addition of a non-JSON type
        # (a Path, an Enum) is caught.
        payload = build_runner_payload(
            items=[{"system": "s", "user": "u"}],
            model="m", temperature=0.0, max_tokens=10,
        )
        s = json.dumps(payload)
        assert isinstance(s, str)
        assert json.loads(s) == payload


class TestParseRunnerResponse:
    def test_standard_response(self):
        raw = json.dumps({
            "results": [
                {"output": "enhanced one"},
                {"output": "enhanced two"},
            ]
        })
        outputs = parse_runner_response(raw, expected_count=2)
        assert outputs == ["enhanced one", "enhanced two"]

    def test_single_item(self):
        raw = json.dumps({"results": [{"output": "x"}]})
        assert parse_runner_response(raw, expected_count=1) == ["x"]

    def test_count_mismatch_raises(self):
        raw = json.dumps({"results": [{"output": "a"}, {"output": "b"}]})
        with pytest.raises(RunnerError, match="expected 3 items, got 2"):
            parse_runner_response(raw, expected_count=3)

    def test_malformed_json_raises(self):
        with pytest.raises(RunnerError, match="invalid JSON"):
            parse_runner_response("not json {{{", expected_count=1)

    def test_missing_results_key_raises(self):
        with pytest.raises(RunnerError, match="missing 'results'"):
            parse_runner_response(json.dumps({"other": []}), expected_count=1)

    def test_runner_error_field_propagated(self):
        raw = json.dumps({"error": "model not found: foobar"})
        with pytest.raises(RunnerError, match="model not found"):
            parse_runner_response(raw, expected_count=1)

    def test_item_missing_output_raises(self):
        raw = json.dumps({"results": [{"oops": "no output"}]})
        with pytest.raises(RunnerError, match="missing 'output'"):
            parse_runner_response(raw, expected_count=1)


# ── End-to-end via fake runner script ───────────────────────────────────


@pytest.fixture
def fake_runner_script(tmp_path):
    """Write a tiny Python script that mimics enhance_runner: reads JSON
    from stdin, writes JSON to stdout following our wire protocol. The
    fake echoes each input back as ``"ENH: <user>"`` so tests can verify
    the data made it across both directions.
    """
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "results = [{'output': 'ENH: ' + it['user']} for it in payload['items']]\n"
        "json.dump({'results': results}, sys.stdout)\n"
    )
    return script


@pytest.fixture
def fake_runner_failing(tmp_path):
    """Fake runner that prints an error JSON and exits non-zero — mimics
    the real runner when mlx_lm.load fails (gated model, network down,
    missing dependency)."""
    script = tmp_path / "fake_runner_fail.py"
    script.write_text(
        "import json, sys\n"
        "json.dump({'error': 'simulated load failure'}, sys.stdout)\n"
        "sys.exit(1)\n"
    )
    return script


@pytest.fixture
def fake_runner_crashing(tmp_path):
    """Fake runner that crashes WITHOUT writing JSON — mimics OOM-kill
    or SIGSEGV from mlx_lm during inference. The wrapper must surface
    a RunnerError rather than emitting empty results that would be
    silently parsed as zero outputs."""
    script = tmp_path / "fake_runner_crash.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('mock segfault from mlx_lm\\n')\n"
        "sys.exit(139)\n"  # 128 + SIGSEGV
    )
    return script


@pytest.fixture
def fake_runner_slow(tmp_path):
    """Fake runner that sleeps past any reasonable timeout. Used to
    verify ``timeout`` kwarg actually triggers."""
    script = tmp_path / "fake_runner_slow.py"
    script.write_text(
        "import time\n"
        "time.sleep(30)\n"
    )
    return script


class TestRunWithMlxLm:
    def test_happy_path_single_item(self, fake_runner_script):
        outputs = run_with_mlx_lm(
            items=[{"system": "sys", "user": "hello"}],
            model="ignored-by-fake",
            temperature=0.0,
            max_tokens=10,
            python_executable=sys.executable,
            runner_script=fake_runner_script,
            timeout=10,
        )
        assert outputs == ["ENH: hello"]

    def test_happy_path_batch(self, fake_runner_script):
        outputs = run_with_mlx_lm(
            items=[
                {"system": "S", "user": "first"},
                {"system": "S", "user": "second"},
                {"system": "S", "user": "third"},
            ],
            model="m",
            temperature=0.0,
            max_tokens=10,
            python_executable=sys.executable,
            runner_script=fake_runner_script,
            timeout=10,
        )
        assert outputs == ["ENH: first", "ENH: second", "ENH: third"]

    def test_empty_items_short_circuits(self, fake_runner_script):
        # No items in batch → no subprocess, no overhead. Spawning the
        # LLM for zero work would be wasteful.
        outputs = run_with_mlx_lm(
            items=[],
            model="m",
            temperature=0.0,
            max_tokens=10,
            python_executable=sys.executable,
            runner_script=fake_runner_script,
            timeout=10,
        )
        assert outputs == []

    def test_runner_error_response_raises(self, fake_runner_failing):
        with pytest.raises(RunnerError, match="simulated load failure"):
            run_with_mlx_lm(
                items=[{"system": "s", "user": "u"}],
                model="m",
                temperature=0.0,
                max_tokens=10,
                python_executable=sys.executable,
                runner_script=fake_runner_failing,
                timeout=10,
            )

    def test_runner_crash_raises(self, fake_runner_crashing):
        # Non-zero exit + no JSON output = unrecoverable from imgen
        # side. Surface as RunnerError so callers can catch + fall
        # back to text-only generation for the whole batch.
        with pytest.raises(RunnerError):
            run_with_mlx_lm(
                items=[{"system": "s", "user": "u"}],
                model="m",
                temperature=0.0,
                max_tokens=10,
                python_executable=sys.executable,
                runner_script=fake_runner_crashing,
                timeout=10,
            )

    def test_timeout_kills_runner(self, fake_runner_slow):
        # Wrapper must terminate the subprocess and raise RunnerError
        # rather than block indefinitely. 1-second timeout vs 30-second
        # sleep in fake script.
        with pytest.raises(RunnerError, match="timeout|timed out"):
            run_with_mlx_lm(
                items=[{"system": "s", "user": "u"}],
                model="m",
                temperature=0.0,
                max_tokens=10,
                python_executable=sys.executable,
                runner_script=fake_runner_slow,
                timeout=1,
            )

    def test_count_mismatch_raises(self, tmp_path):
        # Fake runner that returns 1 result for 3 items.
        script = tmp_path / "fake_short.py"
        script.write_text(
            "import json, sys\n"
            "json.load(sys.stdin)\n"
            "json.dump({'results': [{'output': 'one'}]}, sys.stdout)\n"
        )
        with pytest.raises(RunnerError, match="expected 3 items, got 1"):
            run_with_mlx_lm(
                items=[{"system": "s", "user": f"u{i}"} for i in range(3)],
                model="m",
                temperature=0.0,
                max_tokens=10,
                python_executable=sys.executable,
                runner_script=script,
                timeout=10,
            )

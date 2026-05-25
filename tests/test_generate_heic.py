"""HEIC pre-conversion in cmd_generate (v0.3.0 bonus).

Pre-v0.3.0, ``imgen generate vacation.heic`` died with a cryptic mflux
``PIL.UnidentifiedImageError``. v0.3.0 extracts ``inputs.resolve_to_
mflux_input`` and plugs it into ``cmd_generate`` so any HEIC input is
sips-converted to JPEG before mflux ever sees it. Same helper that
powers ``cmd_batch``'s per-input HEIC handling.

These tests cover the cmd_generate side — cmd_batch HEIC behaviour is
locked in ``test_batch.py``. Stubbing surface (v0.3.1 post-cmd_helpers
extraction):
``imgen.cmd_helpers.run_with_stderr_redaction`` (fake mflux),
``imgen.commands.generate.load_backend_and_token`` (bypass venv +
binary; patched at the generate.py call site since it imports by
name), ``imgen.commands.generate.detect_resolution`` (no PIL), and
``imgen.inputs.subprocess.run`` (fake sips).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.backends import BACKENDS
from imgen.commands.generate import cmd_generate
from imgen.defaults import DEFAULTS


# ── Fixtures ────────────────────────────────────────────────────────────


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
        # v0.4: 5th element is the custom-backend secret tuple (None for FLUX).
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
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("imgen.inputs.subprocess.run", fake_run)
    return calls


def _gen_args(*, image: Path, **overrides) -> SimpleNamespace:
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
        model="flux",
        scope=None,
        width=None, height=None,
        output=None,
        output_dir=None,
        force=True,
        yes=True,
        no_open=True,
        dry_run=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── HEIC pre-conversion ─────────────────────────────────────────────────


def test_cmd_generate_heic_runs_sips_then_mflux(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips,
):
    """cmd_generate vacation.heic → sips converts → mflux gets the JPEG.
    Confirms the v0.2.x cryptic-error bug is fixed."""
    heic = tmp_path / "vacation.heic"
    heic.write_bytes(b"heic-bytes")
    args = _gen_args(image=heic, output_dir=str(tmp_path / "out"))

    rc = cmd_generate(args)

    assert rc == 0
    assert len(stub_sips) == 1
    # sips called with the original heic
    assert stub_sips[0][:5] == [
        "sips", "-s", "format", "jpeg", str(heic)
    ]
    # mflux was passed the converted JPEG path, NOT the .heic
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    image_arg = mflux_cmd[mflux_cmd.index("--image-path") + 1]
    assert image_arg.endswith("vacation.jpg")
    assert ".heic" not in image_arg


def test_cmd_generate_jpg_does_not_invoke_sips(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips,
):
    """Non-HEIC input must not pay the sips cost. resolve_to_mflux_input
    returns the path unchanged for jpg/png/etc., so the cache TempDir is
    created and immediately cleaned with no sips invocation in between."""
    jpg = tmp_path / "photo.jpg"
    jpg.write_bytes(b"jpeg-bytes")
    args = _gen_args(image=jpg, output_dir=str(tmp_path / "out"))

    rc = cmd_generate(args)

    assert rc == 0
    assert stub_sips == []
    # mflux was passed the ORIGINAL jpeg path.
    mflux_cmd = stub_mflux["calls"][0]["cmd"]
    image_arg = mflux_cmd[mflux_cmd.index("--image-path") + 1]
    assert image_arg == str(jpg)


def test_cmd_generate_heic_history_records_original_path(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips,
):
    """history.input must record the user's ORIGINAL .heic path so
    `imgen replay <id>` can re-convert + re-run. Storing the transient
    cache path would 404 on replay (TemporaryDirectory wiped it)."""
    from imgen.history import load_history
    heic = tmp_path / "vacation.heic"
    heic.write_bytes(b"")
    args = _gen_args(image=heic, output_dir=str(tmp_path / "out"))

    cmd_generate(args)

    e = load_history()[0]
    assert e["input"] == str(heic)
    assert e["input"].endswith(".heic")


def test_cmd_generate_heic_cache_cleaned_after_run(
    tmp_state_dir, tmp_path, stub_mflux, stub_backend, stub_dims,
    stub_open, stub_sips,
):
    """tempfile.TemporaryDirectory wipes the cache on cmd_generate
    exit — no /tmp/imgen-heic-* leaks per invocation."""
    import tempfile
    heic = tmp_path / "x.heic"
    heic.write_bytes(b"")
    args = _gen_args(image=heic, output_dir=str(tmp_path / "out"))

    cache_dirs_before = set(Path(tempfile.gettempdir()).glob("imgen-heic-*"))
    cmd_generate(args)
    cache_dirs_after = set(Path(tempfile.gettempdir()).glob("imgen-heic-*"))
    assert cache_dirs_after == cache_dirs_before

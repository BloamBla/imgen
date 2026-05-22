"""`imgen doctor` Backends section — pure check_backend_health helper.

Mirrors test_doctor_alias.py shape. The doctor's UI loop in
cmd_doctor wraps these BackendHealth results into ok/warn lines and
bumps the issues counter; the pure function is what we lock here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.commands.doctor import BackendHealth, check_backend_health


# ── Built-in backends (always present in merged registry) ───────────────


def test_check_backend_health_includes_builtins(tmp_path):
    """flux + qwen always show up — they're in BUILTIN_BACKENDS."""
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "mflux-generate-kontext").write_text("#!/bin/sh\n")
    (venv / "mflux-generate-qwen-edit").write_text("#!/bin/sh\n")

    results = check_backend_health(venv_bin=venv, env={})
    names = {h.name for h in results}
    assert "flux" in names
    assert "qwen" in names


def test_check_backend_health_marks_builtins_as_builtin(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "mflux-generate-kontext").touch()
    (venv / "mflux-generate-qwen-edit").touch()

    results = check_backend_health(venv_bin=venv, env={})
    for h in results:
        if h.name in {"flux", "qwen"}:
            assert h.origin == "built-in"


def test_check_backend_health_binary_ok_when_file_exists(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "mflux-generate-kontext").touch()
    (venv / "mflux-generate-qwen-edit").touch()

    results = check_backend_health(venv_bin=venv, env={})
    by_name = {h.name: h for h in results}
    assert by_name["flux"].binary_ok is True
    assert by_name["qwen"].binary_ok is True


def test_check_backend_health_binary_missing_when_venv_empty(tmp_path):
    """The missing-binary signal — venv exists but the mflux entries
    aren't installed yet. Doctor surfaces this for the user."""
    venv = tmp_path / "venv-empty"
    venv.mkdir()

    results = check_backend_health(venv_bin=venv, env={})
    for h in results:
        if h.origin == "built-in":
            assert h.binary_ok is False


def test_check_backend_health_binary_ok_false_when_path_is_dir(tmp_path):
    """is_file() (not exists()) — a directory at the resolved path
    should not be reported as ok. Mirrors python-reviewer IMP-1."""
    venv = tmp_path / "venv"
    venv.mkdir()
    # Create a *directory* at the binary path, not a file.
    (venv / "mflux-generate-kontext").mkdir()
    (venv / "mflux-generate-qwen-edit").touch()

    results = check_backend_health(venv_bin=venv, env={})
    by_name = {h.name: h for h in results}
    assert by_name["flux"].binary_ok is False
    assert by_name["qwen"].binary_ok is True


# ── Built-ins have no secret slot ───────────────────────────────────────


def test_check_backend_health_builtins_have_no_secret(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "mflux-generate-kontext").touch()
    (venv / "mflux-generate-qwen-edit").touch()

    results = check_backend_health(venv_bin=venv, env={})
    by_name = {h.name: h for h in results}
    assert by_name["flux"].secret_env_var is None
    assert by_name["flux"].secret_present is None
    assert by_name["qwen"].secret_env_var is None


# ── Custom backends + secret resolution ─────────────────────────────────


@pytest.fixture
def custom_backend(tmp_path, monkeypatch):
    """Drop a single user backend TOML with a [secret] section, then
    reset the cache so check_backend_health re-reads. Returns the
    binary path that test bodies can manipulate (touch to make ok,
    skip to make missing)."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    (backends_dir / "myback.toml").write_text(
        'binary = "myback-bin"\n'
        'image_flag = "--image-path"\n'
        '\n[secret]\n'
        'env_var = "MYBACK_API_KEY"\n'
        'required = true\n'
    )
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", backends_dir)
    backends_mod.reset_backends_cache()
    yield tmp_path
    backends_mod.reset_backends_cache()


def test_check_backend_health_custom_backend_binary_missing(
    custom_backend, tmp_path
):
    """v0.4 python-reviewer IMP-3 + design-memo planned test:
    test_doctor_warns_on_custom_backend_binary_missing. A user backend
    whose binary isn't in VENV_BIN gets binary_ok=False; the doctor
    UI turns this into a warn + issues++."""
    venv = tmp_path / "venv-empty"
    venv.mkdir()  # no myback-bin inside

    results = check_backend_health(venv_bin=venv, env={"MYBACK_API_KEY": "x"})
    by_name = {h.name: h for h in results}
    assert "myback" in by_name
    assert by_name["myback"].origin == "custom"
    assert by_name["myback"].binary_ok is False


def test_check_backend_health_custom_secret_present_when_env_set(
    custom_backend, tmp_path
):
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "myback-bin").touch()

    results = check_backend_health(
        venv_bin=venv, env={"MYBACK_API_KEY": "secret_value_123"}
    )
    by_name = {h.name: h for h in results}
    assert by_name["myback"].secret_present is True


def test_check_backend_health_custom_secret_missing_when_env_unset(
    custom_backend, tmp_path
):
    """The required-secret-not-set signal — doctor UI surfaces this
    as warn + issues++ so a user knows `imgen --backend myback` will
    die before they try."""
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "myback-bin").touch()

    results = check_backend_health(venv_bin=venv, env={})  # no MYBACK_API_KEY
    by_name = {h.name: h for h in results}
    assert by_name["myback"].secret_present is False
    assert by_name["myback"].secret_required is True
    assert by_name["myback"].secret_env_var == "MYBACK_API_KEY"


def test_check_backend_health_custom_secret_empty_string_counts_as_missing(
    custom_backend, tmp_path
):
    """Locks in the v0.4 python-reviewer IMP-2 contract: an env var
    explicitly set to empty string is treated as missing. An empty
    token is useless and forwarding it would produce a confusing
    auth failure from the backend's binary."""
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "myback-bin").touch()

    results = check_backend_health(venv_bin=venv, env={"MYBACK_API_KEY": ""})
    by_name = {h.name: h for h in results}
    assert by_name["myback"].secret_present is False


# ── Absolute binary path bypass VENV_BIN ────────────────────────────────


def test_check_backend_health_absolute_binary_uses_path_as_is(
    tmp_path, monkeypatch
):
    """Custom backend declaring binary = "/abs/path/bin" — the VENV_BIN
    arg should be ignored for that backend."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    abs_bin = tmp_path / "abs_dir" / "abs-bin"
    abs_bin.parent.mkdir()
    abs_bin.touch()
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    (backends_dir / "abso.toml").write_text(
        f'binary = "{abs_bin}"\n'
        'image_flag = "--image-path"\n'
    )
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", backends_dir)
    backends_mod.reset_backends_cache()
    try:
        # Pass an EMPTY venv — would have failed if absolute path
        # went through venv_bin / be.binary instead of being used as-is.
        venv = tmp_path / "empty-venv"
        venv.mkdir()
        results = check_backend_health(venv_bin=venv, env={})
        by_name = {h.name: h for h in results}
        assert by_name["abso"].binary_ok is True
        assert by_name["abso"].binary_path == abs_bin
    finally:
        backends_mod.reset_backends_cache()

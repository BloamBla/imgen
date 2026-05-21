"""Integration test: pipx-style install end-to-end.

Installs the package from this repo into a throwaway venv with
`IMGEN_HOME` unset, then runs `imgen --version` and `imgen --list-styles`.
Catches regressions in startup code paths where `IMGEN_HOME=None` — pure
unit tests can't exercise the entry-point's full import + dispatch path.

Slow (~10-30s for venv create + pip install + build isolation). Skipped
unless `IMGEN_RUN_SLOW=1` is set. Doesn't install `mflux` (~minutes,
hundreds of MB) so `imgen doctor` / `generate` are deliberately out of
scope — those are smoke-tested manually before each release.

Run with:
    IMGEN_RUN_SLOW=1 .venv/bin/pytest tests/test_pipx_install.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from imgen import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("IMGEN_RUN_SLOW") != "1",
    reason="slow integration test — set IMGEN_RUN_SLOW=1 to run",
)


@pytest.fixture(scope="module")
def pipx_like_venv(tmp_path_factory):
    """Fresh venv with imgen installed via pip (no IMGEN_HOME).

    `--no-deps` skips mflux. The point of this fixture is to exercise
    imgen's own startup code without IMGEN_HOME — anything that needs
    mflux is out of scope here.
    """
    base = tmp_path_factory.mktemp("pipx_like")
    venv_dir = base / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    py = venv_dir / "bin" / "python"

    subprocess.run(
        [str(py), "-m", "pip", "install", "--no-deps", "--quiet",
         str(REPO_ROOT)],
        check=True,
    )
    imgen_bin = venv_dir / "bin" / "imgen"
    assert imgen_bin.exists(), "imgen entry point not installed"
    return imgen_bin


def _env_without_imgen_home() -> dict[str, str]:
    """Copy parent env but scrub IMGEN_HOME so the child runs in pipx mode.

    Also drops HF_TOKEN — the dev shell may have a real token set; we
    never want it inheriting into a captured subprocess whose output may
    be logged by CI.
    """
    env = os.environ.copy()
    env.pop("IMGEN_HOME", None)
    env.pop("HF_TOKEN", None)
    return env


def _run_imgen(bin_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run an imgen subcommand and assert clean exit, surfacing output on fail.

    `subprocess.run(..., check=True, capture_output=True)` raises
    CalledProcessError on failure but pytest's default output doesn't
    show the captured stdout/stderr — CI logs become useless. Catch and
    re-raise via pytest.fail with the full output instead.
    """
    proc = subprocess.run(
        [str(bin_path), *args],
        env=_env_without_imgen_home(),
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"imgen {' '.join(args)} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def test_pipx_imgen_version_matches_module_version(pipx_like_venv):
    """`imgen --version` (pipx install, no IMGEN_HOME) prints __version__.

    Regression guard for any module-level reference to IMGEN_HOME that
    forgets to None-check — would AttributeError on import here.
    """
    result = _run_imgen(pipx_like_venv, "--version")
    assert __version__ in (result.stdout + result.stderr)


def test_pipx_imgen_list_styles_no_crash(pipx_like_venv):
    """--list-styles touches BUILTIN_STYLES + colors + styles loader.

    These are the heaviest pure-Python startup paths short of generate /
    doctor; if any of them blows up in pipx mode this catches it.
    """
    result = _run_imgen(pipx_like_venv, "--list-styles")
    assert "Available styles" in result.stdout
    assert "anime" in result.stdout  # one of the built-in presets


def test_pipx_imgen_help_no_crash(pipx_like_venv):
    """`imgen --help` should work without IMGEN_HOME — the upgrade-command
    epilog references IMGEN_HOME in its text, but only via .py-level f-strings
    that must handle None gracefully."""
    result = _run_imgen(pipx_like_venv, "--help")
    assert "imgen" in result.stdout.lower()
    assert "upgrade" in result.stdout

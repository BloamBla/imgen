"""System health checks: Python venv, mflux, Pillow, disk, RAM, battery,
parallel-mflux detection, resource preflight aggregator.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .backends import BACKENDS
from .defaults import MIN_BATTERY_PCT, MIN_DISK_GB, RAM_REQUIRED_GB
from .paths import VENV_BIN

__all__ = [
    "check_disk_gb",
    "check_mflux",
    "check_pillow",
    "check_resources",
    "check_venv",
    "find_running_mflux",
    "get_battery",
    "get_memory_gb",
]


def check_venv() -> bool:
    """True if a Python interpreter is present in the venv hosting imgen.

    For pipx-installed users this is trivially true (we're running from
    inside that venv). For bootstrap users the launcher shim already
    pointed us at .venv/bin/imgen, so reaching this code means the venv
    exists — but we still stat() so cmd_doctor reports cleanly when run
    via `python -m imgen` from a broken venv.
    """
    py = VENV_BIN / "python"
    return py.exists()


def check_mflux() -> str | None:
    """Return installed mflux version string, or None if missing/broken."""
    if not check_venv():
        return None
    pip = VENV_BIN / "pip"
    try:
        out = subprocess.check_output(
            [str(pip), "show", "mflux"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        for line in out.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None
    return None


def check_pillow() -> str | None:
    if not check_venv():
        return None
    py = VENV_BIN / "python"
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import PIL; print(PIL.__version__)"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def check_disk_gb() -> float:
    stat = shutil.disk_usage(Path.home())
    return stat.free / (1024 ** 3)


def get_memory_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb) for the macOS host.

    Uses vm_stat + sysctl — no extra Python deps. Returns (0, 0) on
    non-Darwin or parse failure (caller should treat as 'unknown').
    """
    try:
        total = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=5
        ).decode().strip())
        pagesize = int(subprocess.check_output(
            ["sysctl", "-n", "hw.pagesize"], timeout=5
        ).decode().strip())
        vm_out = subprocess.check_output(["vm_stat"], timeout=5).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError,
            subprocess.TimeoutExpired):
        return 0.0, 0.0

    pages = {}
    for line in vm_out.splitlines():
        m = re.match(r'"?([A-Za-z][^"]*?)"?:\s+(\d+)', line)
        if m:
            pages[m.group(1).strip()] = int(m.group(2))
    free = pages.get("Pages free", 0)
    inactive = pages.get("Pages inactive", 0)
    purgeable = pages.get("Pages purgeable", 0)
    speculative = pages.get("Pages speculative", 0)
    available_bytes = (free + inactive + purgeable + speculative) * pagesize
    return total / (1024 ** 3), available_bytes / (1024 ** 3)


def get_battery() -> tuple[int | None, bool]:
    """Return (percent, on_ac). For desktop Macs returns (None, True)."""
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "batt"], stderr=subprocess.DEVNULL, timeout=5
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return None, True
    pct_match = re.search(r"(\d+)%", out)
    pct = int(pct_match.group(1)) if pct_match else None
    on_ac = (("AC Power" in out)
             or ("charged" in out.lower())
             or ("charging" in out.lower()))
    return pct, on_ac


def find_running_mflux() -> int | None:
    """Return PID of another running mflux-generate-* process, or None.

    Uses `pgrep -x` (exact match on process basename) for each known backend
    binary. Avoids the false positives of the old `pgrep -f mflux-generate`:
    cmdlines like `vim mflux-generate.txt` no longer block runs, and a custom
    prompt containing the literal "mflux-generate" can't trip the check on
    its own re-run.
    """
    for be in BACKENDS.values():
        try:
            out = subprocess.check_output(
                ["pgrep", "-x", be.binary], stderr=subprocess.DEVNULL,
                timeout=5
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            continue
        for line in out.splitlines():
            try:
                return int(line.strip())
            except ValueError:
                pass
    return None


def check_resources(backend: str, quantize: int) -> dict:
    """Snapshot system resources vs requirements for this run."""
    required = RAM_REQUIRED_GB.get((backend, quantize), 16)
    total, available = get_memory_gb()
    disk_free = check_disk_gb()
    battery_pct, on_ac = get_battery()
    return {
        "ram_required_gb": required,
        "ram_total_gb": total,
        "ram_available_gb": available,
        # If we can't read memory (total=0), don't block — assume OK
        "ram_ok": total == 0 or available >= required,
        "disk_free_gb": disk_free,
        "disk_ok": disk_free >= MIN_DISK_GB,
        "battery_pct": battery_pct,
        "on_ac": on_ac,
        "battery_ok": (on_ac
                       or battery_pct is None
                       or battery_pct >= MIN_BATTERY_PCT),
        "other_mflux_pid": find_running_mflux(),
    }

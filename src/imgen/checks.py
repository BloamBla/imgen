"""System health checks: Python venv, mflux, Pillow, disk, RAM, battery,
parallel-mflux detection, resource preflight aggregator.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .backends import BACKENDS
from .defaults import MIN_BATTERY_PCT, MIN_DISK_GB
from .paths import VENV_BIN

__all__ = [
    "check_disk_gb",
    "check_mflux",
    "check_pillow",
    "check_resources",
    "ram_required_gb",
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


def ram_required_gb(
    backend: str, quantize: int, megapixels: float,
    num_frames: int = 1,
) -> float:
    """Per-Model peak RAM estimate (GB) for the given
    ``(model_name, quantize, megapixels)`` invocation.

    v0.8.0 commit 8 (§L) closure: replaces v0.7.14's per-
    ``(backend, quant)`` fixed-table lookup with per-Model math
    sourced from ``Engine.ram_estimate_gb``. The pre-commit-8
    ``defaults.RAM_REQUIRED_GB`` + ``ACTIVATION_GB_PER_MP_ABOVE_BASELINE``
    constants have been deleted; this function is the single
    source-of-truth on the preflight side.

    Lookup path:

    1. Translate v0.7 names (``flux``, ``qwen``) to v0.8 canonical
       names via ``_V07_TO_V08_MODEL_RENAMES`` — back-compat for
       history-replay paths where ``entry["backend"]`` may carry
       the legacy spelling.
    2. ``BUILTIN_MODELS`` lookup → call ``engine.ram_estimate_gb``
       on a minimal ``GenParams`` (only width × height and
       quantize matter for the formula at commit 8).
    3. Unknown model name (user-TOML registered Backend, or a typo)
       → conservative flux-class fallback (baseline 13.5, slope 4.0,
       no encoder, mflux overhead). Same physical math, no magic
       16 GB constant — the pre-commit-8 ``.get(..., 16)`` fallback
       was a v0.4-era guard against missing table rows; commit 8's
       model-driven approach treats absent rows as "unknown backend
       so assume flux-equivalent peak".

    Pure: no I/O, no subprocess.

    Signature retained as ``(backend, quantize, megapixels)`` for
    back-compat with the cmd_helpers preflight call site that
    threads ``args.model`` through ``check_resources``; the param
    name kept as ``backend`` to avoid the rename churn in
    test_checks.py (semantically it's the model name).
    """
    from pathlib import Path
    from .engines import DiffusersMpsEngine, MfluxEngine
    from .engines.base import GenParams
    from .models import _V07_TO_V08_MODEL_RENAMES, BUILTIN_MODELS

    # Normalise v0.7 → v0.8 names. Unchanged names pass through.
    v08_name = _V07_TO_V08_MODEL_RENAMES.get(backend, backend)
    model = BUILTIN_MODELS.get(v08_name)

    # Build a minimal GenParams that ``ram_estimate_gb`` can consume.
    # Width / height map to ``megapixels`` (sqrt approximation); other
    # fields are placeholders since the current formula only reads
    # quantize + dimensions.
    side = max(64, int((max(megapixels, 0.001) * 1_000_000) ** 0.5))
    # v0.9 commit 7.1 (§R.2 HIGH-1): num_frames threaded so
    # DiffusersMpsEngine._ram_estimate_video's ``0.1 * num_frames``
    # frame-term reflects the actual planned video length, not the
    # default placeholder. Image callers leave num_frames=1 and the
    # video branch isn't reached anyway.
    params = GenParams(
        prompt="", negative="", width=side, height=side,
        steps=1, guidance=0.0, seed=0, quantize=quantize,
        strength=0.0, input_path=None,
        output_path=Path("/tmp/_ram_estimate_placeholder.png"),
        loras=(),
        num_frames=num_frames,
    )

    if model is not None:
        # v0.9.5 M-2: dispatch via ENGINES registry (single source of
        # truth for engine-name → class). Old if/elif kept the
        # ``"mflux" else default-to-DiffusersMps`` branch implicit —
        # any future 3rd engine would silently land in the else.
        from .engines import get_engine
        engine = get_engine(model.engine)
        return engine.ram_estimate_gb(model, params)

    # Unknown model → conservative flux-class fallback. Same formula
    # shape; baseline/slope/encoder picked to land at ~16-18 GB for
    # the canonical Q4 1MP case (matches the pre-commit-8 ``.get(...,
    # 16)`` floor for unknown rows but scales physically).
    from .models import Model
    fallback_model = Model(
        engine="mflux",
        binary="mflux-generate-fake",
        ram_baseline_gb=13.5,
        ram_slope_gb_per_mp=4.0,
        encoder_ram_gb=0.0,
    )
    return MfluxEngine().ram_estimate_gb(fallback_model, params)


def check_resources(
    backend: str, quantize: int, megapixels: float = 1.0,
    num_frames: int = 1,
) -> dict:
    """Snapshot system resources vs requirements for this run.

    v0.7.14: ``megapixels`` argument added for dimension-aware RAM
    estimation via :func:`ram_required_gb`. Default 1.0 preserves
    pre-v0.7.14 behaviour for callers (tests, future extensions) that
    don't yet pass dimensions; the four real callers (cmd_generate /
    cmd_batch / cmd_refine / cmd_draw) all compute and pass it.

    v0.9 commit 7.1 (§R.2 HIGH-1): ``num_frames`` threaded so video
    callers (cmd_video) propagate the planned frame count into the
    RAM estimate. Image callers leave the default 1; the video
    branch of ``ram_estimate_gb`` is only reached for video Models.
    """
    required = ram_required_gb(
        backend, quantize, megapixels, num_frames=num_frames,
    )
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

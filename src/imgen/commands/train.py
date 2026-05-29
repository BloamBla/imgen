"""v0.10.0 — `imgen train` subcommand handler + dataset validator.

Per [[project-v100-design]] §G + §D.3 + §R.1 ROUND-1 CLOSURES preamble.

Surface:

* :class:`UserDatasetError` — raised on user-correctable dataset
  issues; cli.py catches and converts to die().
* :class:`DatasetEntry` — one validated row (image_path + caption).
* :func:`validate_dataset_dir` — pure-function validator with symlink
  reject, size cap, PIL decompression-bomb gate, sidecar UTF-8 strict
  decode, control-byte filter, trigger fallback, trigger-presence
  warn (§M.3).
* :func:`cmd_train` — subcommand handler implementing the 12-step
  pipeline per §G: validate → resolve → collision check → build
  TrainingParams → dry-run branch → preflight → confirm gate →
  materialise scratch → MfluxEngine.train → promote → meta-json →
  cleanup → history.append.

Security posture per §R.1 + §N trust boundary:

* Symlink reject mirrors v0.4 ``styles.d/`` + v0.9 ``.venv-diffusers/``
  same-uid attacker mitigation.
* ``lstat()`` runs BEFORE ``is_dir()`` so a broken symlink reports as
  symlink rather than 'not found' (python H-1 closure).
* Per-image size cap 50 MB + PIL ``Image.open(path).size`` probe
  ``width × height > 50M px`` close the DoS / decompression-bomb
  vector before mflux-train sees the file (security C-1 closure).
* Sidecar reads via ``read_bytes()`` then ``decode("utf-8",
  errors="strict")`` — invalid UTF-8 raises (no silent mangling).
* Control-byte filter on filenames AND sidecar contents — terminal
  escape injection vector closed.
* cmd_train mkdir(mode=0o700) on ~/.imgen/loras/ + scratch dir
  (security C-2 — PII-bearing weights + dataset path leak).
* ``build_mflux_env(token=hf_token)`` passed to MfluxEngine.train —
  NEVER ``env=None`` (security H-4).
"""
from __future__ import annotations

import json
import platform
import random
import shlex
import shutil
import stat as _stat
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._safe import has_control_bytes
from ..colors import C, die, err, info, ok, step, warn
from ..defaults import HISTORY_SCHEMA_VERSION


__all__ = [
    "DatasetEntry",
    "UserDatasetError",
    "cmd_train",
    "validate_dataset_dir",
]


# ── Constants (mirror v0.8.0 §E.1 USER_BACKEND_MAX_BYTES pattern) ──

# Per §R.1 security C-1: 50 MB per-image cap. Realistic identity-LoRA
# photos are 2-8 MB; 50 MB is generous enough that no legitimate input
# trips it while being well below DoS territory.
_MAX_IMAGE_BYTES: int = 50 * 1024 * 1024

# Per §R.1 security C-1: PIL decompression-bomb gate. A solid-color
# PNG can compress to ~10 KB on disk but decode into a 30 GB
# framebuffer if width × height is unbounded. 50M px (e.g. 7000×7000)
# is bigger than any realistic photo (10MP = 5000×2000) but small
# enough to refuse the bomb.
_MAX_IMAGE_PIXELS: int = 50_000_000

# Mirror of USER_BACKEND_MAX_BYTES = 16_384 in backends.py; sidecars
# are smaller-scope user-content blobs, 4 KB is generous.
_MAX_SIDECAR_BYTES: int = 4096

# Supported allowlist mirrors v0.3.0 ``imgen batch`` discipline.
_SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp",
})
# HEIC/HEIF rejected with a sips conversion hint (mirror v0.3.0 batch).
_HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif"})

_DEFAULT_MIN_IMAGES: int = 3
_DEFAULT_MAX_IMAGES_WARN: int = 30   # informational; not blocking


class UserDatasetError(Exception):
    """Raised by :func:`validate_dataset_dir` on any user-correctable
    dataset issue. ``cmd_train`` (commit 8) catches and converts to
    ``die()`` for a clean CLI error.

    Mirrors :class:`backends.UserBackendError` discipline — the
    exception itself carries the human-readable message; no separate
    error-code surface.
    """


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One validated dataset row.

    Attributes:
        image_path: absolute path under ``dataset_dir`` (caller-supplied
            base + child filename; symlink-validated at lstat time).
        caption: EITHER the sidecar ``.txt`` content (UTF-8 strict-decoded,
            stripped, control-byte-rejected) OR the trigger word fallback
            when no sidecar exists / the sidecar is empty.

    Frozen + slots so the parent dataset entry list stays hashable for
    test fixtures.
    """
    image_path: Path
    caption: str


def validate_dataset_dir(
    dataset_dir: Path,
    trigger: str,
    *,
    max_image_bytes: int = _MAX_IMAGE_BYTES,
    max_image_pixels: int = _MAX_IMAGE_PIXELS,
    max_sidecar_bytes: int = _MAX_SIDECAR_BYTES,
    min_images: int = _DEFAULT_MIN_IMAGES,
    max_images_warn: int = _DEFAULT_MAX_IMAGES_WARN,
) -> list[DatasetEntry]:
    """Validate ``dataset_dir`` for ``imgen train``.

    Returns a sorted list of :class:`DatasetEntry` (one per accepted
    image). Raises :class:`UserDatasetError` on any user-correctable
    issue.

    Pure: no subprocess, no env, no network. PIL ``Image.open(path).size``
    probes header-only (no full decode), so the decompression-bomb
    gate fires cheaply.

    Validation pipeline:

    1. ``lstat()`` on ``dataset_dir`` — disambiguates broken-symlink
       from missing-dir per §R.1 python H-1 closure.
    2. Iterate children sorted by name (determinism). For each child:

       a. Dotfile skip (``.DS_Store``, ``.hidden`` etc.).
       b. Filename C0/DEL/C1 control-byte filter (security risk row).
       c. ``lstat()`` symlink reject — hardlink TOCTOU mitigation per
          §R.1 security H-7.
       d. Sidecar ``.txt`` files handled at image-resolution time, not
          iterated standalone.
       e. HEIC/HEIF accumulated for batched rejection with sips hint
          (mirror v0.3.0 batch).
       f. Unsupported extensions silently skipped.
       g. Per-image size cap ``max_image_bytes`` (50 MB default).
       h. PIL ``Image.open(child).size`` probe — reject if
          ``width × height > max_image_pixels`` (50M px default).
       i. Sidecar resolution (UTF-8 strict, 4 KB cap, control-byte
          reject) OR trigger fallback.

    3. HEIC reject (informative — point at ``sips``).
    4. Count gate: ``len(entries) < min_images`` → reject.
    5. Informational warns: too-many-images (training noise), sidecars
       missing the trigger word (§M.3).
    """
    # ── 1. lstat() on dataset_dir ──
    try:
        st = dataset_dir.lstat()
    except FileNotFoundError:
        raise UserDatasetError(
            f"dataset dir does not exist: {dataset_dir}"
        )
    if _stat.S_ISLNK(st.st_mode):
        raise UserDatasetError(
            f"dataset dir is a symlink — refusing to read. "
            f"Same-uid attacker may have planted it. "
            f"Use a real directory at: {dataset_dir}"
        )
    if not _stat.S_ISDIR(st.st_mode):
        raise UserDatasetError(
            f"dataset path is not a directory: {dataset_dir}"
        )

    # ── 2. Iterate children (sorted) ──
    entries: list[DatasetEntry] = []
    heic_rejected: list[str] = []
    for child in sorted(dataset_dir.iterdir()):
        # 2a. Dotfile skip.
        if child.name.startswith("."):
            continue
        # 2b. Filename control-byte filter.
        if has_control_bytes(child.name):
            raise UserDatasetError(
                f"image filename contains control bytes "
                f"(C0/DEL/C1): {child.name!r}"
            )
        # 2c. lstat() + symlink reject (hardlink TOCTOU mitigation).
        child_st = child.lstat()
        if _stat.S_ISLNK(child_st.st_mode):
            raise UserDatasetError(
                f"dataset child is a symlink: {child.name} — refusing. "
                f"Same-uid attacker may have planted it. Move actual "
                f"image files into the dataset dir instead."
            )
        if not _stat.S_ISREG(child_st.st_mode):
            # Subdirs / sockets / etc — silently ignored. Non-recursive
            # discipline per §D.1.
            continue
        # 2d. Sidecar files (.txt) consumed by _resolve_caption at image
        # iteration time, not iterated standalone.
        ext = child.suffix.lower()
        if ext == ".txt":
            continue
        # 2e. HEIC/HEIF — accumulate for batched rejection.
        if ext in _HEIC_EXTS:
            heic_rejected.append(child.name)
            continue
        # 2f. Unsupported (but-not-HEIC) extensions silently skipped.
        if ext not in _SUPPORTED_IMAGE_EXTS:
            continue
        # 2g. Per-image size cap.
        if child_st.st_size > max_image_bytes:
            raise UserDatasetError(
                f"image {child.name} too big "
                f"({child_st.st_size / 1_048_576:.1f} MB > "
                f"{max_image_bytes / 1_048_576:.0f} MB cap). "
                f"DoS / decompression-bomb guard per §R.1 security C-1."
            )
        # 2h. PIL decompression-bomb gate (header-only probe).
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError:  # pragma: no cover — PIL is a project dep
            raise UserDatasetError(
                "PIL not available — cannot validate image dimensions"
            )
        try:
            with Image.open(child) as img:
                w, h = img.size
        except (UnidentifiedImageError, OSError) as e:
            raise UserDatasetError(
                f"cannot read image {child.name}: {e}"
            )
        if w * h > max_image_pixels:
            raise UserDatasetError(
                f"image {child.name} too large "
                f"({w}×{h} = {w * h:,} pixels > "
                f"{max_image_pixels:,} cap). Decompression bomb "
                f"guard per §R.1 security C-1."
            )
        # 2i. Caption resolution.
        caption = _resolve_caption(
            child, trigger, max_sidecar_bytes=max_sidecar_bytes,
        )
        entries.append(DatasetEntry(image_path=child, caption=caption))

    # ── 3. HEIC reject (informative) ──
    if heic_rejected:
        raise UserDatasetError(
            f"HEIC/HEIF images not supported (mflux-train reads via "
            f"PIL which doesn't handle them natively). Convert with "
            f"`sips -s format png input.heic --out output.png` per "
            f"v0.3.0 batch HEIC discipline. "
            f"Rejected files: {', '.join(heic_rejected)}"
        )

    # ── 4. Count gate ──
    if len(entries) < min_images:
        raise UserDatasetError(
            f"dataset has {len(entries)} image(s); minimum "
            f"{min_images} required. Recommended: 10-20 images of "
            f"consistent subject with varied poses/lighting per "
            f"colleague's M5 Pro recipe."
        )

    # ── 5. Informational warns ──
    if len(entries) > max_images_warn:
        warn(
            f"dataset has {len(entries)} images — training noise risk; "
            f"recommended ≤ {max_images_warn} for identity LoRAs. "
            f"Large datasets work but may dilute the trigger's "
            f"specificity."
        )

    # 5a. Trigger-presence warn per §M.3 — for each entry with a
    # sidecar that doesn't contain the trigger (case-insensitive),
    # warn so the user notices captions that won't activate the LoRA
    # cleanly. Only fires when caption != trigger verbatim (the
    # fallback path is skipped — that's the expected shape).
    trigger_lower = trigger.lower()
    for entry in entries:
        if entry.caption == trigger:
            continue  # fallback path — no sidecar to warn about
        if trigger_lower not in entry.caption.lower():
            warn(
                f"sidecar for {entry.image_path.name} doesn't include "
                f"the trigger {trigger!r} — trained LoRA may not "
                f"activate when this caption-shape is replayed at "
                f"inference time."
            )

    return entries


def _resolve_caption(
    image_path: Path,
    trigger: str,
    *,
    max_sidecar_bytes: int,
) -> str:
    """Resolve the caption for ``image_path`` from its sidecar ``.txt``
    file OR fall back to ``trigger``.

    Sidecar shape: same-stem ``.txt`` next to the image (e.g.
    ``photo01.jpg`` ↔ ``photo01.txt``). Missing sidecar OR empty
    sidecar → trigger fallback. Raises UserDatasetError on any sidecar
    issue (oversized, non-UTF-8, control bytes, symlink).
    """
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return trigger
    sidecar_st = sidecar.lstat()
    # Symlink-on-sidecar — refuse for the same reason as image children.
    if _stat.S_ISLNK(sidecar_st.st_mode):
        raise UserDatasetError(
            f"sidecar is a symlink: {sidecar.name} — refusing"
        )
    if sidecar_st.st_size > max_sidecar_bytes:
        raise UserDatasetError(
            f"sidecar {sidecar.name} too big "
            f"({sidecar_st.st_size} bytes > {max_sidecar_bytes} cap)"
        )
    try:
        raw = sidecar.read_bytes()
    except OSError as e:
        raise UserDatasetError(
            f"cannot read sidecar {sidecar.name}: {e}"
        )
    try:
        caption = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise UserDatasetError(
            f"sidecar {sidecar.name} is not valid UTF-8 "
            f"(strict decode failed at byte {e.start}). "
            f"Caption files must be UTF-8 encoded."
        )
    caption = caption.strip()
    if has_control_bytes(caption):
        raise UserDatasetError(
            f"sidecar {sidecar.name} contains control bytes "
            f"(C0/DEL/C1) — reject per security risk row"
        )
    # Empty sidecar (after strip) → fall back to trigger.
    if not caption:
        return trigger
    return caption


# ── v0.10.0 commit 8 — preflight + wall-time + dry-run + history helpers ──
#
# Per [[project-v100-design]] §G + §K + §J + §R.1 ROUND-1 CLOSURES.


# Per §K.2: video-style 3 GB safety buffer (wider than image's 1 GB
# default — training peaks are noisier than inference).
_PREFLIGHT_SAFETY_BUFFER_GB: float = 3.0

# Per v0.8.2 architecture: absolute floor below which any subprocess
# spawn is unsafe regardless of estimate.
_ABSOLUTE_RAM_FLOOR_GB: float = 4.0

# Wall-time heuristic for the confirm-gate estimate. §M.1 smoke
# (2026-05-28, M2 Pro 32 GB, q4/rank16/res512/low_ram, cached base):
# 50 --steps ran end-to-end in 419 s → ~8.4 s/--step. The earlier 27.5
# came from the colleague's DENSE-preview M5 Pro run (previews ≈ doubled
# wall); imgen's default preview_frequency=100 is sparse, so the real
# rate is ~3× lower. 8.5 = measured + small buffer. Note: long runs with
# previews enabled (--preview-every below --steps) add a full image-gen
# per preview, so treat this as a baseline, not a ceiling.
_M2_PRO_SECONDS_PER_STEP: float = 8.5


def _train_preflight(model, params, *, force: bool) -> None:
    """Pre-spawn gate: RAM headroom + battery state + binary presence.

    Raises via :func:`die` on hard failures. Battery-on-AC is a
    warn-not-die (mflux-train's own ``--battery-percentage-stop-limit``
    catches the runtime end-of-battery case).

    ``--force`` bypasses the RAM headroom check but NOT the absolute
    4 GB floor — that's a v0.8.2 invariant: below 4 GB available a
    subprocess spawn risks OOM-killer terminating the whole imgen
    process mid-run.
    """
    from ..checks import get_battery, get_memory_gb

    total_gb, available_gb = get_memory_gb()
    estimate = model.training.training_peak_ram_gb
    needed = estimate + _PREFLIGHT_SAFETY_BUFFER_GB

    if not force and available_gb < needed:
        die(
            f"Insufficient RAM for training: need ~{estimate:.1f} GB + "
            f"{_PREFLIGHT_SAFETY_BUFFER_GB:.1f} GB safety = "
            f"{needed:.1f} GB, have {available_gb:.1f} GB available. "
            f"Close other apps, or pass --force to override "
            f"(unsafe — may OOM mid-run).",
            code=2,
        )

    # Absolute floor — even --force can't bypass.
    if available_gb < _ABSOLUTE_RAM_FLOOR_GB:
        die(
            f"available_gb={available_gb:.1f} below "
            f"{_ABSOLUTE_RAM_FLOOR_GB:.1f} GB safety floor — "
            "refusing to spawn mflux-train.",
            code=2,
        )

    # Battery check: warn-not-die. mflux-train handles the runtime
    # cutoff via --battery-percentage-stop-limit; we just surface the
    # AC-power expectation so the user can plug in before kicking off
    # a 10h overnight job.
    bat_pct, on_ac = get_battery()
    if not on_ac and bat_pct is not None:
        warn(
            f"Mac is on battery ({bat_pct}%); training pulls ~60-90 W "
            f"continuous. Plug in to AC or training will stop at "
            f"{params.battery_stop}% per "
            f"--battery-percentage-stop-limit."
        )


def _estimate_wall_hours(params) -> float:
    """Heuristic wall-time estimate for the confirm-gate UX.

    Pure (params dataclass + module constants). Grounded in the §M.1
    smoke measurement (~8.4 s/--step end-to-end on M2 Pro 32 GB with
    sparse previews); see _M2_PRO_SECONDS_PER_STEP. So a default
    ``80 × N_images`` run lands around ~2 h for ~10 photos / ~4 h for
    ~20. Dense previews (``--preview-every`` below ``--steps``) add a
    full image-gen per preview and push the real wall above this floor.
    """
    return params.total_steps * _M2_PRO_SECONDS_PER_STEP / 3600.0


def _print_train_dryrun(
    params,
    config: dict,
    num_entries: int,
) -> int:
    """Dry-run branch: render the mflux-train JSON config + the
    equivalent shell invocation, exit 0 without spawning.

    Mirrors v0.9 ``imgen video --dry-run`` shape — the JSON is
    pretty-printed so users can sanity-check the exact training
    parameters before committing to a 10h overnight run.
    """
    from ..paths import VENV_BIN

    print(f"{C.BOLD}=== imgen train --dry-run ==={C.END}")
    print(f"LoRA name:       {params.lora_name}")
    print(f"Trigger:         {params.trigger!r}")
    print(f"Base model:      {params.base_model}")
    print(f"Dataset:         {params.dataset_dir} "
          f"({num_entries} images)")
    print(f"Output:          {params.output_path}")
    print(f"Total steps:     {params.total_steps}")
    print(f"Estimated wall:  ~{_estimate_wall_hours(params):.1f}h on "
          f"{platform.machine()}")
    print()
    print(f"{C.DIM}--- mflux-train JSON config ---{C.END}")
    print(json.dumps(config, indent=2))
    print()
    print(f"{C.DIM}--- equivalent shell invocation ---{C.END}")
    mflux_train_bin = VENV_BIN / "mflux-train"
    config_path = params.scratch_dir / "config.json"
    argv = [str(mflux_train_bin), "--config", str(config_path)]
    if params.battery_stop != 5:
        argv += [
            "--battery-percentage-stop-limit",
            str(params.battery_stop),
        ]
    print(" ".join(shlex.quote(a) for a in argv))
    print()
    print(f"{C.DIM}(scratch dir would be materialised at "
          f"{params.scratch_dir}){C.END}")
    return 0


def _append_train_history(
    params,
    num_entries: int,
    wall_seconds: int,
    status: str,
) -> None:
    """Append a v=4 history entry for the ``command="train"`` row
    shape per §J.1.

    ``status`` is one of: ``"success"``, ``"fail"``, ``"cancelled"``
    — same vocabulary as draw/refine/video for ``cmd_history`` /
    ``--list`` rendering.
    """
    from ..engine_dispatch import safe_append_history

    entry = {
        "v": HISTORY_SCHEMA_VERSION,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "command": "train",
        "model": params.base_model,
        "lora_name": params.lora_name,
        "trigger": params.trigger,
        "dataset_path": str(params.dataset_dir),
        "dataset_image_count": num_entries,
        "total_steps": params.total_steps,
        "lora_rank": params.lora_rank,
        "quantize": params.quantize,
        "max_resolution": params.max_resolution,
        # preview_frequency + battery_stop are written so `imgen replay`
        # reconstructs the run at full fidelity — _replay_train_entry
        # reads both back; omitting them silently reverted a replayed
        # job to the config defaults.
        "preview_frequency": params.preview_frequency,
        "battery_stop": params.battery_stop,
        "seed": params.seed,
        "output": str(params.output_path),
        "status": status,
        "wall_seconds": wall_seconds,
    }
    safe_append_history(entry)


def _existing_lora_summary(meta_path: Path) -> str:
    """Best-effort one-line ``" (trigger 'x', trained Y)"`` suffix for the
    name-collision refusal message, read from an existing LoRA's
    ``.meta.json``.

    Display-only on an error path — NEVER raises. A missing / corrupt /
    oversized / control-byte-bearing meta degrades to ``""`` (the bare
    "already exists" message). Same 16 KB read-cap + control-byte filter
    posture as :func:`imgen.lora_meta.read_lora_meta`.
    """
    try:
        raw = meta_path.read_bytes()
        if len(raw) > 16 * 1024:
            return ""
        meta = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(meta, dict):
        return ""

    bits: list[str] = []
    trigger = meta.get("trigger")
    if isinstance(trigger, str) and trigger.strip() and not has_control_bytes(trigger):
        bits.append(f"trigger {trigger.strip()!r}")
    trained_at = meta.get("trained_at")
    if isinstance(trained_at, str) and trained_at.strip() and not has_control_bytes(trained_at):
        bits.append(f"trained {trained_at.strip()}")
    return f" ({', '.join(bits)})" if bits else ""


def cmd_train(args) -> int:
    """v0.10.0 — ``imgen train`` subcommand handler.

    12-step pipeline per [[project-v100-design]] §G + §R.1 closures.
    Train a LoRA adapter from a dataset of images using mflux-train.

    Flow (each numbered step maps 1:1 with §G):

    1. Validate dataset via :func:`validate_dataset_dir`.
    2. Resolve base Model + ``TrainingConfig`` from BUILTIN_MODELS.
       Reject if ``model.training is None`` (e.g. flux-dev row has no
       training side).
    3. Output collision check (``--overwrite`` required if existing).
    4. Build :class:`TrainingParams` from CLI args > TrainingConfig
       defaults (None CLI flag = use config default).
    5. Build mflux JSON config via :func:`build_config_json`.
    6. ``--dry-run`` branch: print config + shell invocation, exit 0.
    7. Preflight: RAM + battery.
    8. Confirm gate (skipped with ``-y``).
    9. mkdir 0o700 on ``~/.imgen/loras/`` + materialise scratch.
    10. ``MfluxEngine.train(model, params, env=build_mflux_env(token=))``.
    11. On rc=0: promote ``.safetensors`` + write ``.meta.json`` +
        cleanup scratch. On rc!=0: keep scratch for diagnosis.
    12. ``history.append`` (always, on success / fail / cancelled).

    Returns mflux-train's exit code on success, 1 on confirm-gate
    decline, 2 on validation / preflight / collision die, 130 on
    KeyboardInterrupt (matches shell SIGINT convention).
    """
    from ..engines._training import TrainingParams, build_config_json
    from ..engines.mflux_engine import MfluxEngine
    from ..models import get_model
    from ..paths import STATE_DIR, ensure_state_dir
    from ..subprocess_helpers import build_mflux_env
    from ..tokens import load_token
    from . import _train_scratch
    from ..cmd_helpers import prompt_yes_no

    # ── 1. Validate dataset ──
    dataset_dir = Path(args.dataset).expanduser().resolve()
    try:
        entries = validate_dataset_dir(dataset_dir, trigger=args.trigger)
    except UserDatasetError as e:
        die(str(e), code=2)
    num_entries = len(entries)

    # ── 2. Resolve base model + TrainingConfig ──
    try:
        model = get_model(args.base)
    except (KeyError, ValueError) as e:
        die(str(e), code=2)
    if model.training is None:
        from ..models import BUILTIN_MODELS
        trainable = sorted(
            n for n, m in BUILTIN_MODELS.items() if m.training_supported
        )
        die(
            f"--base {args.base!r} does not support training. "
            f"Trainable bases: {', '.join(trainable)}.",
            code=2,
        )
    tc = model.training

    # ── 3. Output path + collision check ──
    ensure_state_dir()
    loras_dir = STATE_DIR / "loras"
    # mkdir mode=0o700 explicit (security C-2). exist_ok so re-runs
    # don't trip on the existing dir; the 0o700 mode is set only on
    # CREATE — caller-side chmod would race against a reader.
    loras_dir.mkdir(mode=0o700, exist_ok=True)

    output_path = loras_dir / f"{args.name}.safetensors"
    meta_path = loras_dir / f"{args.name}.meta.json"
    if output_path.exists() and not args.overwrite:
        die(
            f"~/.imgen/loras/{args.name}.safetensors already exists"
            f"{_existing_lora_summary(meta_path)}. "
            "Pass --overwrite to replace, or pick a different --name.",
            code=2,
        )

    # ── 4. Resolve training params (CLI args > TrainingConfig defaults) ──
    total_steps = args.steps if args.steps is not None else (
        tc.default_epochs * num_entries
    )
    lora_rank = args.rank if args.rank is not None else tc.default_lora_rank
    quantize = args.quantize if args.quantize is not None else (
        tc.default_quantize
    )
    max_resolution = args.max_resolution if args.max_resolution is not None else (
        tc.default_max_resolution
    )
    preview_frequency = args.preview_every if args.preview_every is not None else (
        tc.default_preview_frequency
    )
    seed = (
        args.seed if args.seed is not None
        else random.randint(0, 2**32 - 1)
    )

    scratch_dir = loras_dir / f".{args.name}.training"
    # TrainingParams.__post_init__ range-checks every numeric field.
    # argparse already bounds the direct-CLI path, but `imgen replay`
    # reconstructs args from a hand-editable history entry — a tampered
    # rank/quantize/steps must die cleanly (exit 2) rather than escape
    # as an uncaught ValueError traceback.
    try:
        params = TrainingParams(
            dataset_dir=dataset_dir,
            scratch_dir=scratch_dir,
            lora_name=args.name,
            trigger=args.trigger,
            base_model=args.base,
            total_steps=total_steps,
            lora_rank=lora_rank,
            max_resolution=max_resolution,
            quantize=quantize,
            low_ram=tc.default_low_ram,
            optimizer_name=tc.optimizer_name,
            optimizer_lr=tc.optimizer_lr,
            target_modules=tc.target_modules,
            preview_frequency=preview_frequency,
            seed=seed,
            battery_stop=args.battery_stop,
            output_path=output_path,
        )
    except (ValueError, TypeError) as e:
        die(f"invalid training parameter: {e}", code=2)

    # ── 5. Build mflux JSON config (used by both dry-run + real spawn) ──
    config = build_config_json(params, num_entries=num_entries)

    # ── 6. Dry-run branch ──
    if args.dry_run:
        return _print_train_dryrun(params, config, num_entries)

    # ── 7. Preflight (RAM + battery) ──
    _train_preflight(model, params, force=args.force)

    # ── 8. Confirm gate ──
    if not args.yes:
        wall_h = _estimate_wall_hours(params)
        proceed = prompt_yes_no(
            f"Train LoRA {args.name!r} on {num_entries} images "
            f"({total_steps} steps, ~{wall_h:.1f}h on "
            f"{platform.machine()})? [y/N]: "
        )
        if not proceed:
            info("Training cancelled (no confirmation).")
            return 1

    # ── 9. Materialise scratch ──
    # If a prior failed run left a scratch dir, wipe it before re-materialise
    # (materialise refuses pre-existing dir — caller's responsibility).
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    _train_scratch._materialise_scratch_dataset(scratch_dir, entries)

    # ── 10. Resolve HF token + spawn MfluxEngine.train ──
    hf_token = load_token()
    env = build_mflux_env(token=hf_token)

    # python C-1 closure: rc = -1 BEFORE try so the `finally` block
    # can safely check `rc == 0` even on KeyboardInterrupt / exception
    # before assignment.
    rc: int = -1
    status: str = "fail"
    started = time.monotonic()
    wall_seconds = 0
    try:
        step(f"Training {args.name!r} via mflux-train "
             f"({total_steps} steps; scratch at {scratch_dir})…")
        try:
            rc = MfluxEngine().train(model, params, env=env)
        except KeyboardInterrupt:
            status = "cancelled"
            warn(
                f"Training cancelled by user. Scratch kept at "
                f"{scratch_dir} for inspection or restart."
            )
            raise

        if rc == 0:
            # ── 11. Promote + write meta.json ──
            # mflux-train can exit 0 yet leave no usable checkpoint
            # (e.g. --battery-percentage-stop-limit tripped before the
            # first save, or a full disk on the meta write). Treat a
            # promote/meta failure as a run failure: status stays
            # "fail", scratch is kept (cleanup is gated on status, not
            # rc), and we surface a clean error instead of letting an
            # OSError escape as an uncaught traceback.
            try:
                _train_scratch._promote_final_safetensors(
                    scratch_dir, output_path,
                )
                meta = _train_scratch.build_meta_json(
                    params=params,
                    model=model,
                    num_entries=num_entries,
                    wall_seconds=int(time.monotonic() - started),
                    # v0.10.0: peak observed RAM is not measured live —
                    # would need a sampler thread. Recorded as 0.0
                    # sentinel; pre-tag smoke (§M.1) writes a real number
                    # post-run via doctor inspection. v0.10.x can add a
                    # background pmem sampler if it proves useful.
                    peak_ram_gb_observed=0.0,
                    trained_at_iso=datetime.now().isoformat(timespec="seconds"),
                )
                _train_scratch._write_meta_json(meta_path, meta)
            except OSError as e:
                err(
                    f"mflux-train exited 0 but no usable LoRA could be "
                    f"promoted ({e}); scratch kept at {scratch_dir} for "
                    "diagnosis."
                )
                # status stays "fail" → finally keeps scratch + records
                # the failed run; the post-finally branch returns non-zero.
            else:
                status = "success"
        else:
            err(
                f"mflux-train exited {rc}; scratch kept at "
                f"{scratch_dir} for diagnosis."
            )
            # status stays "fail"; final history append happens in
            # the finally block + the post-finally die().
    finally:
        wall_seconds = int(time.monotonic() - started)
        # ── 12. Cleanup scratch on full success; KEEP on any failure ──
        # Gate on ``status``, NOT ``rc``: a promote failure leaves rc==0
        # but status=="fail", and the scratch dir must survive so the
        # user can inspect what mflux-train actually wrote.
        if status == "success":
            shutil.rmtree(scratch_dir, ignore_errors=True)
        # History entry: always recorded so cmd_history can show
        # the cancelled / failed runs alongside successes.
        _append_train_history(
            params=params,
            num_entries=num_entries,
            wall_seconds=wall_seconds,
            status=status,
        )

    if status != "success":
        # Either mflux-train failed (rc != 0) or it exited 0 but no
        # usable checkpoint could be promoted (rc == 0). Error already
        # surfaced via err(); exit non-zero — prefer mflux-train's own
        # code, falling back to 2 for the promote-failure case.
        return rc if rc != 0 else 2

    ok(
        f"LoRA trained: ~/.imgen/loras/{args.name}.safetensors "
        f"(use with `imgen draw --lora {args.name} \"...\"`)"
    )
    return 0

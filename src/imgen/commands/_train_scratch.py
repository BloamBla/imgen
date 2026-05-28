"""v0.10.0 commit 6 — scratch dir materialisation + atomic promotion.

FS-side helpers sequenced by ``cmd_train`` (commit 8):

1. ``cmd_train`` validates the dataset (commit 3 :func:`validate_dataset_dir`).
2. → :func:`_materialise_scratch_dataset` hardlinks (or copies on cross-FS)
   images + writes caption sidecars into a fresh
   ``~/.imgen/loras/.<name>.training/data/`` workspace with mode 0o700
   (security C-2 — PII-bearing trained weights).
3. ``cmd_train`` writes the mflux JSON via
   :func:`imgen.engines._training.build_config_json`.
4. ``cmd_train`` spawns ``mflux-train --config <FILE>`` (commit 7).
5. → :func:`_promote_final_safetensors` globs
   ``<scratch>/checkpoints/{NNNNNNN}_adapter.safetensors`` (7-digit
   zero-padded iteration; final = highest), atomic-renames to
   ``~/.imgen/loras/<name>.safetensors`` via ``os.replace``.
6. → :func:`_write_meta_json` writes the ``.meta.json`` sidecar with
   ``build_meta_json``-built dict, mode 0o600.
7. ``cmd_train`` removes scratch dir on success; KEEPS scratch on
   failure for inspection.

Per [[project-v100-design]] §E.3 + §H.3 + §R.1 ROUND-1 CLOSURES, as
corrected by the §M.1 32 GB smoke (2026-05-28):

* mflux-train REAL output (verified by smoke, NOT the memo's guess):
  ``<output_path>/checkpoints/{NNNNNNN}_checkpoint.zip`` — a resumable
  training-state archive. Our configured ``output_path`` is
  ``scratch/checkpoints``, so the zips land in
  ``scratch/checkpoints/checkpoints/``. The usable LoRA lives INSIDE
  each zip as ``{NNNNNNN}_adapter.safetensors`` (alongside the
  optimizer state + iterator/config json used by ``mflux-train
  --resume``). The original memo's bare-``{NNNNNNN}_adapter.safetensors``
  convention was wrong — :func:`_promote_final_safetensors` now globs
  the zips, picks the max iteration, and extracts the adapter member.
* meta.json schema includes ``lora_compat_group: str`` (architect H-5 —
  required for compat-checks against ``--model`` at inference).
* Atomic file writes via ``<path>.tmp`` + ``os.replace`` mirror the
  ``history.py`` discipline.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .. import __version__
from ..defaults import MFLUX_PIN

if TYPE_CHECKING:
    from ..engines._training import TrainingParams
    from ..models import Model
    from .train import DatasetEntry

__all__ = [
    "_materialise_scratch_dataset",
    "_promote_final_safetensors",
    "_write_meta_json",
    "build_meta_json",
]


# mflux-train checkpoint archive: 7-digit zero-padded iteration prefix.
# Each zip contains ``{NNNNNNN}_adapter.safetensors`` (the LoRA) + the
# optimizer/iterator state for ``--resume``. Verified by the §M.1 smoke.
_CHECKPOINT_ZIP_RE = re.compile(r"^(\d{7})_checkpoint\.zip$")
# The adapter member name inside the zip mirrors the iteration prefix.
_ADAPTER_MEMBER_SUFFIX = "_adapter.safetensors"


def _materialise_scratch_dataset(
    scratch_dir: Path,
    entries: "list[DatasetEntry]",
) -> None:
    """Create ``scratch_dir`` + ``scratch_dir/data/`` (mode 0o700),
    hardlink (or copy on cross-FS) every entry's image, write each
    entry's caption to a sibling ``.txt`` sidecar.

    Raises:
      * ``FileExistsError`` if ``scratch_dir`` already exists. Caller
        (cmd_train) is responsible for cleaning up failed-run scratch
        before re-invocation; the helper refuses to mix old + new.
      * ``ValueError`` on empty ``entries`` — defence-in-depth above
        the upstream :func:`validate_dataset_dir` reject.
      * ``OSError`` on FS-level failures other than cross-FS (full
        disk, permission denied) — propagated to the caller.

    The hardlink-first strategy keeps disk usage flat for the typical
    case (source + scratch both under ``~/``). On cross-FS sources
    (external SSD, NAS mount) ``os.link`` raises ``OSError`` and we
    fall back to ``shutil.copy2`` per file (preserves mtime + perms).
    """
    if not entries:
        raise ValueError(
            "_materialise_scratch_dataset: entries is empty — caller "
            "must validate dataset has at least one image upstream."
        )

    # Refuse pre-existing scratch dir (exclusive create at the leaf).
    scratch_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    data_dir = scratch_dir / "data"
    data_dir.mkdir(mode=0o700)

    for entry in entries:
        dst_img = data_dir / entry.image_path.name
        try:
            os.link(entry.image_path, dst_img)
        except OSError:
            # Cross-FS, link unsupported, or other link-specific
            # failure — fall back to copy (preserves perms + mtime).
            shutil.copy2(entry.image_path, dst_img)

        # Caption sidecar — always a fresh write (no copy from source
        # sidecar even when it exists, because :func:`validate_dataset_dir`
        # has already resolved the caption via trigger fallback for
        # empty/missing sidecars).
        sidecar = data_dir / (entry.image_path.stem + ".txt")
        sidecar.write_text(entry.caption, encoding="utf-8")


def _promote_final_safetensors(
    scratch_dir: Path,
    output_path: Path,
) -> Path:
    """Find the highest-iteration ``{NNNNNNN}_checkpoint.zip`` written by
    mflux-train, extract its ``{NNNNNNN}_adapter.safetensors`` member, and
    write it atomically (mode 0o600) to ``output_path``.

    mflux-train nests its checkpoints under
    ``<config.output_path>/checkpoints/``; our config sets ``output_path``
    to ``scratch_dir/checkpoints``, so the zips land in
    ``scratch_dir/checkpoints/checkpoints/`` (verified by the §M.1 smoke).
    Each zip is a resumable training-state archive holding the adapter +
    optimizer state; we lift out only the adapter safetensors.

    Returns the ``output_path`` on success.

    Raises:
      * ``FileNotFoundError`` if the checkpoints dir is missing, contains
        zero ``{NNNNNNN}_checkpoint.zip`` files, or the chosen zip has no
        ``*_adapter.safetensors`` member — mflux-train produced no usable
        LoRA, caller surfaces as a clean training failure.
    """
    ckpt_dir = scratch_dir / "checkpoints" / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"mflux-train checkpoints dir missing: {ckpt_dir} — "
            "training likely failed before any checkpoint save."
        )

    # Filter to the canonical 7-digit checkpoint-zip shape, pick max
    # iteration (the latest = most-trained; iteration 0 is the pre-train
    # snapshot).
    candidates: list[tuple[int, Path]] = []
    for entry in ckpt_dir.iterdir():
        m = _CHECKPOINT_ZIP_RE.match(entry.name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), entry))

    if not candidates:
        raise FileNotFoundError(
            f"no {{NNNNNNN}}_checkpoint.zip files found in {ckpt_dir} — "
            "mflux-train produced no LoRA artifact."
        )

    candidates.sort(key=lambda pair: pair[0])
    final_iter, final_zip = candidates[-1]

    # Pull the adapter member out of the zip. Read by exact name first
    # (``{NNNNNNN}_adapter.safetensors``), fall back to any member ending
    # in ``_adapter.safetensors``. We read the single member into memory
    # (~27 MB) rather than extractall — no zip-slip path-traversal risk.
    with zipfile.ZipFile(final_zip) as zf:
        names = zf.namelist()
        member = f"{final_iter:07d}{_ADAPTER_MEMBER_SUFFIX}"
        if member not in names:
            fallback = [n for n in names if n.endswith(_ADAPTER_MEMBER_SUFFIX)]
            if not fallback:
                raise FileNotFoundError(
                    f"checkpoint zip {final_zip.name} has no "
                    f"*_adapter.safetensors member (saw {names!r}) — "
                    "mflux-train output shape changed; promotion needs review."
                )
            member = fallback[-1]
        adapter_bytes = zf.read(member)

    # Atomic write, mode 0o600 — trained weights are PII-bearing identity
    # data (mirror _write_meta_json discipline). tmp + os.replace.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(adapter_bytes)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return output_path


def build_meta_json(
    *,
    params: "TrainingParams",
    model: "Model",
    num_entries: int,
    wall_seconds: int,
    peak_ram_gb_observed: float,
    trained_at_iso: str,
    imgen_version: str | None = None,
) -> dict:
    """Pure: returns the ``.meta.json`` dict for one trained LoRA.

    Read by ``--lora <name>`` resolver (commit 10 trigger auto-prepend),
    by ``imgen doctor`` (lists trained LoRAs), and by ``imgen replay``
    for the ``"train"`` command (rebuilds equivalent invocation).

    Schema ``version=1`` per [[project-v100-design]] §H.3. A v0.10.x
    bump can add fields without breaking readers; renames/removals
    require ``version=2``.

    ``imgen_version`` defaults to the runtime ``imgen.__version__``;
    tests pass an explicit value so they don't drift with version
    bumps.
    """
    if imgen_version is None:
        imgen_version = __version__

    # MFLUX_PIN is the pip spec ``"mflux==0.17.5"``; strip prefix for
    # human-readable display in the meta-json + future ``imgen doctor``
    # rows.
    if "==" in MFLUX_PIN:
        mflux_version = MFLUX_PIN.split("==", 1)[1]
    else:
        mflux_version = MFLUX_PIN

    return {
        "version": 1,
        "lora_name": params.lora_name,
        "trigger": params.trigger,
        "dataset_path": str(params.dataset_dir),
        "dataset_image_count": num_entries,
        "base_model": params.base_model,
        # §R.1 architect H-5 closure: compat-check field for
        # ``--lora <name>`` × ``--model <other>`` at inference.
        "lora_compat_group": model.lora_compat_group,
        "total_steps": params.total_steps,
        "lora_rank": params.lora_rank,
        "quantize": params.quantize,
        "max_resolution": params.max_resolution,
        "optimizer_name": params.optimizer_name,
        "optimizer_lr": params.optimizer_lr,
        "preview_frequency": params.preview_frequency,
        "seed": params.seed,
        "trained_at": trained_at_iso,
        "imgen_version": imgen_version,
        "mflux_version": mflux_version,
        "wall_seconds": wall_seconds,
        "training_peak_ram_gb_observed": peak_ram_gb_observed,
    }


def _write_meta_json(path: Path, meta: dict) -> None:
    """Atomic write of ``meta`` as pretty-printed JSON to ``path``.

    Pattern mirrors ``history.append`` discipline: write to
    ``<path>.tmp`` (mode 0o600 at open), then ``os.replace`` to the
    final path. ``os.replace`` is atomic on the same FS.

    Mode 0o600 is enforced via ``os.open`` with explicit perms — a
    later ``chmod`` would race against a reader on the same path.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # Open with explicit mode so umask doesn't widen the perms.
    fd = os.open(
        tmp_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
        # Defence-in-depth: re-apply mode in case the file pre-existed
        # with wider perms (umask + O_TRUNC keeps the old mode bits).
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup of the tmp file on any error path —
        # prevents orphan .tmp files cluttering ~/.imgen/loras/.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

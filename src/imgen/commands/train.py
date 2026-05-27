"""v0.10.0 commit 3 — `imgen train` skeleton + dataset validator.

Per [[project-v100-design]] §G + §D.3 + §R.1 ROUND-1 CLOSURES preamble.

Ships at commit 3 (skeleton; full flow lands at commit 8):

* :class:`UserDatasetError` — raised on user-correctable dataset
  issues; cli.py catches and converts to die().
* :class:`DatasetEntry` — one validated row (image_path + caption).
* :func:`validate_dataset_dir` — pure-function validator with symlink
  reject, size cap, PIL decompression-bomb gate, sidecar UTF-8 strict
  decode, control-byte filter, trigger fallback, trigger-presence
  warn (§M.3).
* :func:`cmd_train` — subcommand handler stub raising
  ``NotImplementedError`` until commit 8 wires the real flow (preflight,
  confirm gate, ``MfluxEngine.train`` dispatch, history append).

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
"""
from __future__ import annotations

import stat as _stat
from dataclasses import dataclass
from pathlib import Path

from .._safe import has_control_bytes
from ..colors import warn


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


def cmd_train(args) -> int:
    """v0.10.0 — ``imgen train`` subcommand handler.

    Commit 3 ships ONLY this stub + the dataset validator. The real
    flow lands at commit 8 per [[project-v100-design]] §G:

    1. Validate dataset via :func:`validate_dataset_dir`.
    2. Resolve base model + TrainingConfig from registry.
    3. Resolve output path + collision check (--overwrite).
    4. Build :class:`TrainingParams` from CLI args > TrainingConfig defaults.
    5. Trigger-collision warning (informational).
    6. ``--dry-run`` branch: print config + invocation, exit.
    7. Preflight: RAM + battery + mflux-train binary check.
    8. Confirm gate (10h wall acknowledged).
    9. Materialise scratch + invoke ``MfluxEngine.train(model, params)``.
    10. Promote final ``.safetensors`` + write ``<name>.meta.json``.
    11. Cleanup scratch on success; KEEP scratch on failure.
    12. ``history.append`` for replay-CONFIRM-GATE round-trip.

    Per memo §Q commit 3: this stub raises NotImplementedError so the
    skeleton + import shape are exercisable while later commits land
    the orchestration.
    """
    raise NotImplementedError(
        "cmd_train: full flow lands at v0.10.0 commit 8. "
        "Commit 3 ships only the dataset validator + import shape "
        "(skeleton stub). See [[project-v100-design]] §G + §R.1 "
        "ROUND-1 CLOSURES."
    )

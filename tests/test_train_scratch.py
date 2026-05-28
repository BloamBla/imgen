"""v0.10.0 commit 6 — scratch dir materialisation + atomic promotion.

Covers the FS-side helpers that ``cmd_train`` (commit 8) will sequence:

* :func:`_materialise_scratch_dataset` — copies/hardlinks dataset
  images and caption sidecars into a fresh scratch dir under
  ``~/.imgen/loras/.<name>.training/data/``.
* :func:`_promote_final_safetensors` — globs the highest-iteration
  ``{NNNNNNN}_checkpoint.zip`` from
  ``<scratch>/checkpoints/checkpoints/``, extracts the
  ``{NNNNNNN}_adapter.safetensors`` member, and writes it (0o600) to
  ``~/.imgen/loras/<name>.safetensors``. (Real mflux output shape per
  the §M.1 smoke — NOT the bare-file convention the memo guessed.)
* :func:`build_meta_json` — pure dict builder for the ``.meta.json``
  sidecar (read by ``--lora <name>`` resolver + ``imgen doctor``).
* :func:`_write_meta_json` — atomic write of the meta-json dict
  with mode 0o600.

Per [[project-v100-design]] §E.3 + §H.3 + §R.1 ROUND-1 CLOSURES.
"""
from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from imgen.commands._train_scratch import (
    _materialise_scratch_dataset,
    _promote_final_safetensors,
    _write_meta_json,
    build_meta_json,
)
from imgen.commands.train import DatasetEntry
from imgen.models import _KLEIN_4B_TARGET_MODULES, BUILTIN_MODELS


def _make_dataset(tmp_path: Path, n: int = 2) -> list[DatasetEntry]:
    """Make ``n`` PNG entries with caption sidecars in ``tmp_path``."""
    from PIL import Image
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries: list[DatasetEntry] = []
    for i in range(n):
        img_path = tmp_path / f"photo{i}.png"
        Image.new("RGB", (64, 64), "red").save(img_path)
        entries.append(
            DatasetEntry(
                image_path=img_path, caption=f"a test photo {i}, al1na woman",
            )
        )
    return entries


# ── _materialise_scratch_dataset ─────────────────────────────────

class TestMaterialiseScratchDataset:
    def test_creates_data_subdir(self, tmp_path):
        entries = _make_dataset(tmp_path / "src", n=2)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        assert (scratch / "data").is_dir()

    def test_scratch_dir_mode_is_0o700(self, tmp_path):
        """Security C-2: PII-bearing trained weights — restrict to user."""
        entries = _make_dataset(tmp_path / "src", n=1)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        mode = scratch.stat().st_mode & 0o777
        assert mode == 0o700, f"scratch_dir mode {oct(mode)} != 0o700"

    def test_data_subdir_mode_is_0o700(self, tmp_path):
        entries = _make_dataset(tmp_path / "src", n=1)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        mode = (scratch / "data").stat().st_mode & 0o777
        assert mode == 0o700

    def test_images_present_in_data_dir(self, tmp_path):
        entries = _make_dataset(tmp_path / "src", n=3)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        for entry in entries:
            assert (scratch / "data" / entry.image_path.name).is_file()

    def test_caption_sidecars_written(self, tmp_path):
        entries = _make_dataset(tmp_path / "src", n=2)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        for entry in entries:
            sidecar = scratch / "data" / (entry.image_path.stem + ".txt")
            assert sidecar.is_file()
            assert sidecar.read_text(encoding="utf-8") == entry.caption

    def test_hardlinks_when_same_fs(self, tmp_path):
        """Typical case — source dataset + scratch both under ~/. The
        hardlink keeps disk usage flat across the run and the source
        file unchanged (same inode)."""
        entries = _make_dataset(tmp_path / "src", n=1)
        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)
        src_inode = entries[0].image_path.stat().st_ino
        dst_inode = (scratch / "data" / entries[0].image_path.name).stat().st_ino
        assert src_inode == dst_inode, "expected hardlink (same inode)"

    def test_copy_fallback_when_os_link_fails(
        self, tmp_path, monkeypatch,
    ):
        """Cross-FS dataset OR pre-existing-target — ``os.link`` raises
        ``OSError`` and the materialiser must fall back to
        ``shutil.copy2``. Locked because production cases (external SSD
        dataset, Time Machine dest, NAS mount) exercise this branch."""
        entries = _make_dataset(tmp_path / "src", n=1)
        scratch = tmp_path / ".alina.training"

        call_count = {"link": 0, "copy2": 0}
        real_copy2 = __import__("shutil").copy2

        def boom_link(src, dst):
            call_count["link"] += 1
            raise OSError("simulated cross-FS")

        def counting_copy2(src, dst, *args, **kwargs):
            call_count["copy2"] += 1
            return real_copy2(src, dst, *args, **kwargs)

        monkeypatch.setattr("os.link", boom_link)
        monkeypatch.setattr(
            "imgen.commands._train_scratch.shutil.copy2", counting_copy2,
        )

        _materialise_scratch_dataset(scratch, entries)

        assert call_count["link"] >= 1, "os.link must be tried first"
        assert call_count["copy2"] >= 1, "must fall back to copy2 on link OSError"
        assert (scratch / "data" / entries[0].image_path.name).is_file()

    def test_rejects_pre_existing_scratch_dir(self, tmp_path):
        """Caller (cmd_train) is responsible for cleaning up failed-run
        scratch before re-invocation. Helper raises so a stale scratch
        dir doesn't silently mix old + new data."""
        entries = _make_dataset(tmp_path / "src", n=1)
        scratch = tmp_path / ".alina.training"
        scratch.mkdir()
        with pytest.raises(FileExistsError):
            _materialise_scratch_dataset(scratch, entries)

    def test_source_dataset_unchanged_after_materialise(self, tmp_path):
        """Read-only source semantic: source file size + mtime preserved
        after the materialise. Catches accidental mutation."""
        entries = _make_dataset(tmp_path / "src", n=1)
        src_path = entries[0].image_path
        src_size = src_path.stat().st_size
        src_mtime = src_path.stat().st_mtime

        scratch = tmp_path / ".alina.training"
        _materialise_scratch_dataset(scratch, entries)

        assert src_path.stat().st_size == src_size
        assert src_path.stat().st_mtime == src_mtime

    def test_empty_entries_rejected(self, tmp_path):
        """Empty dataset = caller bug. ``validate_dataset_dir`` already
        rejects empty datasets; defence-in-depth here."""
        scratch = tmp_path / ".alina.training"
        with pytest.raises(ValueError, match="entries"):
            _materialise_scratch_dataset(scratch, [])


# ── _promote_final_safetensors ──────────────────────────────────


def _write_checkpoint_zip(
    scratch: Path,
    iteration: int,
    adapter_bytes: bytes,
    *,
    adapter_member: str | None = None,
    extra_members: bool = True,
) -> Path:
    """Build a ``{NNNNNNN}_checkpoint.zip`` under
    ``scratch/checkpoints/checkpoints/`` containing the adapter member
    (+ optimizer/run decoys), mirroring real mflux-train output verified
    by the §M.1 smoke."""
    ckpt_dir = scratch / "checkpoints" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ckpt_dir / f"{iteration:07d}_checkpoint.zip"
    member = adapter_member or f"{iteration:07d}_adapter.safetensors"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(member, adapter_bytes)
        if extra_members:
            zf.writestr(f"{iteration:07d}_optimizer.safetensors", b"optstate")
            zf.writestr("run.json", b"{}")
    return zip_path


class TestPromoteFinalSafetensors:
    def test_picks_single_checkpoint(self, tmp_path):
        scratch = tmp_path / ".alina.training"
        _write_checkpoint_zip(scratch, 800, b"\x00\x01\x02")
        output = tmp_path / "alina.safetensors"
        promoted = _promote_final_safetensors(scratch, output)
        assert promoted == output
        assert output.is_file()
        assert output.read_bytes() == b"\x00\x01\x02"

    def test_picks_max_iteration_among_many(self, tmp_path):
        """mflux-train writes ``{NNNNNNN}_checkpoint.zip`` (7-digit
        zero-padded; iteration 0 is the pre-train snapshot). Final =
        highest iteration count."""
        scratch = tmp_path / ".alina.training"
        # Out of order to verify sort isn't by listdir() order; include
        # the iteration-0 pre-train snapshot which must NOT be picked.
        for idx in (0, 200, 800, 400, 100):
            _write_checkpoint_zip(scratch, idx, f"weights-{idx}".encode())
        output = tmp_path / "alina.safetensors"
        _promote_final_safetensors(scratch, output)
        assert output.read_bytes() == b"weights-800"

    def test_output_mode_is_0o600(self, tmp_path):
        """Trained weights are PII-bearing identity data → promoted file
        is mode 0o600 (mirror _write_meta_json)."""
        scratch = tmp_path / ".alina.training"
        _write_checkpoint_zip(scratch, 50, b"x")
        output = tmp_path / "alina.safetensors"
        _promote_final_safetensors(scratch, output)
        assert stat.S_IMODE(output.stat().st_mode) == 0o600

    def test_source_zip_is_not_consumed(self, tmp_path):
        """Extraction copies the adapter out of the zip — the resumable
        checkpoint zip itself survives (cmd_train rmtree's the whole
        scratch dir on success; promotion alone must not delete it)."""
        scratch = tmp_path / ".alina.training"
        zip_path = _write_checkpoint_zip(scratch, 50, b"x")
        output = tmp_path / "alina.safetensors"
        _promote_final_safetensors(scratch, output)
        assert zip_path.exists()

    def test_raises_when_no_checkpoint_zips(self, tmp_path):
        """``mflux-train`` produced zero checkpoint zips — training failed
        silently. Caller (cmd_train) should surface this as an error,
        not promote nothing."""
        scratch = tmp_path / ".alina.training"
        (scratch / "checkpoints" / "checkpoints").mkdir(parents=True)
        output = tmp_path / "alina.safetensors"
        with pytest.raises(FileNotFoundError, match="checkpoint.zip"):
            _promote_final_safetensors(scratch, output)

    def test_raises_when_checkpoints_dir_missing(self, tmp_path):
        """Scratch dir exists but mflux-train never wrote the nested
        checkpoints/checkpoints/ dir."""
        scratch = tmp_path / ".alina.training"
        scratch.mkdir()
        output = tmp_path / "alina.safetensors"
        with pytest.raises(FileNotFoundError):
            _promote_final_safetensors(scratch, output)

    def test_raises_when_only_pretrain_snapshot(self, tmp_path):
        """Iteration 0 is the pre-train snapshot. If it's the only (hence
        max) checkpoint — e.g. --steps at the floor saved nothing past it
        — promoting would write an untrained adapter. Must raise."""
        scratch = tmp_path / ".alina.training"
        _write_checkpoint_zip(scratch, 0, b"pretrain-snapshot")
        output = tmp_path / "alina.safetensors"
        with pytest.raises(FileNotFoundError, match="pre-train snapshot"):
            _promote_final_safetensors(scratch, output)
        assert not output.exists()

    def test_raises_when_zip_has_no_adapter_member(self, tmp_path):
        """A checkpoint zip with no ``*_adapter.safetensors`` member means
        mflux-train's output shape changed — surface loudly, don't write
        an empty/garbage LoRA."""
        scratch = tmp_path / ".alina.training"
        _write_checkpoint_zip(
            scratch, 100, b"unused",
            adapter_member="0000100_optimizer.safetensors",
            extra_members=False,
        )
        output = tmp_path / "alina.safetensors"
        with pytest.raises(FileNotFoundError, match="adapter.safetensors"):
            _promote_final_safetensors(scratch, output)

    def test_ignores_non_zip_files_in_checkpoints(self, tmp_path):
        """Decoy files (logs, loss html, partial files) in the nested
        checkpoints dir must not be picked. Only
        ``{NNNNNNN}_checkpoint.zip``."""
        scratch = tmp_path / ".alina.training"
        ckpt_dir = scratch / "checkpoints" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "training.log").write_text("log")
        (ckpt_dir / "0000050_checkpoint.bin").write_bytes(b"wrong-ext")
        # Real one we want picked.
        _write_checkpoint_zip(scratch, 100, b"real")
        output = tmp_path / "alina.safetensors"
        _promote_final_safetensors(scratch, output)
        assert output.read_bytes() == b"real"

    def test_overwrites_existing_output_path(self, tmp_path):
        """The cmd_train flow has already collision-checked
        ``--overwrite``; the tmp + os.replace here overwrites
        unconditionally so a half-promoted leftover doesn't block
        the final write."""
        scratch = tmp_path / ".alina.training"
        _write_checkpoint_zip(scratch, 800, b"new")
        output = tmp_path / "alina.safetensors"
        output.write_bytes(b"old")
        _promote_final_safetensors(scratch, output)
        assert output.read_bytes() == b"new"


# ── build_meta_json ──────────────────────────────────────────────

def _klein_4b_params(tmp_path: Path):
    """Minimal training params on absolute tmp_path. Mirrors the
    test_build_config_json fixture but uses tmp_path so paths are
    real for FS tests."""
    from imgen.engines._training import TrainingParams
    return TrainingParams(
        dataset_dir=tmp_path / "datasets" / "alina",
        scratch_dir=tmp_path / "loras" / ".alina.training",
        lora_name="alina",
        trigger="al1na woman",
        base_model="flux2-klein-4b",
        total_steps=880,
        lora_rank=16,
        max_resolution=512,
        quantize=4,
        low_ram=True,
        optimizer_name="AdamW",
        optimizer_lr=1e-4,
        target_modules=_KLEIN_4B_TARGET_MODULES,
        preview_frequency=100,
        seed=42,
        battery_stop=20,
        output_path=tmp_path / "loras" / "alina.safetensors",
    )


class TestBuildMetaJson:
    def _build(self, tmp_path):
        return build_meta_json(
            params=_klein_4b_params(tmp_path),
            model=BUILTIN_MODELS["flux2-klein-4b"],
            num_entries=10,
            wall_seconds=36000,
            peak_ram_gb_observed=27.3,
            trained_at_iso="2026-05-28T03:14:15",
            imgen_version="0.10.0",
        )

    def test_version_is_one(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["version"] == 1

    def test_lora_name_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["lora_name"] == "alina"

    def test_trigger_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["trigger"] == "al1na woman"

    def test_dataset_path_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["dataset_path"] == str(tmp_path / "datasets" / "alina")

    def test_dataset_image_count_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["dataset_image_count"] == 10

    def test_base_model_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["base_model"] == "flux2-klein-4b"

    def test_total_steps_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["total_steps"] == 880

    def test_training_run_params_propagate(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["lora_rank"] == 16
        assert meta["quantize"] == 4
        assert meta["max_resolution"] == 512
        assert meta["optimizer_name"] == "AdamW"
        assert meta["optimizer_lr"] == 1e-4
        assert meta["preview_frequency"] == 100
        assert meta["seed"] == 42

    def test_lora_compat_group_field(self, tmp_path):
        """§R.1 architect H-5 closure: ``.meta.json`` schema gains
        ``lora_compat_group: str`` so ``--lora <name>`` can compat-check
        against ``--model`` at inference time. Sourced from the base
        Model's row (NOT recomputed)."""
        meta = self._build(tmp_path)
        assert meta["lora_compat_group"] == "flux2-klein-4b"

    def test_trained_at_iso_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["trained_at"] == "2026-05-28T03:14:15"

    def test_imgen_version_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["imgen_version"] == "0.10.0"

    def test_mflux_version_field_strips_pkg_prefix(self, tmp_path):
        """``MFLUX_PIN`` is the pip spec ``"mflux==0.17.5"``; meta-json
        stores just ``"0.17.5"`` for human-readable display."""
        meta = self._build(tmp_path)
        assert meta["mflux_version"] == "0.17.5"

    def test_wall_seconds_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["wall_seconds"] == 36000

    def test_peak_ram_field(self, tmp_path):
        meta = self._build(tmp_path)
        assert meta["training_peak_ram_gb_observed"] == 27.3

    def test_round_trips_through_json_dumps(self, tmp_path):
        meta = self._build(tmp_path)
        text = json.dumps(meta)
        re_parsed = json.loads(text)
        assert re_parsed == meta

    def test_returns_new_dict_each_call(self, tmp_path):
        meta1 = self._build(tmp_path)
        meta2 = self._build(tmp_path)
        assert meta1 == meta2
        assert meta1 is not meta2


# ── _write_meta_json ─────────────────────────────────────────────

class TestWriteMetaJson:
    def _build(self, tmp_path):
        return build_meta_json(
            params=_klein_4b_params(tmp_path),
            model=BUILTIN_MODELS["flux2-klein-4b"],
            num_entries=10,
            wall_seconds=36000,
            peak_ram_gb_observed=27.3,
            trained_at_iso="2026-05-28T03:14:15",
            imgen_version="0.10.0",
        )

    def test_writes_file(self, tmp_path):
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        _write_meta_json(path, meta)
        assert path.is_file()

    def test_file_content_is_valid_json(self, tmp_path):
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        _write_meta_json(path, meta)
        with path.open(encoding="utf-8") as f:
            re_parsed = json.load(f)
        assert re_parsed == meta

    def test_file_mode_is_0o600(self, tmp_path):
        """Security C-2: meta.json carries dataset_path + trigger word
        (potential PII for personal-identity LoRAs). Restrict to owner."""
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        _write_meta_json(path, meta)
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"meta.json mode {oct(mode)} != 0o600"

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path):
        """Atomic write = write to ``<path>.tmp`` then ``os.replace`` —
        no orphan ``.tmp`` should remain on success."""
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        _write_meta_json(path, meta)
        tmp_paths = list(tmp_path.glob("*.tmp"))
        assert tmp_paths == [], f"orphan tmp files: {tmp_paths!r}"

    def test_overwrites_existing_file(self, tmp_path):
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        path.write_text('{"old": true}')
        _write_meta_json(path, meta)
        with path.open(encoding="utf-8") as f:
            assert json.load(f) == meta

    def test_content_is_pretty_printed(self, tmp_path):
        """Human-readable indent — colleagues will eyeball this file
        when inspecting trained LoRAs. ``json.dumps`` default is
        compact one-liner; we want indent."""
        meta = self._build(tmp_path)
        path = tmp_path / "alina.meta.json"
        _write_meta_json(path, meta)
        text = path.read_text(encoding="utf-8")
        assert "\n" in text, "meta.json should be multi-line for human read"

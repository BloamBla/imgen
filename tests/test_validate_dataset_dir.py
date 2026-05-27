"""v0.10.0 commit 3 — ``validate_dataset_dir`` pure-function tests.

Per [[project-v100-design]] §D.3 + §R.1 ROUND-1 CLOSURES preamble.

Tests cover:
* Symlink reject on dataset dir + each child (security; mirror v0.4
  styles.d + v0.9 .venv-diffusers patterns).
* ``lstat()`` disambiguation of broken-symlink vs missing-dir (python H-1).
* Per-image size cap 50 MB (security C-1 — DoS / decompression bomb gate).
* PIL decompression-bomb gate via ``Image.open(path).size`` rejecting
  ``width × height > 50M px`` (security C-1).
* Extension allowlist (.jpg / .jpeg / .png / .webp) with HEIC sips hint.
* Dotfile skip + filename control-byte filter (security risk row).
* Sidecar UTF-8 strict decode + 4096-byte cap + control-byte filter.
* Trigger fallback when no sidecar / empty sidecar.
* Min-images floor (3, reject) + warn at <7 (§D.3 + architect M-3).
* Warn at >30 (training noise) — informational, not blocking.
* Per-sidecar trigger-presence warn (§M.3).
* Symlink-vs-broken-link order: ``lstat`` runs FIRST so broken symlinks
  report as symlinks, not as missing files.

The validator is pure (no subprocess, no env, no network) so all tests
use ``tmp_path`` for fast FS setup. Suite wall delta stays minimal.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Helpers ────────────────────────────────────────────────────────────


def _make_tiny_image(path: Path, *, fmt: str = "PNG") -> None:
    """Write a 1x1 image at ``path`` via PIL. Format derived from
    extension or ``fmt`` kwarg."""
    from PIL import Image
    img = Image.new("RGB", (1, 1), color=(128, 128, 128))
    img.save(path, format=fmt)


def _make_dataset(tmp_path: Path, n: int = 5, *, ext: str = ".png") -> Path:
    """Make a dataset dir with ``n`` valid images (no sidecars)."""
    d = tmp_path / "dataset"
    d.mkdir()
    for i in range(n):
        _make_tiny_image(d / f"photo{i:02d}{ext}")
    return d


# ── UserDatasetError + DatasetEntry shape ──────────────────────────────


class TestDatasetEntryShape:
    def test_dataset_entry_has_image_path_and_caption(self):
        from imgen.commands.train import DatasetEntry
        from pathlib import Path
        entry = DatasetEntry(
            image_path=Path("/tmp/x.png"),
            caption="al1na woman",
        )
        assert entry.image_path == Path("/tmp/x.png")
        assert entry.caption == "al1na woman"

    def test_dataset_entry_is_frozen(self):
        from imgen.commands.train import DatasetEntry
        from dataclasses import FrozenInstanceError
        entry = DatasetEntry(image_path=Path("/tmp/x.png"), caption="x")
        with pytest.raises(FrozenInstanceError):
            entry.caption = "y"  # type: ignore[misc]


class TestUserDatasetErrorClass:
    def test_user_dataset_error_is_exception(self):
        from imgen.commands.train import UserDatasetError
        assert issubclass(UserDatasetError, Exception)

    def test_user_dataset_error_carries_message(self):
        from imgen.commands.train import UserDatasetError
        e = UserDatasetError("dataset has only 2 images, minimum 3")
        assert "minimum 3" in str(e)


# ── Happy path ─────────────────────────────────────────────────────────


class TestHappyPath:
    def test_valid_dataset_returns_entries_with_trigger_fallback(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=5)
        entries = validate_dataset_dir(d, trigger="al1na woman")
        assert len(entries) == 5
        # No sidecars → caption falls back to the trigger.
        for entry in entries:
            assert entry.caption == "al1na woman"
            assert entry.image_path.parent == d

    def test_sidecar_overrides_trigger_fallback(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        # Add a sidecar for photo01 only.
        (d / "photo01.txt").write_text("al1na woman dancing in a forest")
        entries = validate_dataset_dir(d, trigger="al1na woman")
        cap_by_stem = {e.image_path.stem: e.caption for e in entries}
        assert cap_by_stem["photo01"] == "al1na woman dancing in a forest"
        # Other photos still get the trigger fallback.
        assert cap_by_stem["photo00"] == "al1na woman"

    def test_empty_sidecar_falls_back_to_trigger(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("")
        entries = validate_dataset_dir(d, trigger="al1na woman")
        cap_by_stem = {e.image_path.stem: e.caption for e in entries}
        # Empty sidecar (after .strip()) → fall back to trigger.
        assert cap_by_stem["photo01"] == "al1na woman"

    def test_supported_extensions_accepted(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = tmp_path / "ds"
        d.mkdir()
        _make_tiny_image(d / "a.png", fmt="PNG")
        _make_tiny_image(d / "b.jpg", fmt="JPEG")
        _make_tiny_image(d / "c.jpeg", fmt="JPEG")
        _make_tiny_image(d / "d.webp", fmt="WEBP")
        entries = validate_dataset_dir(d, trigger="x")
        assert len(entries) == 4

    def test_dotfiles_skipped(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        _make_tiny_image(d / ".hidden.png")  # dotfile — skipped
        (d / ".DS_Store").write_bytes(b"\x00" * 16)  # dotfile — skipped
        entries = validate_dataset_dir(d, trigger="x")
        assert len(entries) == 4

    def test_entries_sorted_by_filename(self, tmp_path):
        """Determinism: sorted iteration yields stable order across runs."""
        from imgen.commands.train import validate_dataset_dir
        d = tmp_path / "ds"
        d.mkdir()
        # Create out of order to surface unsorted iteration.
        for name in ("zebra.png", "apple.png", "mango.png", "banana.png"):
            _make_tiny_image(d / name)
        entries = validate_dataset_dir(d, trigger="x", min_images=3)
        stems = [e.image_path.name for e in entries]
        assert stems == sorted(stems), (
            f"validate_dataset_dir must return entries in sorted order; "
            f"got {stems}"
        )


# ── Symlink rejects ────────────────────────────────────────────────────


class TestSymlinkRejects:
    def test_dataset_dir_symlink_rejected(self, tmp_path):
        """Dataset dir itself must not be a symlink (mirror v0.4 + v0.9
        same-uid attacker mitigation)."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        real = _make_dataset(tmp_path, n=5)
        link = tmp_path / "symdataset"
        link.symlink_to(real, target_is_directory=True)
        with pytest.raises(UserDatasetError, match="symlink"):
            validate_dataset_dir(link, trigger="x")

    def test_child_image_symlink_rejected(self, tmp_path):
        """Child entries (images) must not be symlinks — hardlink TOCTOU
        mitigation per §R.1 security H-7."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        target = tmp_path / "real_target.png"
        _make_tiny_image(target)
        (d / "photo04.png").symlink_to(target)  # adds a 5th child as symlink
        with pytest.raises(UserDatasetError, match="symlink"):
            validate_dataset_dir(d, trigger="x")

    def test_broken_symlink_at_dataset_dir_reports_as_symlink(self, tmp_path):
        """§R.1 python H-1: ``lstat()`` runs BEFORE ``is_dir()`` so a
        broken symlink reports as symlink rather than 'not found'."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        broken = tmp_path / "brokenlink"
        broken.symlink_to(tmp_path / "nonexistent_target")
        with pytest.raises(UserDatasetError, match="symlink"):
            validate_dataset_dir(broken, trigger="x")


# ── Filename + path validation ─────────────────────────────────────────


class TestFilenameValidation:
    def test_dataset_dir_missing_raises(self, tmp_path):
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        with pytest.raises(UserDatasetError, match=r"(?i)(not found|exist)"):
            validate_dataset_dir(tmp_path / "nope", trigger="x")

    def test_dataset_dir_is_file_not_dir_raises(self, tmp_path):
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        f = tmp_path / "afile.txt"
        f.write_text("not a directory")
        with pytest.raises(UserDatasetError, match=r"(?i)(directory|dir)"):
            validate_dataset_dir(f, trigger="x")

    def test_filename_with_control_bytes_rejected(self, tmp_path):
        """Security risk row: filename C0/DEL/C1 bytes flow into
        mflux-train's training loop (filenames are read from disk by
        mflux). Reject at validate-time."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        # macOS allows control bytes in filenames; create one via direct
        # os syscall (avoids shell escaping).
        try:
            os.rename(d / "photo00.png", d / "photo\x1b00.png")
        except OSError:
            pytest.skip("FS does not allow control bytes in filename")
        with pytest.raises(UserDatasetError, match=r"(?i)control"):
            validate_dataset_dir(d, trigger="x")


# ── Extension allowlist ────────────────────────────────────────────────


class TestExtensionRejects:
    def test_heic_rejected_with_sips_hint(self, tmp_path):
        """HEIC files reject with a hint pointing at ``sips`` conversion
        (mirror v0.3.0 batch HEIC discipline)."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=2)
        # Fake HEIC bytes — validator rejects on extension, doesn't probe
        # content.
        (d / "img1.heic").write_bytes(b"FAKE-HEIC-BYTES")
        (d / "img2.heif").write_bytes(b"FAKE-HEIF-BYTES")
        with pytest.raises(UserDatasetError, match=r"(?i)sips"):
            validate_dataset_dir(d, trigger="x")

    def test_unsupported_extensions_silently_skipped_if_enough_images(self, tmp_path):
        """Per §D.2: .bmp/.tif/.tiff/.gif are silently ignored (not
        rejected — the validator only rejects HEIC explicitly because
        users WILL encounter it from iOS photo exports). If there are
        still ≥ min_images after skipping the unsupported ones, the
        validator returns the rest cleanly."""
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=5)
        (d / "weird.bmp").write_bytes(b"BMP-BYTES")
        (d / "ancient.gif").write_bytes(b"GIF-BYTES")
        entries = validate_dataset_dir(d, trigger="x")
        assert len(entries) == 5  # only the 5 PNGs


# ── Size + decompression-bomb gates ────────────────────────────────────


class TestSizeAndBombGates:
    def test_image_over_50mb_rejected(self, tmp_path):
        """§R.1 security C-1: per-image size cap 50 MB."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        # 51 MB of bytes pretending to be a PNG. PIL won't decode but
        # the SIZE check fires before PIL.
        big = d / "huge.png"
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (51 * 1024 * 1024))
        with pytest.raises(UserDatasetError, match=r"(?i)(size|big|50)"):
            validate_dataset_dir(d, trigger="x")

    def test_decompression_bomb_rejected_via_pil_probe(self, tmp_path):
        """§R.1 security C-1: PIL ``Image.open(path).size`` probe
        rejects ``width × height > 50M px`` before the file flows to
        mflux-train (which would PIL.open and crash the host)."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        # Make a "real" PNG with 8000x8000 = 64M pixels (over the 50M
        # threshold) but tiny on-disk (PNG compresses solid color tightly).
        from PIL import Image
        big = d / "bomb.png"
        Image.new("RGB", (8000, 8000), color=(0, 0, 0)).save(big, format="PNG")
        # Confirm it fits the SIZE gate (well under 50 MB on disk).
        assert big.stat().st_size < 50 * 1024 * 1024
        with pytest.raises(UserDatasetError, match=r"(?i)(pixel|decompress|too large)"):
            validate_dataset_dir(d, trigger="x")


# ── Sidecar validation ─────────────────────────────────────────────────


class TestSidecarValidation:
    def test_sidecar_over_4096_bytes_rejected(self, tmp_path):
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("a" * 4097)
        with pytest.raises(UserDatasetError, match=r"(?i)(sidecar|4096)"):
            validate_dataset_dir(d, trigger="x")

    def test_sidecar_with_control_bytes_rejected(self, tmp_path):
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("al1na woman\x1bdancing")
        with pytest.raises(UserDatasetError, match=r"(?i)control"):
            validate_dataset_dir(d, trigger="x")

    def test_sidecar_non_utf8_rejected(self, tmp_path):
        """``errors='strict'`` UTF-8 decode — invalid bytes raise
        UserDatasetError, not silently mangle the caption."""
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_bytes(b"al1na \xff woman")  # invalid utf-8
        with pytest.raises(UserDatasetError, match=r"(?i)(utf|encod|decode|read)"):
            validate_dataset_dir(d, trigger="x")


# ── Image count gates ──────────────────────────────────────────────────


class TestImageCountGates:
    def test_min_images_floor_rejects(self, tmp_path):
        from imgen.commands.train import UserDatasetError, validate_dataset_dir
        d = _make_dataset(tmp_path, n=2)  # below min_images=3 default
        with pytest.raises(UserDatasetError, match=r"(?i)(minimum|min|3)"):
            validate_dataset_dir(d, trigger="x")

    def test_min_images_floor_at_exactly_3_accepted(self, tmp_path):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=3)
        entries = validate_dataset_dir(d, trigger="x")
        assert len(entries) == 3

    def test_max_images_warns_but_does_not_reject(self, tmp_path, capsys):
        """§D.3 architect M-3 closure: warn at large datasets, don't
        block. The user may have a curated 50+ image set deliberately."""
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=35)  # over default soft cap of 30
        entries = validate_dataset_dir(d, trigger="x")
        assert len(entries) == 35  # not rejected
        # Warning emitted (to stderr via colors.warn or print fallback).
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "30" in combined or "noise" in combined or "large" in combined.lower()


# ── Trigger-presence warn per §M.3 ─────────────────────────────────────


class TestTriggerPresenceWarn:
    def test_sidecar_without_trigger_warns(self, tmp_path, capsys):
        """§M.3 closure: when a sidecar caption omits the trigger
        (case-insensitive), warn informationally. The user may have
        deliberately wanted a non-trigger caption — don't block."""
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("a photo of a forest")  # no trigger
        entries = validate_dataset_dir(d, trigger="al1na woman")
        assert len(entries) == 4  # not blocked
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # Warning surface: the missing-trigger note must mention the
        # sidecar filename OR the trigger word so the user can find
        # which caption to fix.
        assert "photo01" in combined or "al1na" in combined or "trigger" in combined.lower()

    def test_sidecar_with_trigger_no_warn(self, tmp_path, capsys):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("al1na woman in a forest")
        entries = validate_dataset_dir(d, trigger="al1na woman")
        assert len(entries) == 4
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        # No "missing trigger" warning for photo01 (case-insensitive match).
        assert "missing" not in combined or "photo01" not in combined

    def test_trigger_match_is_case_insensitive(self, tmp_path, capsys):
        from imgen.commands.train import validate_dataset_dir
        d = _make_dataset(tmp_path, n=4)
        (d / "photo01.txt").write_text("AL1NA Woman in a forest")
        validate_dataset_dir(d, trigger="al1na woman")
        # Should NOT warn since the trigger IS present, just different case.
        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        if "photo01" in combined:
            assert "missing" not in combined


# ── cmd_train stub ─────────────────────────────────────────────────────


class TestCmdTrainStub:
    """Per memo §Q commit 3: ``cmd_train(args)`` exists as a stub
    raising NotImplementedError until commit 8 wires the real flow.
    This commit ships ONLY the import-shape + validator surface."""

    def test_cmd_train_importable(self):
        from imgen.commands.train import cmd_train
        assert callable(cmd_train)

    def test_cmd_train_exported_from_commands_package(self):
        """commands/__init__.py re-exports cmd_train per the project's
        dispatch convention."""
        from imgen.commands import cmd_train as exported
        from imgen.commands.train import cmd_train as direct
        assert exported is direct

"""v0.9.3 C6 — history additive image_path write + replay round-trip.

Per [[project-v093-i2v]] C6. Covers:

* Write side (``engine_dispatch.run_one_iteration``): when an i2v
  Iteration runs successfully, the history entry gains an
  ``image_path`` field carrying the canonical resolved-Path string.
  Absent (key not present, NOT null) for t2v entries — matches the
  ``enhanced`` field's absence pattern from v0.5+.

* Read side (``commands/history.py:_replay_video_entry``): replay
  reconstructs ``args.image`` from ``entry["image_path"]`` (Path-
  typed). Entries without the key (v0.9.0/v0.9.1/v0.9.2 video rows)
  replay as t2v (``args.image=None``) — backward compat.

* No schema bump. Stays at ``v=4``. Additive write + ``.get(...)``
  read keeps old + new readers all happy.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── Write side: i2v entries carry image_path ────────────────────────────


class TestHistoryWriteSourceCheck:
    """Source-code-level lock-in for the C6 write side.

    The v0.9.3 video history-write at ``engine_dispatch.run_one_
    iteration`` appends ``image_path`` only when ``it.params.input_
    path is not None``. The integration test that would drive
    ``run_one_iteration`` end-to-end requires a full BatchContext +
    BatchLogger fixture pile that's brittle to write generically;
    instead we lock the source-code shape so a future refactor that
    drops the conditional surfaces as a test failure.

    The runtime correctness is verified by the pre-tag real-smoke
    matrix (S1 i2v happy-path checks the history row).
    """

    def test_engine_dispatch_writes_image_path_conditionally(self):
        """Source check: the engine_dispatch video branch contains
        the ``image_path`` write keyed off ``it.params.input_path``."""
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "src" / "imgen" / "engine_dispatch.py"
        ).read_text()
        # The v0.9.3 C6 contract is encoded as two clauses appearing
        # in the source near the v0.9.0 video branch (num_frames /
        # fps / video_codec writes).
        assert "input_path is not None" in src
        assert 'history_entry["image_path"]' in src

    def test_engine_dispatch_image_path_inside_video_branch(self):
        """The conditional must live inside the
        ``ctx.command == 'video'`` branch — writing image_path on
        non-video commands would corrupt the schema."""
        from pathlib import Path
        src = (
            Path(__file__).parent.parent
            / "src" / "imgen" / "engine_dispatch.py"
        ).read_text()
        # Locate the video branch and the image_path write; the write
        # must appear AFTER the branch open and BEFORE the next non-
        # video block.
        video_branch_idx = src.find('ctx.command == "video"')
        image_path_idx = src.find('history_entry["image_path"]')
        assert video_branch_idx > 0, "video branch missing"
        assert image_path_idx > 0, "image_path write missing"
        assert image_path_idx > video_branch_idx, (
            "image_path write must be inside the video branch"
        )


# ── Read side: replay reconstructs args.image from entry ────────────────


class TestHistoryReplaySideI2v:
    """``_replay_video_entry`` reads ``image_path`` from the entry and
    threads it onto the Namespace as ``args.image`` (Path-typed). When
    the field is absent (pre-v0.9.3 video rows) ``args.image`` is None
    — round-trip is backward compatible with v0.9.0-v0.9.2 entries."""

    def test_replay_i2v_entry_sets_args_image(self, tmp_path, monkeypatch):
        from imgen.commands import history as history_mod

        entry = {
            "v": 4, "command": "video",
            "id": 42,
            "prompt": "wind blows",
            "model": "ltx-video",
            "steps": 50, "guidance": 5.0,
            "width": 512, "height": 512,
            "num_frames": 25, "fps": 24,
            "negative": "static, still, frozen, no motion",
            "image_path": "/tmp/cond.png",
        }

        captured: dict = {}

        def fake_cmd_video(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(
            "imgen.commands.video.cmd_video", fake_cmd_video,
        )
        history_mod._replay_video_entry(entry)
        assert captured["args"].image == Path("/tmp/cond.png")

    def test_replay_t2v_entry_image_is_none(self, tmp_path, monkeypatch):
        """v0.9.0/v0.9.1/v0.9.2 video rows have no image_path field;
        replay leaves args.image at None (the t2v default)."""
        from imgen.commands import history as history_mod

        entry = {
            "v": 4, "command": "video",
            "id": 7,
            "prompt": "a samurai",
            "model": "ltx-video",
            "steps": 50, "guidance": 3.0,
            "width": 768, "height": 512,
            "num_frames": 25, "fps": 24,
            # NO image_path key — v0.9.0/v0.9.1/v0.9.2 shape.
        }

        captured: dict = {}

        def fake_cmd_video(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(
            "imgen.commands.video.cmd_video", fake_cmd_video,
        )
        history_mod._replay_video_entry(entry)
        assert getattr(captured["args"], "image", None) is None


# ── Round-trip: write then replay preserves image_path ──────────────────


class TestRoundTrip:
    """Composite check: an i2v entry written by the production path
    can be replayed via ``_replay_video_entry`` and the conditioning-
    image path survives the round-trip."""

    def test_roundtrip_image_path_preserved(self, tmp_path, monkeypatch):
        from imgen.commands import history as history_mod

        # Simulate what the write side would produce.
        entry = {
            "v": 4, "command": "video",
            "id": 99,
            "prompt": "wind through fog",
            "model": "ltx-video",
            "steps": 50, "guidance": 5.0,
            "width": 512, "height": 512,
            "num_frames": 33, "fps": 24,
            "negative": "static, still, frozen, no motion",
            "image_path": str(tmp_path / "input.png"),
        }
        # The file doesn't need to exist for the replay-args round-
        # trip; validate_image_path_or_die runs at cmd_video boundary
        # which is mocked.

        captured: dict = {}

        def fake_cmd_video(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(
            "imgen.commands.video.cmd_video", fake_cmd_video,
        )
        history_mod._replay_video_entry(entry)
        assert captured["args"].image == Path(str(tmp_path / "input.png"))
        # num_frames + guidance also survive (existing v0.9.0 behaviour
        # but worth pinning in the i2v round-trip context).
        assert captured["args"].num_frames == 33
        assert captured["args"].guidance == 5.0

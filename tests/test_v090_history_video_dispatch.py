"""v0.9 commit 5 — history v=4 + additive video fields + replay dispatch.

Per [[project-v090-design]] §J. History schema STAYS at v=4 — v0.9
adds purely additive fields (num_frames, fps, video_codec,
command="video") that v0.8.x readers ignore via the existing
``entry.get(...)`` pattern. No version bump per v0.8.0 schema-version
precedent ("bump on RENAME or DROP, not on additive").

The replay dispatch in commands/history.py gains a video branch +
a control-byte filter on ``entry["command"]`` (security §R.1 NIT-1
— mirrors the v0.8.1 LOW-3 filter pattern applied to entry["backend"]
and entry["model"]).

The actual ``cmd_video`` lands at commit 7; commit 5 ships the
dispatch surface with a lazy import so the module loads cleanly.
Tests inject a fake cmd_video via ``sys.modules`` so the dispatch
path is exercised end-to-end.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from imgen.defaults import HISTORY_SCHEMA_VERSION


def _install_fake_cmd_video(monkeypatch):
    """Install a MagicMock at imgen.commands.video so the lazy import
    inside _replay_video_entry resolves to our fake. Returns the
    fake cmd_video function so tests can inspect its calls."""
    fake_cmd_video = MagicMock(return_value=0)
    fake_module = MagicMock(cmd_video=fake_cmd_video)
    monkeypatch.setitem(sys.modules, "imgen.commands.video", fake_module)
    return fake_cmd_video


def _v4_video_entry(**overrides):
    """Canonical v=4 video history entry — what cmd_video will write
    starting in commit 7."""
    entry = {
        "id": 99,
        "v": 4,
        "ts": "2026-05-26T15:30:00",
        "command": "video",
        "model": "ltx-video",
        "prompt": "a samurai walking through bamboo forest",
        "negative": "",
        "steps": 25,
        "guidance": 3.0,
        "seed": 42,
        "width": 768,
        "height": 512,
        "num_frames": 25,
        "fps": 24,
        "video_codec": "libx264",
        "output": "/Users/x/imgen-output/ltx-smoke.mp4",
        "status": "success",
    }
    entry.update(overrides)
    return entry


# ── Dispatch routing ──────────────────────────────────────────────────


class TestReplayDispatchRouting:
    """replay_entry's command-discriminator dispatch — existing draw/
    refine/generate paths preserved; video joins the matrix."""

    def test_v4_video_entry_routes_to_cmd_video(self, monkeypatch):
        """Lock-in: command=='video' triggers the video replay branch."""
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        rc = replay_entry(_v4_video_entry())
        assert rc == 0
        fake_cmd_video.assert_called_once()

    def test_v4_video_entry_carries_num_frames_through_namespace(self, monkeypatch):
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry(num_frames=49))
        args = fake_cmd_video.call_args[0][0]
        assert args.num_frames == 49

    def test_v4_video_entry_carries_fps_through_namespace(self, monkeypatch):
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry(fps=30))
        args = fake_cmd_video.call_args[0][0]
        assert args.fps == 30

    def test_v4_video_entry_carries_prompt_through_namespace(self, monkeypatch):
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry(prompt="cyberpunk neon city night"))
        args = fake_cmd_video.call_args[0][0]
        assert args.prompt == "cyberpunk neon city night"

    def test_v4_video_entry_carries_model_through_namespace(self, monkeypatch):
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry())
        args = fake_cmd_video.call_args[0][0]
        assert args.model == "ltx-video"

    def test_v4_video_entry_seed_set_to_none_for_fresh_random(self, monkeypatch):
        """Replay always uses a fresh seed — same pattern as draw/refine."""
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry(seed=12345))
        args = fake_cmd_video.call_args[0][0]
        assert args.seed is None


# ── Backwards-compat regression locks ─────────────────────────────────


class TestReplayDispatchBackwardsCompat:
    """v0.9 widening must NOT break existing v=3/v=4 image dispatch."""

    def test_v3_generate_entry_no_command_still_routes_to_cmd_generate(
        self, monkeypatch,
    ):
        """Pre-v0.7 entries lack the command field — replay defaults
        to generate (i2i)."""
        import imgen.commands.history as history_cmd
        captured = []
        monkeypatch.setattr(
            history_cmd, "cmd_generate",
            lambda args: (captured.append(("generate", args)), 0)[1],
        )
        entry = {
            "id": 1,
            "v": 3,
            "input": "/tmp/in.jpg",
            "style": "anime",
            "backend": "flux",
            "quantize": 8,
            "steps": 20,
            "guidance": 3.5,
            "strength": 0.55,
        }
        history_cmd.replay_entry(entry)
        assert len(captured) == 1
        assert captured[0][0] == "generate"

    def test_v4_draw_entry_still_routes_to_cmd_draw(self, monkeypatch):
        import imgen.commands.history as history_cmd
        captured = []
        monkeypatch.setattr(
            history_cmd, "cmd_draw",
            lambda args: (captured.append("draw"), 0)[1],
        )
        entry = {
            "id": 1, "v": 4, "command": "draw",
            "prompt": "a samurai",
            "model": "flux-dev",
            "steps": 20, "guidance": 3.5, "seed": 42,
            "width": 1024, "height": 1024,
            "negative": "",
        }
        history_cmd.replay_entry(entry)
        assert captured == ["draw"]

    def test_v4_refine_entry_still_routes_to_cmd_refine(self, monkeypatch):
        import imgen.commands.history as history_cmd
        captured = []
        monkeypatch.setattr(
            history_cmd, "cmd_refine",
            lambda args: (captured.append("refine"), 0)[1],
        )
        entry = {
            "id": 1, "v": 4, "command": "refine",
            "input": "/tmp/in.jpg",
            "custom_prompt": "refine this",
            "model": "flux2-klein-edit-9b",
            "steps": 20, "guidance": 1.0,
            "strength": 0.3, "seed": 42,
            "width": 2048, "height": 2048,
        }
        history_cmd.replay_entry(entry)
        assert captured == ["refine"]


# ── Security §R.1 NIT-1: control-byte filter on command ───────────────


class TestReplayCommandControlByteFilter:
    """Mirrors the v0.8.1 LOW-3 filter applied to entry['backend']
    and entry['model']. A hand-edited history.jsonl with a control
    byte in ``command`` (C0/DEL/C1) must NOT reach the dispatcher
    — the replay refuses with die() rather than feeding the dirty
    string into argparse / argv."""

    def test_command_with_c0_control_byte_refused(self, monkeypatch):
        from imgen.commands.history import replay_entry
        _install_fake_cmd_video(monkeypatch)
        # Pre-pend an ESC byte to "video" — same shape attacker would
        # use to corrupt the user's terminal via a control sequence
        # echoed in an error message.
        entry = _v4_video_entry(command="\x1b[31mvideo")
        with pytest.raises(SystemExit) as exc_info:
            replay_entry(entry)
        assert exc_info.value.code == 2

    def test_command_with_null_byte_refused(self, monkeypatch):
        from imgen.commands.history import replay_entry
        _install_fake_cmd_video(monkeypatch)
        entry = _v4_video_entry(command="video\x00")
        with pytest.raises(SystemExit) as exc_info:
            replay_entry(entry)
        assert exc_info.value.code == 2

    def test_command_with_del_byte_refused(self, monkeypatch):
        from imgen.commands.history import replay_entry
        _install_fake_cmd_video(monkeypatch)
        entry = _v4_video_entry(command="video\x7f")
        with pytest.raises(SystemExit) as exc_info:
            replay_entry(entry)
        assert exc_info.value.code == 2

    def test_clean_command_string_not_falsely_refused(self, monkeypatch):
        """Sanity check — the filter must NOT reject any of the
        legitimate command values."""
        from imgen.commands.history import replay_entry
        fake_cmd_video = _install_fake_cmd_video(monkeypatch)
        replay_entry(_v4_video_entry())
        # Reached cmd_video → filter passed
        fake_cmd_video.assert_called_once()


# ── Read-compat: v0.8.x reading a v=4 video entry ──────────────────────


class TestReadCompatVideoEntryOnV08:
    """Forward-compat lock: a v=4 video entry (written by v0.9.0) must
    load cleanly via load_history() — additive fields are preserved
    in the returned dict but don't break the schema gate."""

    def test_video_entry_loads_via_load_history(self, monkeypatch):
        from imgen.history import append_history, load_history
        entry = {
            "input": None,
            "output": "/tmp/out.mp4",
            "command": "video",
            "model": "ltx-video",
            "prompt": "a samurai",
            "num_frames": 25,
            "fps": 24,
            "video_codec": "libx264",
        }
        append_history(entry)
        loaded = load_history()
        assert len(loaded) == 1
        assert loaded[0]["command"] == "video"
        assert loaded[0]["num_frames"] == 25
        assert loaded[0]["fps"] == 24
        assert loaded[0]["video_codec"] == "libx264"

    def test_video_entry_v_stays_at_4(self, monkeypatch):
        """§J: history STAYS at v=4. Video fields land additively;
        the schema version is unchanged."""
        from imgen.history import append_history, load_history
        append_history({
            "input": None, "output": "/tmp/out.mp4",
            "command": "video", "num_frames": 25, "fps": 24,
        })
        loaded = load_history()
        assert loaded[0]["v"] == 4
        assert HISTORY_SCHEMA_VERSION == 4, (
            "HISTORY_SCHEMA_VERSION must stay at 4 across v0.9 — "
            "additive-only video fields don't trigger a bump"
        )

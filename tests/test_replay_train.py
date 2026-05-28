"""v0.10.0 commit 11 — replay-CONFIRM-GATE for train entries + B-19.

Two surfaces:

1. ``_replay_train_entry`` — the replay-CONFIRM-GATE handler. Unlike
   draw/refine/video (immediate re-execute), a train replay prints the
   equivalent command, asks for confirmation (~10h job), and only
   then re-validates + dispatches to ``cmd_train``.

2. B-19 (architect H-1) — replay-dispatch registration-completeness:
   every command that can appear in a history entry must be handled
   (in ``_REPLAY_DISPATCH`` or a known fall-through), so a future
   command can't silently mis-route through the i2i generate path.

Per [[project-v100-design]] §J.2 + §M.11 + §R.1 ROUND-1 CLOSURES.
"""
from __future__ import annotations

import argparse

import pytest


def _train_entry(**overrides) -> dict:
    base = {
        "v": 4,
        "id": 142,
        "ts": "2026-05-28T03:14:15",
        "command": "train",
        "model": "flux2-klein-4b",
        "lora_name": "alina",
        "trigger": "al1na woman",
        "dataset_path": "/Users/me/.imgen/datasets/alina",
        "dataset_image_count": 10,
        "total_steps": 880,
        "lora_rank": 16,
        "quantize": 4,
        "max_resolution": 512,
        "seed": 42,
        "output": "/Users/me/.imgen/loras/alina.safetensors",
        "status": "success",
        "wall_seconds": 36000,
    }
    base.update(overrides)
    return base


@pytest.fixture
def captured_cmd_train(monkeypatch):
    """Patch cmd_train (as imported INTO history module's handler) so
    the replay dispatch is observable without spawning training."""
    calls = {"count": 0, "args": None}

    def fake_cmd_train(args):
        calls["count"] += 1
        calls["args"] = args
        return 0

    # _replay_train_entry does `from .train import cmd_train` lazily,
    # so patch the source module attribute.
    from imgen.commands import train as train_module
    monkeypatch.setattr(train_module, "cmd_train", fake_cmd_train)
    return calls


@pytest.fixture
def patch_prompt(monkeypatch):
    state = {"answer": False, "calls": 0}
    from imgen import cmd_helpers

    def fake(question="? "):
        state["calls"] += 1
        return state["answer"]

    monkeypatch.setattr(cmd_helpers, "prompt_yes_no", fake)
    return state


# ── decline path ─────────────────────────────────────────────────


class TestReplayTrainDecline:
    def test_decline_returns_zero_no_dispatch(
        self, captured_cmd_train, patch_prompt, capsys,
    ):
        patch_prompt["answer"] = False
        from imgen.commands.history import _replay_train_entry
        rc = _replay_train_entry(_train_entry())
        assert rc == 0
        assert captured_cmd_train["count"] == 0

    def test_decline_prints_equivalent_command(
        self, captured_cmd_train, patch_prompt, capsys,
    ):
        patch_prompt["answer"] = False
        from imgen.commands.history import _replay_train_entry
        _replay_train_entry(_train_entry())
        out = capsys.readouterr().out
        assert "imgen train" in out
        assert "--dataset" in out
        assert "--name" in out
        assert "--trigger" in out


# ── accept path → dispatch ───────────────────────────────────────


class TestReplayTrainAccept:
    def test_accept_dispatches_to_cmd_train(
        self, captured_cmd_train, patch_prompt,
    ):
        patch_prompt["answer"] = True
        from imgen.commands.history import _replay_train_entry
        rc = _replay_train_entry(_train_entry())
        assert rc == 0
        assert captured_cmd_train["count"] == 1

    def test_dispatched_args_carry_yes_and_overwrite(
        self, captured_cmd_train, patch_prompt,
    ):
        """The replay gate IS the confirmation → cmd_train gets yes=True
        (no double-prompt) and overwrite=True (re-run replaces)."""
        patch_prompt["answer"] = True
        from imgen.commands.history import _replay_train_entry
        _replay_train_entry(_train_entry())
        args = captured_cmd_train["args"]
        assert args.yes is True
        assert args.overwrite is True

    def test_dispatched_args_round_trip_training_params(
        self, captured_cmd_train, patch_prompt,
    ):
        patch_prompt["answer"] = True
        from imgen.commands.history import _replay_train_entry
        _replay_train_entry(_train_entry())
        args = captured_cmd_train["args"]
        assert args.name == "alina"
        assert args.trigger == "al1na woman"
        assert args.dataset == "/Users/me/.imgen/datasets/alina"
        assert args.base == "flux2-klein-4b"
        assert args.steps == 880
        assert args.rank == 16
        assert args.quantize == 4
        assert args.max_resolution == 512
        assert args.seed == 42


# ── §M.11 N-2 — re-validation on READ ────────────────────────────


class TestReplayTrainRevalidatesDirtyMeta:
    def test_rejects_dirty_lora_name_via_argparse_revalidate(
        self, captured_cmd_train, patch_prompt,
    ):
        """A hand-edited history.jsonl with a traversal-shaped
        lora_name must be re-rejected by _lora_name_arg BEFORE
        reaching cmd_train (which would build a filesystem path from
        it)."""
        patch_prompt["answer"] = True
        from imgen.commands.history import _replay_train_entry
        with pytest.raises(SystemExit) as exc:
            _replay_train_entry(_train_entry(lora_name="../../etc/passwd"))
        assert exc.value.code == 2
        assert captured_cmd_train["count"] == 0

    def test_rejects_dirty_trigger_via_argparse_revalidate(
        self, captured_cmd_train, patch_prompt,
    ):
        """A control-byte-bearing trigger from a tampered entry must
        be re-rejected by _trigger_token_arg."""
        patch_prompt["answer"] = True
        from imgen.commands.history import _replay_train_entry
        with pytest.raises(SystemExit) as exc:
            _replay_train_entry(_train_entry(trigger="al1na\x1b[2J"))
        assert exc.value.code == 2
        assert captured_cmd_train["count"] == 0

    def test_revalidation_happens_only_after_confirm(
        self, captured_cmd_train, patch_prompt,
    ):
        """If the user declines, a dirty entry should NOT die — the
        command is just printed (the user never asked to run it)."""
        patch_prompt["answer"] = False
        from imgen.commands.history import _replay_train_entry
        # Dirty name, but declined → no SystemExit, returns 0.
        rc = _replay_train_entry(_train_entry(lora_name="../evil"))
        assert rc == 0


# ── registered in dispatch table ─────────────────────────────────


class TestReplayTrainRegistered:
    def test_train_in_replay_dispatch(self):
        from imgen.commands.history import (
            _REPLAY_DISPATCH,
            _replay_train_entry,
        )
        assert _REPLAY_DISPATCH.get("train") is _replay_train_entry

    def test_train_entry_routes_through_dispatch(
        self, captured_cmd_train, patch_prompt,
    ):
        """replay_entry() must route a command=train entry to the
        train handler (not the i2i generate fall-through)."""
        patch_prompt["answer"] = False
        from imgen.commands.history import replay_entry
        rc = replay_entry(_train_entry())
        assert rc == 0  # declined; but it reached the train handler
        # (the i2i fall-through would have die'd on missing "input").


# ── B-19 — replay dispatch registration completeness ─────────────


class TestB19DispatchCompleteness:
    """architect H-1 (folds into commit 11): every command that can
    appear in a history entry must be explicitly handled in replay —
    either registered in _REPLAY_DISPATCH or a known i2i fall-through
    (generate / batch). Guards against a future BatchContext.command
    Literal extension silently mis-routing through cmd_generate."""

    def test_every_batchcontext_command_is_handled(self):
        import typing

        from imgen.commands.history import _REPLAY_DISPATCH
        from imgen.runs import BatchContext

        # Extract the Literal members of BatchContext.command.
        hints = typing.get_type_hints(BatchContext)
        literal_args = set(typing.get_args(hints["command"]))
        assert literal_args, "BatchContext.command should be a Literal"

        # Known i2i fall-throughs (handled at the bottom of
        # replay_entry, NOT in the dispatch table — documented there).
        fall_through = {"generate", "batch"}

        for cmd in literal_args:
            handled = cmd in _REPLAY_DISPATCH or cmd in fall_through
            assert handled, (
                f"BatchContext.command Literal member {cmd!r} is neither "
                f"in _REPLAY_DISPATCH nor a known fall-through "
                f"({fall_through}). A new command must register a replay "
                f"handler so it can't silently mis-route through "
                f"cmd_generate."
            )

    def test_train_handled_even_though_outside_batchcontext_literal(self):
        """``train`` writes history entries (via _append_train_history)
        but does NOT go through BatchContext (standalone pipeline, §B.5).
        So it won't appear in the Literal — assert it's covered in the
        dispatch table explicitly."""
        from imgen.commands.history import _REPLAY_DISPATCH
        assert "train" in _REPLAY_DISPATCH

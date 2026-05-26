"""History persistence + schema-version refuse-on-future.

Uses the tmp_state_dir fixture so HISTORY_FILE is per-test, no real
~/.imgen/history.jsonl is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from imgen.defaults import HISTORY_SCHEMA_VERSION
from imgen.history import append_history, load_history


def test_load_history_empty_returns_empty_list(tmp_state_dir):
    assert load_history() == []


def test_append_history_assigns_v_and_monotonic_id(tmp_state_dir):
    id1 = append_history({"input": "/a.jpg", "output": "/a.png"})
    id2 = append_history({"input": "/b.jpg", "output": "/b.png"})
    id3 = append_history({"input": "/c.jpg", "output": "/c.png"})
    assert (id1, id2, id3) == (1, 2, 3)


def test_append_history_sets_schema_version(tmp_state_dir):
    """The STORED record (not the caller's dict) gets `v` stamped. The
    earlier version of this test asserted mutation on the caller dict —
    that was a side-effect bug; the test now reads back via load_history
    and verifies the persisted record has the schema version."""
    append_history({"input": "/a.jpg", "output": "/a.png"})
    [entry] = load_history()
    assert entry["v"] == HISTORY_SCHEMA_VERSION


def test_load_history_roundtrips_appended_entries(tmp_state_dir):
    append_history({"input": "/a.jpg", "output": "/a.png", "style": "anime"})
    append_history({"input": "/b.jpg", "output": "/b.png", "style": "pixar"})
    entries = load_history()
    assert len(entries) == 2
    assert entries[0]["style"] == "anime"
    assert entries[1]["style"] == "pixar"
    # ids reassigned in order, v stamped
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2
    assert entries[0]["v"] == HISTORY_SCHEMA_VERSION


def test_load_history_tolerates_corrupted_lines(tmp_state_dir):
    # Simulate a partial write / disk corruption mid-line
    from imgen.history import HISTORY_FILE
    HISTORY_FILE.write_text(
        json.dumps({"id": 1, "input": "/a.jpg"}) + "\n"
        + "{not json at all\n"
        + json.dumps({"id": 2, "input": "/b.jpg"}) + "\n",
    )
    entries = load_history()
    assert len(entries) == 2  # corrupted line dropped, others survive
    assert entries[0]["id"] == 1
    assert entries[1]["id"] == 2


def test_load_history_warns_on_corrupted_line(tmp_state_dir, capsys):
    """Silent `pass` on JSONDecodeError loses user data with no feedback.
    A warn surfaces the loss so the user knows. (python-reviewer I3)"""
    from imgen.history import HISTORY_FILE
    HISTORY_FILE.write_text(
        json.dumps({"id": 1, "input": "/a.jpg"}) + "\n"
        + "{broken\n"
    )
    load_history()
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "skip" in combined.lower() or "malformed" in combined.lower()


def test_append_history_does_not_mutate_caller_dict(tmp_state_dir):
    """`entry["id"] = ...` was being written into the caller's dict
    as a hidden side-effect. (python-reviewer C1)"""
    original = {"input": "/a.jpg", "output": "/a.png"}
    snapshot = dict(original)
    append_history(original)
    # Caller's dict must be untouched
    assert original == snapshot


def test_append_history_resets_mode_on_existing_world_readable_file(tmp_state_dir):
    """If history.jsonl pre-existed at 0o644 (e.g. v0.1.0 install before
    the v0.1.1 chmod fix), os.open(O_CREAT, 0o600) ignores mode on existing
    files. Must explicitly fchmod under the lock. (security I3)"""
    import os as _os
    from imgen.history import HISTORY_FILE
    # Pre-create with permissive mode
    HISTORY_FILE.write_text("")
    HISTORY_FILE.chmod(0o644)
    assert (HISTORY_FILE.stat().st_mode & 0o777) == 0o644

    append_history({"input": "/a.jpg", "output": "/a.png"})

    mode = HISTORY_FILE.stat().st_mode & 0o777
    assert mode == 0o600, f"history.jsonl mode {oct(mode)} after append, want 0o600"


def test_replay_entry_refuses_future_schema(tmp_state_dir, capsys):
    """Entry written by a newer install must refuse replay rather than
    silently producing weird output. Pins architect #3."""
    from imgen.commands.history import replay_entry
    entry = {
        "id": 99,
        "v": HISTORY_SCHEMA_VERSION + 1,
        "input": "/some.jpg",
    }
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "newer schema" in stderr  # the gate's specific message
    assert "imgen upgrade" in stderr  # the user-facing hint


def test_replay_entry_missing_input_fails_cleanly(tmp_state_dir):
    from imgen.commands.history import replay_entry
    entry = {"id": 99, "v": HISTORY_SCHEMA_VERSION}  # no "input"
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 1


def test_replay_entry_legacy_v0_entries_pass_schema_gate(tmp_state_dir, capsys):
    """Pre-v0.2 history.jsonl entries lack the 'v' field. Default-to-0
    treatment means they pass the schema gate (don't refuse-on-future).
    Pins architect #3 'best-effort .get' contract."""
    from imgen.commands.history import replay_entry
    legacy_entry = {"id": 1, "input": "/nonexistent_xyz.jpg"}  # no 'v' key
    with pytest.raises(SystemExit):
        replay_entry(legacy_entry)
    stderr = capsys.readouterr().err
    # The future-schema gate would say "newer schema"; legacy entries
    # must NOT trigger it (they pass through to cmd_generate which then
    # dies on the missing input file — that's an unrelated failure).
    assert "newer schema" not in stderr


def test_replay_entry_namespace_has_explicit_v021_fields(tmp_state_dir, monkeypatch):
    """architect #7: replay_entry constructs an argparse.Namespace that
    cmd_generate reads. v0.2 introduced --prompt-file and the
    imgen_merged_defaults stash, both consumed via getattr-with-default
    today. Pin the explicit fields so a future required attribute fails
    loudly instead of silently."""
    import imgen.commands.history as history_cmd
    captured_args = {}

    def fake_cmd_generate(args):
        # Snapshot args so the test can inspect — return 0 (success)
        captured_args["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    entry = {
        "id": 1,
        "v": HISTORY_SCHEMA_VERSION,
        "input": "/some.jpg",
        "style": "anime",
        "backend": "flux",
        "quantize": 8,
        "steps": 20,
        "guidance": 3.5,
        "strength": 0.55,
    }
    history_cmd.replay_entry(entry)
    args = captured_args["args"]
    assert hasattr(args, "prompt_file"), "replay Namespace missing prompt_file"
    assert args.prompt_file is None
    assert hasattr(args, "imgen_merged_defaults"), \
        "replay Namespace missing imgen_merged_defaults"


def test_replay_entry_namespace_has_explicit_v05_enhance_fields(
    tmp_state_dir, monkeypatch,
):
    """v0.5 added args.enhance / args.enhance_model / args.enhance_temperature
    / args.imgen_config_enhance to the cmd_generate surface. Replay must
    pin those explicitly in _REPLAY_DEFAULTS so the policy "replay never
    auto-enhances" is loud (AttributeError on a missing field) rather
    than silent (getattr-with-default flow). Closes architect CRITICAL
    #1 from the v0.5 pre-tag review."""
    import imgen.commands.history as history_cmd
    captured_args = {}

    def fake_cmd_generate(args):
        captured_args["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    # A v=2 entry written by an --enhance-prompt run. Replay reconstructs
    # args from the entry's stored fields; the enhance_* args should be
    # forced to "off" regardless of the entry's enhanced=True history.
    entry = {
        "id": 1,
        "v": 2,
        "input": "/some.jpg",
        "style": "anime",
        "backend": "flux",
        "quantize": 8, "steps": 20, "guidance": 3.5, "strength": 0.55,
        # v0.5 fields on the entry — replay sees these but ignores
        # for purposes of re-enhancing.
        "enhanced": True,
        "enhance_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "enhance_fallback_reason": None,
        "prompt_original": "raw pre-LLM prompt",
        "prompt": "raw POST-LLM enhanced prompt",
    }
    history_cmd.replay_entry(entry)
    args = captured_args["args"]

    # All four v0.5 fields present on the Namespace.
    assert hasattr(args, "enhance"), "replay Namespace missing enhance"
    assert args.enhance is False, (
        "replay must NOT auto-enhance — would surprise users and pay "
        "a 4 GB download + 5 s inference cost they didn't ask for"
    )
    assert hasattr(args, "enhance_model")
    assert args.enhance_model is None
    assert hasattr(args, "enhance_temperature")
    assert args.enhance_temperature is None
    assert hasattr(args, "imgen_config_enhance")
    assert args.imgen_config_enhance == {}


# ── v=3 LoRA rehydration on replay (v0.6 — Architect CRITICAL #1) ──────


def test_replay_entry_v3_rehydrates_stored_loras(tmp_state_dir, monkeypatch):
    """A v=3 entry carrying a non-empty ``loras`` list must rehydrate
    into ``args.lora`` (list[LoraRef]) + ``args.no_lora=True`` so the
    style's CURRENT built-in LoRAs don't sneak in alongside the stored
    snapshot. Architect-CRITICAL #1 fix from the v0.6 pre-tag review.

    Without this rehydration, ``imgen replay <id>`` silently diverged
    on LoRA selection: a generation originally run with ``--lora REF``
    replayed WITHOUT the LoRA, and ``--no-lora`` original runs replayed
    WITH the style's new built-in LoRA re-injected — both broke replay
    determinism."""
    import imgen.commands.history as history_cmd
    from imgen.styles import LoraRef

    captured = {}

    def fake_cmd_generate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    entry = {
        "id": 7, "v": 3,
        "input": "/photo.jpg",
        "style": "anime",
        "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 3.5, "strength": 0.55,
        "loras": [
            {
                "ref": "strangerzonehf/Flux-Animeo-v1-LoRA",
                "weight": 0.8,
                "compatible_with": ["flux-1"],
                "trigger": "Animeo",
            },
        ],
    }
    history_cmd.replay_entry(entry)
    args = captured["args"]

    # args.lora reconstructed as a list of LoraRef matching the stored shape.
    assert args.lora == [LoraRef(
        ref="strangerzonehf/Flux-Animeo-v1-LoRA",
        weight=0.8,
        compatible_with=("flux-1",),
        trigger="Animeo",
    )]
    # args.no_lora=True suppresses the style's CURRENT built-ins so
    # only the stored stack is used (resolve_effective_loras carve-out
    # keeps cli_lora when no_lora=True).
    assert args.no_lora is True


def test_replay_entry_v3_text_only_run_rehydrates_as_no_lora(
    tmp_state_dir, monkeypatch,
):
    """A v=3 entry with ``loras=[]`` records a text-only original run
    (either a style without built-in LoRAs OR an explicit --no-lora
    invocation). Replay must reproduce the text-only behaviour even if
    the style now ships built-in LoRAs."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_generate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    entry = {
        "id": 8, "v": 3,
        "input": "/photo.jpg",
        "style": "simpsons",
        "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 4.5, "strength": 0.65,
        "loras": [],  # text-only original run
    }
    history_cmd.replay_entry(entry)
    args = captured["args"]

    # No CLI LoRAs to inject + no_lora=True → resolve_effective_loras
    # returns () via the v0.5 path (carve-out only kicks in when
    # cli_lora is non-empty).
    assert args.lora is None
    assert args.no_lora is True


def test_replay_entry_pre_v3_falls_back_to_current_style_loras(
    tmp_state_dir, monkeypatch,
):
    """v=1 and v=2 entries pre-date the LoRA persistence. Replay must
    not crash on them — instead use the style's current LoRA stack
    (best-effort fallback, matches v0.5 behaviour). args.lora=None
    + args.no_lora=False reproduces "no CLI override, no opt-out"."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_generate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    entry = {
        "id": 9, "v": 2,
        "input": "/photo.jpg",
        "style": "anime",
        "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 4.0, "strength": 0.6,
        # NO loras field — pre-v0.6 schema.
    }
    history_cmd.replay_entry(entry)
    args = captured["args"]

    assert args.lora is None
    assert args.no_lora is False


def test_replay_entry_v3_malformed_loras_field_gracefully_falls_back(
    tmp_state_dir, monkeypatch,
):
    """Defensive: a hand-edited history.jsonl with a typo'd loras
    shape (not a list, or list with non-dict entries) must NOT crash
    replay. Falls back to "no LoRA info" so the generation still runs."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_generate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    # Loras is a string instead of a list — corruption / hand-edit.
    entry = {
        "id": 10, "v": 3,
        "input": "/photo.jpg",
        "style": "anime",
        "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 4.0, "strength": 0.6,
        "loras": "not-a-list",
    }
    history_cmd.replay_entry(entry)
    args = captured["args"]

    # Falls back to the pre-v3 behaviour: no override, no opt-out.
    assert args.lora is None
    assert args.no_lora is False


def test_history_entry_persists_loras_field(tmp_state_dir):
    """End-to-end persistence: writing an entry with a loras list
    round-trips through history.jsonl with shape intact (list of dicts,
    each dict carrying ref/weight/compatible_with/trigger)."""
    append_history({
        "input": "/p.jpg",
        "output": "/o.png",
        "prompt": "Restyle this person as anime",
        "backend": "flux",
        "loras": [
            {
                "ref": "strangerzonehf/Flux-Animeo-v1-LoRA",
                "weight": 0.8,
                "compatible_with": ["flux-1"],
                "trigger": "Animeo",
            },
        ],
    })
    entries = load_history()
    assert len(entries) == 1
    assert entries[0]["v"] == HISTORY_SCHEMA_VERSION
    assert entries[0]["loras"] == [
        {
            "ref": "strangerzonehf/Flux-Animeo-v1-LoRA",
            "weight": 0.8,
            "compatible_with": ["flux-1"],
            "trigger": "Animeo",
        },
    ]


# ── v=2 schema migration (v0.5 — LLM prompt enhancer) ──────────────────


def test_history_schema_version_is_4(tmp_state_dir):
    """v0.8.0 commit 9 (§K + §Q) bumps the schema to 4 for the
    ``backend`` → ``model`` key rename. Dual-shape read dispatch in
    ``history.entry_model_name`` keeps v=3 rows on disk readable.
    Lock-in against accidental downgrade in a future commit."""
    assert HISTORY_SCHEMA_VERSION == 4


def test_v1_entries_still_pass_replay_schema_gate(tmp_state_dir, monkeypatch):
    """A history.jsonl row written by v0.4.x carries v=1 and no
    enhance_* / loras fields. Replay must NOT refuse it as "newer
    schema" — 1 < HISTORY_SCHEMA_VERSION, treat as if enhancement was
    off and LoRA info absent (fall back to current style's LoRAs)."""
    import imgen.commands.history as history_cmd

    captured = {}

    def fake_cmd_generate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    v1_entry = {
        "id": 42, "v": 1,
        "input": "/photo.jpg",
        "style": "anime",
        "prompt": "Restyle this person as anime while preserving identity",
        "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 3.5, "strength": 0.55,
    }
    # Must not raise — schema gate passes (1 <= HISTORY_SCHEMA_VERSION=2).
    history_cmd.replay_entry(v1_entry)
    assert "args" in captured


def test_v2_entry_with_enhance_fields_roundtrips(tmp_state_dir):
    """Write a v=2 entry carrying the new enhance_* fields, read it
    back via load_history(), verify every new field survives the
    JSON round-trip with type intact (bool / str / None)."""
    append_history({
        "input": "/p.jpg",
        "output": "/o.png",
        "prompt": (
            "Restyle this person as cel-shaded anime, vibrant studio "
            "colors, while preserving facial identity"
        ),
        "prompt_original": (
            "Restyle this person as anime while preserving identity"
        ),
        "enhanced": True,
        "enhance_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "enhance_fallback_reason": None,
        "backend": "flux",
    })
    entries = load_history()
    assert len(entries) == 1
    e = entries[0]
    assert e["v"] == HISTORY_SCHEMA_VERSION
    assert e["enhanced"] is True
    assert e["enhance_model"] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert e["enhance_fallback_reason"] is None
    assert "cel-shaded" in e["prompt"]
    assert "cel-shaded" not in e["prompt_original"]


def test_v2_entry_with_fallback_records_reason(tmp_state_dir):
    """When the LLM fell back (empty output / invariant violated /
    runner crashed), the entry records `enhanced=False` plus the
    diagnostic reason. ``prompt`` equals ``prompt_original``."""
    raw = (
        "Restyle this person as anime while preserving identity"
    )
    append_history({
        "input": "/p.jpg",
        "output": "/o.png",
        "prompt": raw,
        "prompt_original": raw,
        "enhanced": False,
        "enhance_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "enhance_fallback_reason": "invariant_violated",
        "backend": "flux",
    })
    entries = load_history()
    assert entries[0]["enhanced"] is False
    assert entries[0]["enhance_fallback_reason"] == "invariant_violated"
    assert entries[0]["prompt"] == entries[0]["prompt_original"]


def test_v_current_entry_without_enhance_fields_is_legal(tmp_state_dir):
    """When --enhance-prompt is OFF (default), the entry doesn't write
    enhance_* fields at all — keeps the per-row JSON terse and matches
    "no LLM was involved" semantics. Schema stamping is unconditional
    (every new entry gets the current schema version)."""
    append_history({
        "input": "/p.jpg",
        "output": "/o.png",
        "prompt": "Restyle this person as anime",
        "backend": "flux",
    })
    entries = load_history()
    assert entries[0]["v"] == HISTORY_SCHEMA_VERSION
    # Absence of the enhance_* keys is the "enhancer was off" signal.
    assert "enhanced" not in entries[0]
    assert "enhance_model" not in entries[0]
    assert "prompt_original" not in entries[0]


# ── v0.7.0: draw command discriminator + replay routing ──────────────


def test_replay_entry_routes_draw_to_cmd_draw(tmp_state_dir, monkeypatch):
    """v0.7.0 (architect §J + pre-tag CRITICAL #1): replay_entry must
    inspect entry.get("command") and route accordingly. A draw entry
    has input=null and goes to cmd_draw — not cmd_generate, which would
    die at _validate_input_path."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_draw(args):
        captured["args"] = args
        return 0

    def fake_cmd_generate(args):
        captured["generate_called"] = True
        return 0

    monkeypatch.setattr(history_cmd, "cmd_draw", fake_cmd_draw)
    monkeypatch.setattr(history_cmd, "cmd_generate", fake_cmd_generate)

    entry = {
        "id": 42,
        "v": HISTORY_SCHEMA_VERSION,
        "input": None,                # v0.7.0 nullable for draw
        "command": "draw",
        "prompt": "a samurai on a misty mountain",
        "backend": "flux-dev",
        "quantize": 8,
        "steps": 20,
        "guidance": 3.5,
        "width": 1024,
        "height": 1024,
    }
    rc = history_cmd.replay_entry(entry)
    assert rc == 0
    # cmd_draw was called, cmd_generate was NOT.
    assert "generate_called" not in captured
    args = captured["args"]
    assert args.prompt == "a samurai on a misty mountain"
    assert args.model == "flux-dev"
    assert args.width == 1024
    assert args.height == 1024


def test_replay_draw_entry_round_trips_negative_prompt(
    tmp_state_dir, monkeypatch,
):
    """v0.8.2 M-2 closure: ``_replay_draw_entry`` must round-trip the
    ``"negative"`` field from the history entry into ``args.negative_prompt``
    on the Namespace it builds for cmd_draw. Pre-fix the field was
    silently dropped — ``build_draw_iterations`` resolved it via
    ``getattr(args, "negative_prompt", None) or ""`` so replays of
    entries with a negative produced argv without it, bit-divergent
    from the original run. Lock-in against regression."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_draw(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_draw", fake_cmd_draw)

    entry = {
        "id": 43,
        "v": HISTORY_SCHEMA_VERSION,
        "input": None,
        "command": "draw",
        "prompt": "samurai",
        "negative": "blurry, low quality, jpeg artifacts",
        "model": "flux-dev",
        "quantize": 8,
        "steps": 20,
        "guidance": 3.5,
        "width": 1024,
        "height": 1024,
    }
    rc = history_cmd.replay_entry(entry)
    assert rc == 0
    args = captured["args"]
    assert args.negative_prompt == "blurry, low quality, jpeg artifacts"


def test_replay_draw_entry_missing_negative_yields_none(
    tmp_state_dir, monkeypatch,
):
    """v0.8.2 M-2 boundary case: an entry with no ``"negative"`` field
    (pre-v0.7.11 entries) round-trips to ``args.negative_prompt = None``,
    not an empty string. Empty-string vs None matters downstream in
    build_draw_iterations (`getattr or ""` collapses both, but other
    surfaces — replay history list, future replay diff — may treat
    them differently). Explicit None preserves the "no negative was
    set" semantic."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_draw(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(history_cmd, "cmd_draw", fake_cmd_draw)

    entry = {
        "id": 44,
        "v": HISTORY_SCHEMA_VERSION,
        "input": None,
        "command": "draw",
        "prompt": "samurai",
        # no "negative" key — pre-v0.7.11 shape
        "model": "flux-dev",
        "quantize": 8,
        "steps": 20,
        "guidance": 3.5,
        "width": 1024,
        "height": 1024,
    }
    rc = history_cmd.replay_entry(entry)
    assert rc == 0
    args = captured["args"]
    assert args.negative_prompt is None


def test_replay_entry_routes_refine_to_cmd_refine(
    tmp_state_dir, monkeypatch,
):
    """v0.7.5 (architect IMPORTANT #A): replay_entry must route
    command="refine" entries to cmd_refine, not cmd_generate. Without
    this dispatch the entry's non-null input + custom_prompt would
    pass the i2i guard and mis-replay as a Kontext restyle."""
    import imgen.commands.history as history_cmd
    captured = {}

    def fake_cmd_refine(args):
        captured["args"] = args
        captured["ran"] = "refine"
        return 0

    monkeypatch.setattr(history_cmd, "cmd_refine", fake_cmd_refine)
    monkeypatch.setattr(
        history_cmd, "cmd_generate",
        lambda args: captured.setdefault("ran", "generate") or 0,
    )
    monkeypatch.setattr(
        history_cmd, "cmd_draw",
        lambda args: captured.setdefault("ran", "draw") or 0,
    )

    entry = {
        "id": 77,
        "v": HISTORY_SCHEMA_VERSION,
        "input": "/some/winner.png",
        "command": "refine",
        "custom_prompt": "Same scene. Refine sharper detail.",
        "style": None,
        "prompt": "Same scene. Refine sharper detail.",
        "backend": "flux2-klein-edit-9b",
        "quantize": 4,
        "steps": 20,
        "guidance": 3.5,
        "strength": 0.3,
        "width": 1536,
        "height": 1536,
    }
    rc = history_cmd.replay_entry(entry)
    assert rc == 0
    assert captured["ran"] == "refine"
    args = captured["args"]
    assert args.input == "/some/winner.png"
    assert args.model == "flux2-klein-edit-9b"
    assert args.width == 1536
    assert args.height == 1536
    assert args.quantize == 4
    assert args.strength == 0.3
    assert args.prompt == "Same scene. Refine sharper detail."
    # --scale is None when --width/--height carry the dims (mutex
    # enforced by _resolve_target_dimensions, and replay always uses
    # explicit dims for bit-stable round-trip).
    assert args.scale is None
    # New random seed each replay (not the stored one).
    assert args.seed is None


def test_replay_entry_refine_missing_input_fails_cleanly(tmp_state_dir):
    """A malformed refine entry with command="refine" but no input dies
    with exit 1 and a clear message — same loud-fail discipline as
    the i2i "no input path" guard."""
    from imgen.commands.history import replay_entry
    entry = {
        "id": 78, "v": HISTORY_SCHEMA_VERSION,
        "command": "refine",
        "input": None,
    }
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 1


def test_replay_entry_generate_command_uses_cmd_generate(
    tmp_state_dir, monkeypatch,
):
    """v0.7.0: existing i2i entries (no command field, or
    command="generate") route to cmd_generate unchanged. Backward-
    compat lock-in."""
    import imgen.commands.history as history_cmd
    captured = {}

    monkeypatch.setattr(
        history_cmd, "cmd_generate",
        lambda args: captured.setdefault("ran", "generate") or 0,
    )
    monkeypatch.setattr(
        history_cmd, "cmd_draw",
        lambda args: captured.setdefault("ran", "draw") or 0,
    )

    entry = {
        "id": 1,
        "v": HISTORY_SCHEMA_VERSION,
        "input": "/some.jpg",
        "style": "anime",
        "backend": "flux",
        "quantize": 8,
        "steps": 20,
        "guidance": 3.5,
        "strength": 0.55,
    }
    history_cmd.replay_entry(entry)
    # No command field present → routes through cmd_generate
    # (the backward-compat default).
    assert captured["ran"] == "generate"


def test_replay_entry_draw_missing_prompt_fails_cleanly(tmp_state_dir):
    """A malformed draw entry with command="draw" but no prompt dies
    with exit 1 and a clear message — same loud-fail discipline as
    the i2i "no input path" guard."""
    from imgen.commands.history import replay_entry
    entry = {
        "id": 99, "v": HISTORY_SCHEMA_VERSION,
        "command": "draw",
        "input": None,
        # NO prompt field.
    }
    with pytest.raises(SystemExit) as exc_info:
        replay_entry(entry)
    assert exc_info.value.code == 1


def test_history_entry_carries_command_field_for_draw(
    tmp_state_dir, monkeypatch, tmp_path,
):
    """End-to-end: a draw run writes "command": "draw" + "input": null
    to history.jsonl. Locks the v0.7.0 schema additive fields at the
    producer side (run_one_iteration via cmd_draw)."""
    from imgen.backends import BACKENDS
    from imgen.commands.draw import cmd_draw
    from types import SimpleNamespace
    from imgen.defaults import DEFAULTS

    def fake_load(args):
        return ("flux-dev", BACKENDS["flux-dev"], "tok",
                Path("/fake/mflux-generate"), None)
    monkeypatch.setattr(
        "imgen.commands.draw.load_backend_and_token", fake_load,
    )
    # Stub the actual mflux subprocess + battery/disk preflight.
    def fake_run_subprocess(*args, **kwargs):
        # Touch the output file so success path passes.
        for arg in args[0]:
            pass
        return 0
    _touch_and_zero = lambda cmd, *a, **kw: (
        Path(cmd[cmd.index("--output") + 1]).touch(),
        0,
    )[1]
    # v0.8.3 M-NEW-C: single-patch — Engine.run path only.
    monkeypatch.setattr(
        "imgen.subprocess_helpers.run_with_stderr_redaction", _touch_and_zero,
    )
    monkeypatch.setattr(
        "imgen.cmd_helpers.preflight_resources",
        lambda **kw: None,
    )
    args = SimpleNamespace(
        prompt="a draw test prompt",
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        seed=42,
        model="flux-dev",
        preview=False,
        width=1024,
        height=1024,
        no_open=True,
        yes=True,
        dry_run=False,
        force=True,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
        output=None,
        output_dir=str(tmp_path),
        lora=None,
        no_lora=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    rc = cmd_draw(args)
    assert rc == 0
    entries = load_history()
    assert len(entries) >= 1
    e = entries[-1]
    assert e["command"] == "draw"
    assert e["input"] is None
    assert e["prompt"] == "a draw test prompt"
    # Generate entries would have "input" as a path string; locking
    # the new nullable shape.


def test_history_n_iterations_records_per_row_seed_ladder(
    tmp_state_dir, monkeypatch, tmp_path,
):
    """v0.7.3 pre-tag CRITICAL fix lock-in: a draw run with N=3 writes
    3 distinct history rows with seeds [base, base+1, base+2]. Pre-fix
    (v0.7.3 commit `6ac326f`) wrote `ctx.seed` (= base) to all 3 rows,
    silently breaking `imgen replay` reproducibility for rows 2..N
    (each replay generated the same image as row 1 because the
    recorded seed was the base, not the actual ladder step).

    End-to-end: cmd_draw → run_one_iteration → safe_append_history.
    No stubbing of run_one_iteration — exercises the real history
    serialiser via the production code path."""
    from imgen.backends import BACKENDS
    from imgen.commands.draw import cmd_draw
    from types import SimpleNamespace
    from imgen.defaults import DEFAULTS

    def fake_load(args):
        return ("flux-dev", BACKENDS["flux-dev"], "tok",
                Path("/fake/mflux-generate"), None)
    monkeypatch.setattr(
        "imgen.commands.draw.load_backend_and_token", fake_load,
    )
    _touch_and_zero2 = lambda cmd, *a, **kw: (
        Path(cmd[cmd.index("--output") + 1]).touch(),
        0,
    )[1]
    # v0.8.3 M-NEW-C: single-patch — Engine.run path only.
    monkeypatch.setattr(
        "imgen.subprocess_helpers.run_with_stderr_redaction", _touch_and_zero2,
    )
    monkeypatch.setattr(
        "imgen.cmd_helpers.preflight_resources", lambda **kw: None,
    )
    args = SimpleNamespace(
        prompt="ladder seed test",
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        seed=700,           # base seed
        num_iterations=3,   # → ladder seeds 700, 701, 702
        model="flux-dev",
        preview=False,
        width=1024,
        height=1024,
        no_open=True,
        yes=True,
        dry_run=False,
        force=True,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
        output=None,
        output_dir=str(tmp_path),
        lora=None,
        no_lora=False,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    rc = cmd_draw(args)
    assert rc == 0
    entries = load_history()
    # Last 3 entries are this draw run's iterations.
    draw_entries = [e for e in entries if e.get("command") == "draw"]
    last_three = draw_entries[-3:]
    assert len(last_three) == 3
    # CRITICAL lock-in: per-row seed values match the ladder.
    seeds = [e["seed"] for e in last_three]
    assert seeds == [700, 701, 702], (
        f"history seed ladder broken: expected [700, 701, 702], got {seeds}. "
        f"Likely regression of the v0.7.3 pre-tag CRITICAL fix where "
        f"run_one_iteration wrote ctx.seed (= base_seed) to all N rows."
    )

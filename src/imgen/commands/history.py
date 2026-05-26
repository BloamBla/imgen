"""`imgen history` / `last` / `replay <id>` command handlers.

Pure UI on top of the history.py data layer (load/append). `replay_entry`
lives here (not in the data layer) because it bridges back into cmd_generate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .._safe import has_control_bytes
from ..colors import C, dim, die, info
from ..defaults import DEFAULTS, HISTORY_SCHEMA_VERSION
from ..history import entry_model_name, load_history
from ..styles import LoraRef
from .draw import cmd_draw
from .generate import cmd_generate
from .refine import cmd_refine


# v0.8.0 commit 9 (§K + §Q): the v0.8.0 commit 4b-era
# ``_normalize_backend_value`` helper is REPLACED by
# ``history.entry_model_name(entry)``. The new helper folds in three
# concerns the previous one didn't:
#   * dual-shape READ dispatch (v=4 ``model`` key wins; v=3 ``backend``
#     fallback) so the same display/replay/ETA code paths work across
#     both schema versions;
#   * §A.5 security filter (C0/DEL/C1 control bytes → None) so a
#     hand-edited history.jsonl can't feed a dirty string into argv;
#   * §R.3 HIGH-1 fix surface (cmd_helpers ETA matcher).
# Replay callers translate ``None`` (entirely missing model key) to the
# subcommand-appropriate default at the call site, not in the helper.

# Fields the replay Namespace must carry so cmd_generate doesn't have to
# rely on getattr-with-default. Each is what cmd_generate reads for a
# "user didn't pass this flag" semantics. Explicit fields > silent
# getattr fallbacks — keeps a future required arg from silently
# producing surprising replay behaviour.
_REPLAY_DEFAULTS = {
    "prompt_file": None,
    "output_dir": None,
    # `yes` skips the multi-style confirm gate. Replay always replays a
    # single entry → cmd_generate's multi=False, gate never fires.
    # Pinning it here means any future code that pushes replay through
    # a multi=True path fails loudly instead of AttributeError'ing on
    # `args.yes`. (python C1 from v0.2.3 review)
    "yes": False,
    "imgen_merged_defaults": DEFAULTS,
    "imgen_config_output_dir": None,
    # v0.5 enhance surface — replay deliberately does NOT auto-enhance
    # even when the original entry was generated with --enhance-prompt.
    # Reason: the enhancer is opt-in by user intent; silently re-enhancing
    # on replay would surprise users (and pay a 4 GB download + 5 s
    # inference cost they didn't ask for). To exactly reproduce an
    # enhanced run, the user must re-pass --enhance-prompt at the
    # replay call site (a future v0.5.x --re-enhance flag is on the
    # backlog). Pinning these as explicit False/None values surfaces a
    # missing flag as a loud AttributeError instead of a silent fallback
    # through getattr — mirrors the v0.2 architect #7 "explicit fields"
    # contract for the rest of this Namespace.
    "enhance": False,
    "enhance_model": None,
    "enhance_temperature": None,
    "imgen_config_enhance": {},
}


def _rehydrate_loras_from_entry(
    entry: dict,
) -> tuple[list[LoraRef] | None, bool]:
    """Reconstruct (cli_lora, no_lora) Namespace fields from a v=3
    history entry's ``loras`` list.

    Replay determinism rule (v0.6 onwards): a history entry's stored
    ``loras`` list is the GROUND TRUTH for what mflux saw on the
    original run. To reproduce, replay must pass exactly that list
    to ``build_iterations`` and SUPPRESS the style's current built-in
    LoRA mapping (which may have changed since the original run — e.g.
    a `pip upgrade` brought new built-in LoRA picks).

    Mapping:

    * v=3 entry with ``loras=[]`` (text-only run) → ``(None, True)``:
      ``--no-lora`` semantics, no CLI LoRAs. Style built-ins suppressed.
    * v=3 entry with ``loras=[...]`` (LoRAs were applied) →
      ``([LoraRef(...), ...], True)``: ``--no-lora`` carve-out keeps
      ``cli_lora`` since :func:`resolve_effective_loras` was updated
      so ``no_lora=True + cli_lora=[X]`` returns ``(X,)``.
    * v<3 entry (no ``loras`` field) → ``(None, False)``: pre-v0.6
      shape with no LoRA info recorded; replay falls back to the
      style's CURRENT LoRA stack. This is best-effort for old
      entries — not bit-deterministic if the style's LoRA mapping
      changed, but the v0.5 behaviour was the same lossy fallback,
      so no regression.

    Returns ``(cli_lora, no_lora)`` to be passed via
    :data:`_REPLAY_DEFAULTS` override.
    """
    if "loras" not in entry:
        # v<3 entry — pre-v0.6, no LoRA info recorded. Replay falls
        # back to whatever the style currently ships with.
        return None, False
    raw = entry["loras"]
    if not isinstance(raw, list):
        # Defensive: a hand-edited history.jsonl with a typo'd shape
        # shouldn't crash replay. Fall back to "no LoRA info" so the
        # generation still runs.
        return None, False
    loras: list[LoraRef] = []
    for item in raw:
        if not isinstance(item, dict) or "ref" not in item:
            continue
        loras.append(LoraRef(
            ref=str(item["ref"]),
            weight=float(item.get("weight", 1.0)),
            compatible_with=tuple(item.get("compatible_with", ("flux-1",))),
            trigger=item.get("trigger"),
        ))
    # Always pin no_lora=True for v=3 entries so the style's current
    # built-in LoRAs don't sneak back in alongside the stored stack.
    # When loras=[] (text-only original run), no_lora=True alone gives
    # the same empty stack via the resolve_effective_loras carve-out.
    return loras or None, True


def cmd_history(args) -> int:
    entries = load_history()
    if not entries:
        dim("No history yet")
        return 0
    n = max(1, args.last or 20)
    for entry in entries[-n:]:
        status_icon = "✅" if entry.get("status") == "success" else "❌"
        ts = entry.get("ts", "?")[:16].replace("T", " ")
        eid = str(entry.get("id", "?"))
        style = entry.get("style") or "custom"
        # v0.8.0 commit 9 (§K + §Q architect IMPORTANT lock-in): unified
        # model column across v=3 + v=4 entries. v=3 rows with
        # ``"backend": "flux"`` render as ``"flux-kontext"`` (the v0.8
        # canonical name) — same identifier the user sees in
        # ``--list-models``, in the parser's --model flag, and in the
        # ``[defaults] model`` config key. ``entry_model_name`` returns
        # None for entries with neither key (very old / hand-edited);
        # rendered as ``"—"`` (em dash) for visual symmetry.
        model = entry_model_name(entry) or "—"
        print(f"{C.DIM}#{eid:<4}{C.END} {status_icon} "
              f"{C.BOLD}{ts}{C.END}  "
              f"{C.INFO}{style:10}{C.END}  "
              f"{C.DIM}{model:14}{C.END}  "
              f"{Path(entry.get('input', '?')).name:30}  "
              f"→ {Path(entry.get('output', '?')).name}")
    return 0


def cmd_last(_args) -> int:
    entries = load_history()
    if not entries:
        die("No history yet", code=1)
    return replay_entry(entries[-1])


def cmd_replay(args) -> int:
    entries = load_history()
    target = next((e for e in entries if e.get("id") == args.id), None)
    if not target:
        die(f"No entry with id {args.id}", code=1)
    return replay_entry(target)


def _replay_draw_entry(entry: dict) -> int:
    """v0.7.0 (architect §J): replay a ``command="draw"`` history entry
    through :func:`cmd_draw`.

    Draw entries have ``input=null`` (no source photo) and carry the
    prompt directly via the ``prompt`` field. cmd_draw reads
    ``args.prompt`` (positional) so the rehydrated Namespace puts the
    stored prompt back there. LoRA rehydration reuses
    :func:`_rehydrate_loras_from_entry` — same v=3 contract as i2i
    entries (whatever mflux saw on the original run is what replay
    reproduces).
    """
    prompt = entry.get("prompt")
    if not prompt:
        die(f"History entry #{entry.get('id', '?')} has no prompt — "
            f"cannot replay.", code=1)
    info(f"Replaying #{entry.get('id')}: draw \"{prompt[:60]}"
         f"{'...' if len(prompt) > 60 else ''}\"")
    cli_lora, no_lora = _rehydrate_loras_from_entry(entry)
    # v0.7.0 t2i Namespace — mirror of cmd_draw's parser shape. NO
    # --scope, --strength, --style, --image; --width/--height carry
    # the entry's recorded dimensions (or DEFAULTS if missing).
    # _REPLAY_DEFAULTS already includes ``prompt_file=None`` + ``output_dir=None``
    # + enhance/yes/imgen_* fields; supply only the t2i-specific values
    # here to avoid duplicate-keyword-arg TypeError.
    args = argparse.Namespace(
        prompt=prompt,
        output=None,
        steps=entry.get("steps", DEFAULTS["steps"]),
        guidance=entry.get("guidance", DEFAULTS["guidance"]),
        seed=None,  # new random seed each replay
        model=entry_model_name(entry) or "flux-dev",
        quantize=entry.get("quantize", DEFAULTS["quantize"]),
        preview=entry.get("preview", False),
        width=entry.get("width", 1024),
        height=entry.get("height", 1024),
        # v0.8.2 M-2 closure: replay must round-trip the negative prompt.
        # The v=3/v=4 history-entry stores it under the "negative" key
        # (see cmd_helpers.py run_one_iteration's entry construction).
        # Pre-v0.8.2 the replay Namespace omitted this field; build_draw_
        # iterations resolved it via ``getattr(args, "negative_prompt",
        # None) or ""`` so the gap silently dropped the negative on
        # replay — bit-divergent reconstruction. Now explicit.
        negative_prompt=entry.get("negative") or None,
        no_open=False,
        dry_run=False,
        force=False,
        lora=cli_lora,
        no_lora=no_lora,
        **_REPLAY_DEFAULTS,
    )
    return cmd_draw(args)


def _replay_refine_entry(entry: dict) -> int:
    """v0.7.5 (architect IMPORTANT #A): replay a ``command="refine"``
    history entry through :func:`cmd_refine`.

    Refine entries have a non-null ``input`` (the original image that
    was refined) and carry the refine prompt under ``custom_prompt``
    (mirror of how generate-with-no-style records the user prompt).
    Width/height fields hold the TARGET dims; replay uses them as
    explicit ``--width/--height`` since storing ``--scale`` would not
    round-trip to identical pixel dims if the original input was
    resaved at different dims between runs.

    LoRA rehydration reuses :func:`_rehydrate_loras_from_entry` —
    same v=3 contract as i2i / draw entries. Whatever mflux saw on
    the original run is what replay reproduces, modulo a fresh seed.
    """
    image = entry.get("input")
    if not image:
        die(f"History entry #{entry.get('id', '?')} (command=refine) has "
            f"no input path — cannot replay.", code=1)
    prompt = entry.get("custom_prompt")
    info(f"Replaying #{entry.get('id')}: refine on {Path(image).name}")
    cli_lora, no_lora = _rehydrate_loras_from_entry(entry)
    # Mirror of cmd_refine's parser shape (_add_refine_args). NO
    # --scope, --style, --custom-prompt, --enhance-* — refine
    # has no such concepts. --scale set to None; --width/--height
    # carry the stored target dims (already 16-multiple-rounded
    # at original-run time, so the round-trip is bit-stable for
    # the resolution pipeline).
    args = argparse.Namespace(
        input=image,
        scale=None,
        width=entry.get("width"),
        height=entry.get("height"),
        prompt=prompt,
        output=None,
        steps=entry.get("steps", DEFAULTS["steps"]),
        guidance=entry.get("guidance", DEFAULTS["guidance"]),
        strength=entry.get("strength", 0.3),
        seed=None,  # new random seed each replay
        model=entry_model_name(entry) or "flux2-klein-edit-9b",
        quantize=entry.get("quantize", 4),
        preview=entry.get("preview", False),
        no_open=False,
        dry_run=False,
        force=False,
        lora=cli_lora,
        no_lora=no_lora,
        **_REPLAY_DEFAULTS,
    )
    return cmd_refine(args)


def _replay_video_entry(entry: dict) -> int:
    """v0.9 commit 5 (§J): replay a ``command="video"`` history entry
    through :func:`cmd_video`.

    Mirror of :func:`_replay_draw_entry` for t2v entries. Video entries
    have ``input=None`` (no source media — v0.9.0 LTX is t2v only) and
    carry the prompt directly. ``num_frames`` + ``fps`` are pinned
    from the stored entry so replay reproduces the same temporal
    structure; ``seed=None`` so each replay rolls a fresh random
    seed (same policy as draw/refine).

    ``cmd_video`` lands at commit 7 — until then this function's body
    will ImportError if reached. The dispatch branch in
    :func:`replay_entry` is unreachable in practice between commits
    5-6 because no user can WRITE a video entry without cmd_video.
    Test surface mocks the cmd_video import via ``sys.modules``.
    """
    # Lazy import: keeps commit 5 importable before cmd_video lands
    # at commit 7. Tests inject a fake module via sys.modules.
    from .video import cmd_video

    prompt = entry.get("prompt")
    if not prompt:
        die(f"History entry #{entry.get('id', '?')} (command=video) has "
            f"no prompt — cannot replay.", code=1)
    info(f"Replaying #{entry.get('id')}: video \"{prompt[:60]}"
         f"{'...' if len(prompt) > 60 else ''}\"")
    args = argparse.Namespace(
        prompt=prompt,
        output=None,
        steps=entry.get("steps", DEFAULTS["steps"]),
        guidance=entry.get("guidance", 3.0),
        seed=None,  # new random seed each replay
        model=entry_model_name(entry) or "ltx-video",
        width=entry.get("width", 768),
        height=entry.get("height", 512),
        negative_prompt=entry.get("negative") or None,
        # v0.9 video-specific Namespace surface — pinned from entry,
        # so replay reproduces num_frames/fps exactly. duration left
        # None because the parser (commit 7) treats --duration and
        # --num-frames as mutex; replay always picks the explicit
        # frame count to round-trip cleanly.
        num_frames=entry.get("num_frames", 25),
        fps=entry.get("fps", 24),
        duration=None,
        no_open=False,
        dry_run=False,
        force=False,
        # Video LoRA is reserved for v0.9.x; commit 7 parser accepts
        # but ignores --lora/--no-lora. Pinning here keeps the
        # Namespace shape complete so an unknown-attr at runtime
        # surfaces loudly instead of silently mis-resolving.
        lora=None,
        no_lora=False,
        **_REPLAY_DEFAULTS,
    )
    return cmd_video(args)


def replay_entry(entry: dict) -> int:
    entry_v = entry.get("v", 0)
    if entry_v > HISTORY_SCHEMA_VERSION:
        die(f"History entry #{entry.get('id', '?')} is from a newer schema "
            f"(v{entry_v} > v{HISTORY_SCHEMA_VERSION}). "
            f"Run `imgen upgrade` to pick up the new fields.", code=2)

    # v0.7.0 (architect §J + CRITICAL #1 from pre-tag review): route by
    # the ``command`` discriminator field. Pre-v0.7 entries were always
    # ``generate``/``batch`` (no field present); ``.get`` default keeps
    # backward compat. ``draw`` entries take the t2i path and skip the
    # "no input path" guard (input=None is legitimate for t2i).
    # v0.7.5 (architect IMPORTANT #A): ``refine`` entries route to
    # cmd_refine — without this dispatch they would mis-replay through
    # cmd_generate (refine entries have non-null input + custom_prompt,
    # so the i2i path's guard passes but builds a Kontext-style
    # restyle invocation instead of the refine pipeline).
    # v0.9 commit 5 (§J + security §R.1 NIT-1): ``video`` entries route
    # to cmd_video. Control-byte filter applied to the command value
    # mirrors the v0.8.1 LOW-3 filter on entry["backend"] /
    # entry["model"] — a hand-edited history.jsonl with C0/DEL/C1
    # bytes in command must NOT reach argparse / argv. Refuse with
    # die() rather than silently routing through a fallback.
    command = entry.get("command", "generate")
    if not isinstance(command, str) or has_control_bytes(command):
        die(f"History entry #{entry.get('id', '?')}: command contains "
            f"control bytes — refusing replay.", code=2)
    if command == "draw":
        return _replay_draw_entry(entry)
    if command == "refine":
        return _replay_refine_entry(entry)
    if command == "video":
        return _replay_video_entry(entry)

    image = entry.get("input")
    if not image:
        die(f"History entry #{entry.get('id', '?')} has no input path — "
            f"cannot replay.", code=1)
    info(f"Replaying #{entry.get('id')}: {entry.get('style')} on "
         f"{Path(image).name}")
    # cmd_generate's args.style is list[str] | None as of v0.2.3 —
    # history entries store a single string per generation (one entry
    # per style in a multi-style invocation), so wrap into a 1-element
    # list for replay.
    #
    # v0.7.13 (gap 8 behaviour pivot) clarifies the routing:
    #   * style truthy + no custom_prompt → preset replay (style_list
    #     populated, cmd_generate goes through build_iterations).
    #   * style truthy + custom_prompt → augmentation replay (same
    #     preset path; build_iterations layers --custom-prompt on top).
    #   * style=None + custom_prompt (pre-v0.7.13 augmentation entries
    #     OR v0.7.13 bare entries) → style_list=None → cmd_generate
    #     routes through bare mode. Pre-v0.7.13 augmented entries lose
    #     the default-style baggage they originally carried; that's
    #     intentional — the baggage was the bug we closed.
    #   * style=None + no custom_prompt → unreachable via normal UI in
    #     v0.7.13+ (would need a manually-edited history.jsonl); the
    #     ``or DEFAULTS["style"]`` fallback below resurrects the
    #     historical default rather than silently routing to bare with
    #     an empty prompt (which would crash mflux). Documented as a
    #     legacy-defence corner case; bare mode requires a non-empty
    #     prompt by design.
    saved_style = entry.get("style") or DEFAULTS["style"]
    style_list = [saved_style] if (saved_style and not entry.get("custom_prompt")) else None
    # v0.6: rehydrate the LoRA stack from the entry's stored snapshot.
    # See _rehydrate_loras_from_entry — for v=3 entries this faithfully
    # reproduces the original mflux invocation; for v<3 entries it
    # falls back to the style's current LoRA mapping (best-effort,
    # matches v0.5 behaviour).
    cli_lora, no_lora = _rehydrate_loras_from_entry(entry)
    args = argparse.Namespace(
        image=image,
        style=style_list,
        custom_prompt=entry.get("custom_prompt"),
        scope=entry.get("scope"),
        preview=entry.get("preview", False),
        output=None,  # auto-generate new output name
        steps=entry.get("steps", DEFAULTS["steps"]),
        guidance=entry.get("guidance", DEFAULTS["guidance"]),
        strength=entry.get("strength", DEFAULTS["strength"]),
        seed=None,  # new random seed
        model=entry_model_name(entry) or DEFAULTS["model"],
        quantize=entry.get("quantize", DEFAULTS["quantize"]),
        width=entry.get("width"),
        height=entry.get("height"),
        no_open=False,
        dry_run=False,
        force=False,
        # v0.6: LoRA rehydration. Replay reads the v=3 ``loras`` field
        # and pins the stack at the original run's snapshot, suppressing
        # whatever the style currently ships. Pre-v0.6 entries default
        # to "no CLI override, no opt-out" → style's current LoRAs apply.
        lora=cli_lora,
        no_lora=no_lora,
        # Explicit fields cmd_generate would otherwise read via getattr-
        # with-default. Pinning them here means a future required arg
        # added to cmd_generate fails replay loudly instead of silently
        # falling back to "user didn't pass it".
        **_REPLAY_DEFAULTS,
    )
    return cmd_generate(args)

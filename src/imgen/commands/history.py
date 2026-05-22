"""`imgen history` / `last` / `replay <id>` command handlers.

Pure UI on top of the history.py data layer (load/append). `replay_entry`
lives here (not in the data layer) because it bridges back into cmd_generate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..colors import C, dim, die, info
from ..defaults import DEFAULTS, HISTORY_SCHEMA_VERSION
from ..history import load_history
from ..styles import LoraRef
from .generate import cmd_generate

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
        print(f"{C.DIM}#{eid:<4}{C.END} {status_icon} "
              f"{C.BOLD}{ts}{C.END}  "
              f"{C.INFO}{style:10}{C.END}  "
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


def replay_entry(entry: dict) -> int:
    entry_v = entry.get("v", 0)
    if entry_v > HISTORY_SCHEMA_VERSION:
        die(f"History entry #{entry.get('id', '?')} is from a newer schema "
            f"(v{entry_v} > v{HISTORY_SCHEMA_VERSION}). "
            f"Run `imgen upgrade` to pick up the new fields.", code=2)
    image = entry.get("input")
    if not image:
        die(f"History entry #{entry.get('id', '?')} has no input path — "
            f"cannot replay.", code=1)
    info(f"Replaying #{entry.get('id')}: {entry.get('style')} on "
         f"{Path(image).name}")
    # cmd_generate's args.style is list[str] | None as of v0.2.3 — history
    # entries store a single string per generation (one entry per style in
    # a multi-style invocation), so wrap into a 1-element list for replay.
    # Default falls through to DEFAULTS["style"], not a hardcoded "pixar",
    # so a future default-style change doesn't silently divert old replay.
    # (python I2 from v0.2.3 review)
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
        backend=entry.get("backend", DEFAULTS["backend"]),
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

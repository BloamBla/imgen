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
from .generate import cmd_generate

# Fields the replay Namespace must carry so cmd_generate doesn't have to
# rely on getattr-with-default. Each is what cmd_generate reads for a
# "user didn't pass this flag" semantics.
_REPLAY_DEFAULTS = {
    "prompt_file": None,
    "imgen_merged_defaults": DEFAULTS,
    "imgen_config_output_dir": None,
}


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
    args = argparse.Namespace(
        image=image,
        style=entry.get("style", "pixar") if not entry.get("custom_prompt") else None,
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
        # Explicit fields cmd_generate would otherwise read via getattr-
        # with-default. Pinning them here means a future required arg
        # added to cmd_generate fails replay loudly instead of silently
        # falling back to "user didn't pass it".
        **_REPLAY_DEFAULTS,
    )
    return cmd_generate(args)

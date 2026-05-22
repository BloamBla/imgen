"""Subcommand handlers — one file per `imgen <command>` subcommand.

Each module exposes a `cmd_<name>(args)` function consumed by cli.main()'s
dispatch table. Add a new subcommand by editing four files:
  1. src/imgen/parser.py — add the argparse stanza.
  2. src/imgen/commands/<name>.py — `def cmd_<name>(args)`.
  3. src/imgen/commands/__init__.py — re-export `cmd_<name>` here.
  4. src/imgen/cli.py — add to `_KNOWN_SUBCOMMANDS` set + `_HANDLERS` map.
"""
from __future__ import annotations

from .batch import cmd_batch
from .clean import cmd_clean
from .doctor import cmd_doctor
from .generate import cmd_generate
from .history import cmd_history, cmd_last, cmd_replay
from .setup import cmd_setup
from .upgrade import cmd_upgrade

__all__ = [
    "cmd_batch",
    "cmd_clean",
    "cmd_doctor",
    "cmd_generate",
    "cmd_history",
    "cmd_last",
    "cmd_replay",
    "cmd_setup",
    "cmd_upgrade",
]

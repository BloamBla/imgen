"""Subcommand handlers — one file per `imgen <command>` subcommand.

Each module exposes a `cmd_<name>(args)` function consumed by cli.main()'s
dispatch table. Add a new subcommand by:
  1. Adding the parser stanza in src/imgen/parser.py.
  2. Creating src/imgen/commands/<name>.py with `def cmd_<name>(args)`.
  3. Adding it to the dispatch table in cli.main().
"""
from __future__ import annotations

from .clean import cmd_clean
from .doctor import cmd_doctor
from .generate import cmd_generate
from .history import cmd_history, cmd_last, cmd_replay
from .setup import cmd_setup
from .upgrade import cmd_upgrade

__all__ = [
    "cmd_clean",
    "cmd_doctor",
    "cmd_generate",
    "cmd_history",
    "cmd_last",
    "cmd_replay",
    "cmd_setup",
    "cmd_upgrade",
]

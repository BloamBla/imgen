"""
imgen — Photo style-transfer CLI for Apple Silicon Macs.

Uses mflux (MLX-native) under the hood. Default backend is FLUX Kontext Dev
(gated, requires HF token + license). Qwen-Image-Edit available as fallback.

Usage:
    imgen photo.jpg                              # default: pixar style
    imgen photo.jpg --style anime
    imgen photo.jpg --custom-prompt "..."
    imgen photo.jpg -s simpsons --steps 30 --strength 0.7
    imgen photo.jpg --backend qwen               # use Qwen Edit instead of FLUX

    imgen --list-styles
    imgen --dry-run photo.jpg --style anime

    imgen setup                                  # first-time install / token
    imgen doctor                                 # check environment
    imgen upgrade                                # update mflux
    imgen clean [--all]                          # cleanup HF cache
    imgen history [--last N]                     # show generation history
    imgen last                                   # repeat last generation
    imgen replay <id>                            # repeat generation by id
"""
from __future__ import annotations

import signal
import sys

from .colors import warn
from .commands import (
    cmd_clean,
    cmd_doctor,
    cmd_generate,
    cmd_history,
    cmd_last,
    cmd_replay,
    cmd_setup,
    cmd_upgrade,
)
from .parser import build_parser, print_styles

_KNOWN_SUBCOMMANDS = {
    "setup", "doctor", "upgrade", "clean",
    "history", "last", "replay", "generate",
}

_HANDLERS = {
    "setup": cmd_setup,
    "doctor": cmd_doctor,
    "upgrade": cmd_upgrade,
    "clean": cmd_clean,
    "history": cmd_history,
    "last": cmd_last,
    "replay": cmd_replay,
    "generate": cmd_generate,
}


def main() -> int:
    # If the FIRST non-flag arg isn't a known subcommand, prepend "generate".
    # Only checking the first positional avoids two prior pitfalls:
    #   - a path like "last.jpg" being mistaken for the "last" subcommand
    #   - an --option value that happens to match a subcommand name
    #     blocking the shorthand dispatch
    argv = sys.argv[1:]
    first_positional = next((a for a in argv if not a.startswith("-")), None)
    if first_positional and first_positional not in _KNOWN_SUBCOMMANDS:
        argv = ["generate"] + argv

    epilog = __doc__.split("Usage:", 1)[1] if __doc__ else None
    parser = build_parser(epilog=epilog)
    args = parser.parse_args(argv)

    # Top-level info actions: handled before subcommand dispatch
    if getattr(args, "list_styles", False):
        return print_styles()

    if not args.command:
        parser.print_help()
        return 0

    handler = _HANDLERS.get(args.command)
    if not handler:
        parser.print_help()
        return 2

    # Graceful SIGINT
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        return handler(args) or 0
    except KeyboardInterrupt:
        print()
        warn("Cancelled by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())

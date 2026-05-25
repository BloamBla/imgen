"""
imgen — Photo style-transfer CLI for Apple Silicon Macs.

Uses mflux (MLX-native) under the hood. Default backend is FLUX Kontext Dev
(gated, requires HF token + license). Qwen-Image-Edit available as fallback.

Usage:
    imgen photo.jpg                              # default: pixar style
    imgen photo.jpg --style anime
    imgen photo.jpg --custom-prompt "..."        # prompt in argv (visible to `ps auxww`)
    imgen photo.jpg --custom-prompt -            # prompt from stdin (hidden from ps)
    imgen photo.jpg --prompt-file ~/p.txt        # prompt from file (hidden from ps)
    imgen photo.jpg -s simpsons --steps 30 --strength 0.7
    imgen photo.jpg --model qwen-image-edit-v1   # use Qwen Edit instead of FLUX
    imgen vacation.heic                          # HEIC auto-converted via sips (v0.3.0)

    imgen batch ~/Desktop/holiday                # v0.3.0: every photo in folder, default style
    imgen batch <dir> -s anime,ghibli,pixar      # N inputs × M styles into one timestamped folder
    imgen batch <dir> --dry-run                  # show every mflux command without running

    imgen --list-styles
    imgen --dry-run photo.jpg --style anime

    imgen setup                                  # first-time install / token / config / styles.d
    imgen doctor                                 # check environment + cached models + user config
    imgen upgrade                                # self-update imgen + refresh mflux
    imgen clean [--all]                          # cleanup HF cache
    imgen history [--last N]                     # show generation history
    imgen last                                   # repeat last generation
    imgen replay <id>                            # repeat generation by id

User config: ~/.imgen/config.toml — see README
User styles: ~/.imgen/styles.d/*.toml — see README
"""
from __future__ import annotations

import signal
import sys

from .colors import warn
from .commands import (
    cmd_batch,
    cmd_clean,
    cmd_doctor,
    cmd_draw,
    cmd_generate,
    cmd_history,
    cmd_last,
    cmd_migrate_toml,
    cmd_refine,
    cmd_replay,
    cmd_setup,
    cmd_upgrade,
)
from .config import ConfigError, effective_defaults, load_validated_config
from .defaults import DEFAULTS
from .parser import (
    _check_for_deprecated_backend_flag,
    _check_for_deprecated_list_backends_flag,
    build_parser,
    print_loras,
    print_models,
    print_styles,
)
from .paths import CONFIG_FILE

_KNOWN_SUBCOMMANDS = {
    "setup", "doctor", "upgrade", "clean",
    "history", "last", "replay", "generate", "batch", "draw", "refine",
    "migrate-toml",
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
    "batch": cmd_batch,
    "draw": cmd_draw,
    "refine": cmd_refine,
    "migrate-toml": cmd_migrate_toml,
}


def main() -> int:
    # If the FIRST non-flag arg isn't a known subcommand, prepend "generate".
    # Only checking the first positional avoids two prior pitfalls:
    #   - a path like "last.jpg" being mistaken for the "last" subcommand
    #   - an --option value that happens to match a subcommand name
    #     blocking the shorthand dispatch
    argv = sys.argv[1:]
    # v0.8.0 commits 4a + 4b: pre-argparse hooks surface clean migration
    # messages for the deprecated flags BEFORE the implicit-generate
    # prepend or argparse's "unrecognized arguments" error. Run on the
    # raw sys.argv tail so both subcommand-prepended and bare forms get
    # the same hint.
    _check_for_deprecated_backend_flag(argv)
    _check_for_deprecated_list_backends_flag(argv)
    first_positional = next((a for a in argv if not a.startswith("-")), None)
    if first_positional and first_positional not in _KNOWN_SUBCOMMANDS:
        argv = ["generate"] + argv

    # Load ~/.imgen/config.toml best-effort. A broken config WARNs and
    # falls back to built-in defaults rather than blocking — keeps
    # `imgen --version`/`doctor` working when the user's config has a
    # typo'd value.
    try:
        config = load_validated_config(CONFIG_FILE)
    except ConfigError as e:
        warn(f"~/.imgen/config.toml: {e}")
        warn("Falling back to built-in defaults. Fix the file or remove it.")
        # Keep all three sections so downstream code's .get() pattern works.
        config = {"defaults": {}, "ui": {}, "enhance": {}}

    merged_defaults = effective_defaults(config["defaults"], DEFAULTS)

    epilog = __doc__.split("Usage:", 1)[1] if __doc__ else None
    parser = build_parser(epilog=epilog, defaults=merged_defaults)
    args = parser.parse_args(argv)

    # Stash config-aware values for handlers (commands/generate.py reads
    # these). `imgen_` prefix to avoid clashing with any future argparse
    # field name.
    args.imgen_merged_defaults = merged_defaults
    args.imgen_config_output_dir = config["defaults"].get("output_dir")
    # v0.5: [enhance] section passed to cmd_generate / cmd_batch where
    # effective_enhance() resolves it against CLI overrides.
    args.imgen_config_enhance = config.get("enhance", {})

    # UI: [ui] open_in_preview = false → behave like --no-open by default
    if getattr(args, "no_open", False) is False:
        if config["ui"].get("open_in_preview", True) is False:
            args.no_open = True

    # Top-level info actions: handled before subcommand dispatch
    if getattr(args, "list_styles", False):
        return print_styles()
    if getattr(args, "list_models", False):
        return print_models()
    if getattr(args, "list_loras", False):
        return print_loras()

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

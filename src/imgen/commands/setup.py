"""`imgen setup` — Apple Silicon check + verify venv/mflux installed by the
installer (bootstrap.sh or pipx), interactive HF token entry, shell alias
for bootstrap-installed users.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from ..checks import check_mflux, check_venv
from ..colors import C, dim, die, info, ok, step, warn
from ..defaults import DEFAULTS
from ..paths import (
    CONFIG_FILE,
    DEFAULT_OUTPUT_DIR,
    IMGEN_HOME,
    STATE_DIR,
    TOKEN_FILE,
    VENV_BIN,
)
from ..tokens import load_token, save_token_atomic, validate_token


_STARTER_CONFIG_TEMPLATE = f"""\
# imgen config — uncomment lines to override built-in defaults.
# All keys are optional; missing keys fall back to module DEFAULTS.
# Precedence: CLI flag > this file > built-in DEFAULTS.

[defaults]
# style = "{DEFAULTS['style']}"             # one of: imgen --list-styles
# backend = "{DEFAULTS['backend']}"             # "flux" (needs HF token) | "qwen" (open)
# quantize = {DEFAULTS['quantize']}                  # 3, 4, 5, 6, or 8
# steps = {DEFAULTS['steps']}                    # 1..200
# guidance = {DEFAULTS['guidance']}                # 0.5..15.0  (preset may override)
# strength = {DEFAULTS['strength']}               # 0.0..1.0   (preset may override)
# output_dir = "~/Desktop/imgen"  # env IMGEN_OUTPUT_DIR still wins over this

[ui]
# open_in_preview = true          # auto-open result in Preview (default true)
# color = "auto"                  # "auto" | "always" | "never"  (reserved for v0.3)
"""


def cmd_setup(_args) -> int:
    step("imgen auto-setup")
    print()

    # Apple Silicon check (MLX requires arm64)
    import platform
    if platform.system() != "Darwin":
        die(f"macOS only — detected {platform.system()}",
            code=3,
            hint="MLX (mflux backend) is Apple-only.")
    if platform.machine() != "arm64":
        die(f"Apple Silicon required — detected {platform.machine()}",
            code=3,
            hint="MLX does not support Intel Macs.")
    ok(f"macOS {platform.mac_ver()[0]} on {platform.machine()}")

    # venv + mflux: install mode is set up by either bootstrap.sh
    # (creates ~/imgen/.venv, pip install -e ., mflux from pyproject deps)
    # or by pipx (manages its own venv). `imgen setup` only verifies and
    # points to the right installer on failure.
    if not check_venv():
        if IMGEN_HOME:
            die("venv missing", code=3,
                hint=f"Run: {IMGEN_HOME / 'bootstrap.sh'}")
        die("venv missing", code=3,
            hint="Reinstall: pipx install --force "
                 "git+https://github.com/BloamBla/imgen")
    ok(f"venv at {VENV_BIN.parent}")

    mflux_ver = check_mflux()
    if not mflux_ver:
        if IMGEN_HOME:
            die(f"mflux not installed in {VENV_BIN.parent}",
                code=3, hint=f"Run: {IMGEN_HOME / 'bootstrap.sh'}")
        die("mflux not installed (should have come with the pipx install)",
            code=3,
            hint="Reinstall: pipx install --force "
                 "git+https://github.com/BloamBla/imgen")
    ok(f"mflux {mflux_ver}")

    # HF token
    print()
    if load_token():
        ok("HF token already configured")
    else:
        info("HuggingFace token setup (optional)")
        print(f"   {C.DIM}Token enables FLUX Kontext (best quality).{C.END}")
        print(f"   {C.DIM}Without token, only Qwen Edit backend works.{C.END}")
        print()
        print(f"   {C.BOLD}Get token:{C.END}")
        print(f"     1. https://huggingface.co/settings/tokens")
        print(f"        Create classic 'Read' token (NOT fine-grained).")
        print(f"     2. Accept FLUX license:")
        print(f"        https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev")
        print()
        try:
            tok = input("   Paste token (or Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            tok = ""

        if tok:
            if not tok.startswith("hf_"):
                warn("Token doesn't start with 'hf_' — saving anyway")
            try:
                if TOKEN_FILE.exists():
                    TOKEN_FILE.unlink()
                save_token_atomic(tok)
            except OSError as e:
                die(f"Couldn't write {TOKEN_FILE}: {e}", code=3)
            ok(f"Token saved to {TOKEN_FILE} (chmod 600)")
            user = validate_token(tok)
            if user:
                ok(f"Token valid (HF user: {user})")
            else:
                warn("Token saved but couldn't validate — could be invalid, "
                     "expired, or network issue. Check at: "
                     "https://huggingface.co/settings/tokens")
        else:
            dim("   Skipped. Run `imgen setup` later to add token.")

    # State dirs
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Starter config.toml — only if not present, never overwrite. All
    # keys commented out so defaults stay in effect until the user opts in.
    print()
    info("Config file")
    if CONFIG_FILE.exists():
        ok(f"Config already at {CONFIG_FILE}")
    else:
        try:
            CONFIG_FILE.write_text(_STARTER_CONFIG_TEMPLATE)
            CONFIG_FILE.chmod(0o600)
            ok(f"Created starter config at {CONFIG_FILE}")
            print(f"   {C.DIM}Edit to customize style/backend/output_dir/etc.{C.END}")
        except OSError as e:
            warn(f"Couldn't write {CONFIG_FILE}: {e}")

    # Shell alias — only for bootstrap-installed users. pipx users have
    # `imgen` in PATH via ~/.local/bin/ already; an alias would shadow.
    if IMGEN_HOME:
        print()
        info("Shell alias")
        shell_path = os.environ.get("SHELL", "")
        shell_name = Path(shell_path).name if shell_path else ""
        rc_files = {
            "zsh": Path.home() / ".zshrc",
            "bash": Path.home() / ".bash_profile",
            "fish": Path.home() / ".config" / "fish" / "config.fish",
        }
        rc_file = rc_files.get(shell_name)
        # shlex.quote() safely escapes paths containing quotes, spaces,
        # $, ;, etc. so a repo cloned into a weird directory can't inject
        # shell code.
        alias_line = f"alias imgen={shlex.quote(str(IMGEN_HOME / 'imgen'))}"

        if rc_file is None:
            warn(f"Unknown shell '{shell_name}' — skipping alias setup")
            print(f"   {C.DIM}Add manually to your shell rc: {alias_line}{C.END}")
        else:
            try:
                existing = rc_file.read_text() if rc_file.exists() else ""
            except OSError:
                existing = ""
            if alias_line in existing:
                ok(f"Alias already in {rc_file}")
            else:
                try:
                    rc_file.parent.mkdir(parents=True, exist_ok=True)
                    with rc_file.open("a") as f:
                        f.write(f"\n# imgen — photo style transfer\n{alias_line}\n")
                    ok(f"Added alias to {rc_file}")
                    print(f"   {C.DIM}Restart terminal or: source {rc_file}{C.END}")
                except OSError as e:
                    warn(f"Couldn't write {rc_file}: {e}")
                    print(f"   {C.DIM}Add manually: {alias_line}{C.END}")
    else:
        print()
        ok("Pipx install — `imgen` already in your PATH, no alias needed")

    print()
    step("Setup complete!")
    print(f"   {C.DIM}Try: imgen photo.jpg{C.END}")
    return 0

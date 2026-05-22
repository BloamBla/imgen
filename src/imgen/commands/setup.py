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
from ..shell_rc import RC_FILE_BY_SHELL
from ..paths import (
    CONFIG_FILE,
    DEFAULT_OUTPUT_DIR,
    IMGEN_HOME,
    LEGACY_TOKEN_FILE,
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
# color = "auto"                  # "auto" | "always" | "never"  (NO_COLOR env wins)
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
            # Drop the legacy ~/.hf_token if it's still hanging around so
            # we don't end up with two copies and silent fallback later.
            if LEGACY_TOKEN_FILE.exists():
                try:
                    LEGACY_TOKEN_FILE.unlink()
                    dim(f"   removed legacy {LEGACY_TOKEN_FILE}")
                except OSError as e:
                    warn(f"Saved new token but couldn't remove legacy "
                         f"{LEGACY_TOKEN_FILE}: {e}")
            result = validate_token(tok)
            if result.username:
                ok(f"Token valid (HF user: {result.username})")
            elif result.error == "auth":
                warn("Token saved but HF rejected it (401) — likely "
                     "invalid, revoked, or missing required scopes. "
                     "Generate a new one at: "
                     "https://huggingface.co/settings/tokens")
            elif result.error == "network":
                warn("Token saved but couldn't reach HF to validate — "
                     "offline, DNS issue, or HF is down. The token will "
                     "be tried as-is on your next `imgen generate`.")
            else:  # "parse" — captive portal / proxy serving a non-JSON 200
                warn("Token saved but HF returned an unexpected response "
                     "(captive portal or proxy?). Confirm internet access "
                     "and re-run `imgen setup` to retry validation.")
        else:
            dim("   Skipped. Run `imgen setup` later to add token.")

    # State dirs
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # User-styles directory (empty placeholder so the user finds where to
    # drop their .toml files without reading README first)
    styles_dir = STATE_DIR / "styles.d"
    if not styles_dir.exists():
        # 0o700 for consistency with STATE_DIR — user-authored style
        # presets may embed proprietary prompts.
        styles_dir.mkdir(mode=0o700)
        (styles_dir / "README.txt").write_text(
            "Drop *.toml files here to add user style presets.\n"
            "Filename (without .toml) becomes the style name.\n"
            "Required fields: none — but if `prompt` is missing, you'll\n"
            "need to pass --custom-prompt at run time.\n"
            "Optional fields: prompt, negative, guidance (0.5-15),\n"
            "                 strength (0-1).\n"
            "\n"
            "Example: ~/.imgen/styles.d/noir.toml\n"
            '  prompt = "film noir, black and white, dramatic shadows"\n'
            '  negative = "color, daylight"\n'
            "  guidance = 4.5\n"
            "  strength = 0.65\n"
        )

    # User-backends directory (v0.4 — same pattern as styles.d).
    # 0o700 + README that doubles as schema docs + an explicit security
    # warning, since `binary = ...` becomes an actual subprocess exec.
    backends_dir = STATE_DIR / "backends.d"
    if not backends_dir.exists():
        backends_dir.mkdir(mode=0o700)
        (backends_dir / "README.txt").write_text(
            "Drop *.toml files here to add image-gen backends beyond\n"
            "the built-in flux + qwen. Filename (without .toml) becomes\n"
            "the --backend NAME.\n"
            "\n"
            "SECURITY: `binary = ...` is executed as a subprocess by imgen.\n"
            "Treat backends.d/ files like shell scripts — only drop in files\n"
            "you wrote yourself or got from a source you trust.\n"
            "\n"
            "Required fields:\n"
            '  binary     = "..."   (bare name on $PATH, or absolute path)\n'
            '  image_flag = "..."   ("--image-path" or "--image-paths")\n'
            "\n"
            "Optional fields:\n"
            '  supports_strength = false   (true → accepts --image-strength)\n'
            '  supports_negative = false   (true → accepts --negative-prompt)\n'
            '  extra_args        = []      (e.g. ["--model", "sdxl"])\n'
            "\n"
            "Optional [secret] section — for backends that need an API\n"
            "key or token in the subprocess env:\n"
            '  [secret]\n'
            '  env_var  = "MY_BACKEND_API_KEY"\n'
            '  required = true   (false = best-effort forward)\n'
            "\n"
            "Example: ~/.imgen/backends.d/sdxl.toml\n"
            '  binary = "mflux-generate-sdxl"\n'
            '  image_flag = "--image-path"\n'
            '  supports_strength = true\n'
            '  extra_args = ["--model", "sdxl"]\n'
            "\n"
            "Verify with: imgen --list-backends   /   imgen doctor\n"
        )

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
        rc_file_rel = RC_FILE_BY_SHELL.get(shell_name)
        rc_file = (Path.home() / rc_file_rel) if rc_file_rel else None
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

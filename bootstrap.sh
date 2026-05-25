#!/usr/bin/env bash
# imgen bootstrap — first-time install on a clean macOS Apple Silicon Mac.
#
# What it does:
#   1. Verify macOS + Apple Silicon
#   2. Verify Python 3.12 (suggest brew install if missing)
#   3. Create venv at .venv/
#   4. Install pinned mflux + deps
#   5. Run `imgen setup` to add shell alias and prompt for HF token
#
# Re-runnable: safe to run multiple times.

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  G='\033[92m'; Y='\033[93m'; R='\033[91m'; B='\033[94m'; D='\033[2m'; N='\033[0m'
else
  G=''; Y=''; R=''; B=''; D=''; N=''
fi
ok()   { printf "${G}✅${N} %s\n" "$*"; }
warn() { printf "${Y}⚠️  %s${N}\n" "$*"; }
err()  { printf "${R}❌${N} %s\n" "$*" >&2; }
step() { printf "${B}🚀 %s${N}\n" "$*"; }

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

step "imgen bootstrap"
echo

# ── 1. macOS / Apple Silicon check ────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" ]]; then
  err "macOS only. Detected: $(uname -s)"
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  err "Apple Silicon (M1/M2/M3/M4) required. Detected: $(uname -m)"
  err "MLX (the backend mflux uses) does not support Intel Macs."
  exit 1
fi
ok "macOS Apple Silicon"

# ── 2. Python 3.12 check ──────────────────────────────────────────────────
if ! command -v python3.12 >/dev/null 2>&1; then
  err "Python 3.12 not found in PATH"
  echo "   Install via Homebrew:"
  echo "     ${D}brew install python@3.12${N}"
  echo "   Then re-run this script."
  exit 1
fi
PY_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
ok "Python ${PY_VERSION}"

# ── 3. Create venv ────────────────────────────────────────────────────────
if [[ -d .venv && -x .venv/bin/python ]]; then
  ok "venv already exists at .venv/"
else
  step "Creating venv"
  python3.12 -m venv .venv
  ok "venv created"
fi

# ── 4. Install imgen package (also pulls mflux as a declared dep) ────────
# Editable install (-e) so `imgen upgrade` -> `git pull` picks up new code
# without needing a reinstall. mflux is declared in pyproject.toml deps.
step "Installing imgen package + dependencies (this can take 3-5 minutes)"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .
MFLUX_VERSION=$(.venv/bin/pip show mflux | awk '/^Version:/ {print $2}')
ok "imgen package installed (mflux ${MFLUX_VERSION})"

# ── 5. Make imgen launcher executable ─────────────────────────────────────
chmod +x ./imgen
ok "imgen launcher is executable"

# ── 5b. Diffusers stack (v0.8.0 commit 6, opt-in) ─────────────────────────
# Per [[project-v080-design]] §E.2. The diffusers_mps engine spawns a
# subprocess in a SEPARATE Python venv (~10 GB on disk: torch +
# diffusers + transformers + accelerate). mflux-only colleagues skip
# this. Architect commit-6 pre-vet C1: anchor the venv path to
# $SCRIPT_DIR (the bootstrap-side install root) — IMGEN_INSTALL_ROOT
# is a Python module constant, NOT a shell variable.
#
# Three install modes (architect H4):
#   - Interactive TTY → prompt [y/N].
#   - Non-interactive (curl|bash, CI, ssh < /dev/null) → skip with a
#     hint pointing at the env-var override below.
#   - IMGEN_INSTALL_DIFFUSERS=1 → install unconditionally
#     (CI / scripted installs).

install_diffusers_stack() {
  step "Installing diffusers stack into .venv-diffusers/ (this adds ~10 GB)"
  python3.12 -m venv "$SCRIPT_DIR/.venv-diffusers"
  # Editable install of imgen INTO the diffusers venv so
  # `.venv-diffusers/bin/python -m imgen.engines._diffusers_runner` resolves
  # without re-installing the whole imgen package.
  "$SCRIPT_DIR/.venv-diffusers/bin/pip" install --quiet --upgrade pip
  "$SCRIPT_DIR/.venv-diffusers/bin/pip" install --quiet -e "$SCRIPT_DIR"
  # Heavy deps. torch downloads ~700 MB; diffusers ~120 MB.
  "$SCRIPT_DIR/.venv-diffusers/bin/pip" install --quiet \
    diffusers transformers accelerate torch
  ok "diffusers stack installed at $SCRIPT_DIR/.venv-diffusers/"
}

if [[ -d "$SCRIPT_DIR/.venv-diffusers" && -x "$SCRIPT_DIR/.venv-diffusers/bin/python" ]]; then
  ok "diffusers venv already exists at .venv-diffusers/"
elif [[ "${IMGEN_INSTALL_DIFFUSERS:-}" = "1" ]]; then
  install_diffusers_stack
elif [[ ! -t 0 ]]; then
  echo
  echo "${D}Skipping diffusers stack install (non-interactive bootstrap).${N}"
  echo "${D}If you need the diffusers_mps engine later, re-run from a TTY${N}"
  echo "${D}OR set IMGEN_INSTALL_DIFFUSERS=1 for non-interactive install:${N}"
  echo "${D}  IMGEN_INSTALL_DIFFUSERS=1 ./bootstrap.sh${N}"
else
  echo
  echo "Engine layer (v0.8.0+): install the diffusers stack for non-mflux models?"
  echo "${D}~10 GB on disk. Needed only for the diffusers_mps engine (e.g.${N}"
  echo "${D}qwen-image-2512-bf16, future HF day-0 models). Mflux-only setups${N}"
  echo "${D}can skip this.${N}"
  echo
  read -r -p "Install diffusers stack? [y/N] " answer
  if [[ "$answer" =~ ^[Yy] ]]; then
    install_diffusers_stack
  else
    echo "${D}Skipping. Re-run bootstrap.sh later (or set IMGEN_INSTALL_DIFFUSERS=1) to install.${N}"
  fi
fi

# ── 6. Hand off to imgen setup (alias + token) ────────────────────────────
echo
step "Running 'imgen setup' for shell alias and HF token"
./imgen setup

# ── 6b. v0.8.0 file-location migration nudge ──────────────────────────────
# Detects legacy ~/.imgen/backends.d/ TOMLs left over from a v0.7.x
# install. Nudge — not a hard error; the loader still reads the legacy
# path with a DEPRECATED warn through the v0.8.x window. (See
# project_v080_design.md §H.)
LEGACY_BACKENDS_D="$HOME/.imgen/backends.d"
if [[ -d "$LEGACY_BACKENDS_D" ]] && compgen -G "$LEGACY_BACKENDS_D/*.toml" >/dev/null 2>&1; then
  echo
  warn "Legacy ~/.imgen/backends.d/ TOMLs detected"
  echo "${D}v0.8.0 renamed the user-TOML directory to ~/.imgen/models.d/.${N}"
  echo "${D}Files in backends.d/ keep working through v0.8.x but emit a${N}"
  echo "${D}DEPRECATED warn on every imgen run. To move them now:${N}"
  echo "${D}  imgen migrate-toml${N}"
fi

echo
step "Bootstrap complete!"
printf "${D}Try: ${N}imgen photo.jpg --preview\n"
printf "${D}Docs: ${N}imgen --help\n"

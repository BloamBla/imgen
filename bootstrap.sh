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

# ── 6. Hand off to imgen setup (alias + token) ────────────────────────────
echo
step "Running 'imgen setup' for shell alias and HF token"
./imgen setup

echo
step "Bootstrap complete!"
printf "${D}Try: ${N}imgen photo.jpg --preview\n"
printf "${D}Docs: ${N}imgen --help\n"

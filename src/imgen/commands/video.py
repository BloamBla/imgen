"""v0.9 ``imgen video`` subcommand + lazy video-deps installer.

This module ships in two stages:

* **commit 6** (current): ``ensure_video_deps_or_die`` only. Lazy
  installer for the three pinned video packages on first use.
  Mirrors the bootstrap.sh diffusers opt-in pattern but scoped to
  video-specific deps so image-only diffusers colleagues never see
  ``imageio`` / ``imageio-ffmpeg`` / ``sentencepiece`` in their venv.

* **commit 7** (next): ``cmd_video`` subcommand handler + shared
  ``_orchestrate_t2x`` extraction shared with cmd_draw.

The deps installer is shipped first because commit 7's cmd_video
calls it; keeping them in the same file matches the v0.8.4
``commands/draw.py``-style locality (each subcommand owns its
adjacent helpers).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..colors import die

__all__ = ["_VIDEO_DEPS_PINNED", "ensure_video_deps_or_die"]


# ── Pinned dependency set (§E.5.1) ────────────────────────────────────
#
# Exact-version pins (no wildcards, no `>=`) per security §R.1
# CRITICAL-1. PyPI compromise / typo-squat / dependency-confusion are
# real threats for ANY fetch-and-exec path that fires on every cold-
# cache invocation across all colleagues. Exact pins make the threat
# window observable: PyPI compromise of one of these three would NOT
# silently land — pip refuses any installed version that doesn't
# match.
#
# Versions captured 2026-05-26 from a clean ``.venv-diffusers/`` smoke
# install. Bumps trigger a fresh security-reviewer pass; no silent
# ``imageio==2.37.*`` wildcard. Hash-pinning
# (``--require-hashes``) deferred to v0.9.x.
_VIDEO_DEPS_PINNED: tuple[str, ...] = (
    "imageio==2.37.3",
    "imageio-ffmpeg==0.6.0",
    "sentencepiece==0.2.1",
)


def _video_deps_present(python_path: Path) -> bool:
    """Probe whether imageio + imageio_ffmpeg + sentencepiece are
    importable inside .venv-diffusers/. Subprocess so the main
    venv stays isolated from the diffusers stack at import time."""
    rc = subprocess.run(
        [str(python_path), "-c",
         "import imageio, imageio_ffmpeg, sentencepiece"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return rc == 0


def _write_audit_marker(state_dir: Path) -> None:
    """Per §E.5.3: stamp ~/.imgen/video_deps_installed_at.txt with
    timestamp + pinned versions. doctor reads this for drift detection.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / "video_deps_installed_at.txt"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines = [f"installed_ts: {ts}", "pinned_versions:"]
    for pin in _VIDEO_DEPS_PINNED:
        lines.append(f"  {pin}")
    marker.write_text("\n".join(lines) + "\n")


def _prompt_install_or_die() -> None:
    """Render the multi-line install prompt + read y/n from stdin.
    Caller has already gated on isatty() so the read is safe to
    block on."""
    print("⚠️  imgen video needs three extra packages in "
          ".venv-diffusers/ (~60 MB total, pinned):",
          file=sys.stderr)
    for pin in _VIDEO_DEPS_PINNED:
        print(f"   • {pin}", file=sys.stderr)
    print("Install now? [y/N]: ", end="", file=sys.stderr, flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        die("Install declined (EOF). Re-run `imgen video ...` "
            "and answer 'y', or set IMGEN_INSTALL_VIDEO_DEPS=1 "
            "to opt in non-interactively.", code=2)
    if answer != "y":
        die("Install declined. Re-run `imgen video ...` and "
            "answer 'y', or set IMGEN_INSTALL_VIDEO_DEPS=1 to opt "
            "in non-interactively.", code=2)


def ensure_video_deps_or_die() -> None:
    """First-call gate for ``imgen video``. If imageio +
    imageio-ffmpeg + sentencepiece are not installed in
    ``.venv-diffusers/``, prompt (TTY) or env-var-opt-in (non-TTY)
    to install the pinned set; otherwise die with helpful hint.

    Returns silently when deps are already present. On install
    success writes the audit marker; on install failure preserves
    the sentinel for the next invocation to surface.

    Caller is responsible for gating on ``args.dry_run`` (dry-run
    must NOT trigger installs).

    Security guards (§E.5.4 + §E.5.5):
    * Sentinel ``~/.imgen/.video_deps_installing`` blocks if a
      previous install was interrupted — venv may be inconsistent.
    * pip / python paths refused if symlinks (same-uid attacker
      plant) or missing (user hasn't bootstrapped .venv-diffusers).
    * pip install bypasses ``subprocess_helpers.run_with_stderr_redaction``
      (the v0.8.2 RAM safety net; pip's RAM profile is fundamentally
      different from an ML subprocess).
    """
    from ..paths import IMGEN_INSTALL_ROOT, STATE_DIR

    pip_path = IMGEN_INSTALL_ROOT / ".venv-diffusers" / "bin" / "pip"
    python_path = IMGEN_INSTALL_ROOT / ".venv-diffusers" / "bin" / "python"
    sentinel = STATE_DIR / ".video_deps_installing"

    # 1. Sentinel guard (security §R.1 HIGH-3) — block if previous
    # install was interrupted.
    if sentinel.exists():
        die(
            "Previous video-deps install was interrupted "
            f"(sentinel file exists at {sentinel}). "
            "The .venv-diffusers/ may be in an inconsistent state. "
            "Remove .venv-diffusers/ entirely and re-run bootstrap.sh, "
            "then delete the sentinel file.",
            code=2,
        )

    # 2. Symlink + is_file guards (security §R.1 HIGH-2).
    for path, name in [(pip_path, "pip"), (python_path, "python")]:
        if path.is_symlink():
            die(
                f".venv-diffusers/bin/{name} is a symlink — refusing "
                "to exec. Same-uid attacker may have planted it. "
                "Remove .venv-diffusers/ and re-run bootstrap.sh.",
                code=2,
            )
        if not path.is_file():
            die(
                f".venv-diffusers/bin/{name} not found.\n"
                "  Run bootstrap.sh and answer 'y' at the diffusers "
                "opt-in prompt, OR set IMGEN_INSTALL_DIFFUSERS=1 "
                "to install non-interactively.",
                code=2,
            )

    # 3. Happy path: deps already present.
    if _video_deps_present(python_path):
        return

    # 4. Non-TTY guard / env-var bypass. The env-var path always
    # prints an audit line on stderr — never silent.
    env_bypass = os.environ.get("IMGEN_INSTALL_VIDEO_DEPS") == "1"
    if env_bypass:
        # Audit line on stderr per security §R.1 CRITICAL-2 — never
        # silent under env-var bypass. Goes to stderr (not stdout
        # via warn()) so colleagues piping output through grep don't
        # lose the audit signal.
        sys.stderr.write(
            "imgen: auto-installing video deps via "
            "IMGEN_INSTALL_VIDEO_DEPS=1\n"
            f"       (pinned: {', '.join(_VIDEO_DEPS_PINNED)})\n"
        )
    else:
        if not sys.stdin.isatty():
            die(
                "Non-interactive shell; refusing to prompt for "
                "video-deps install. Set IMGEN_INSTALL_VIDEO_DEPS=1 "
                "to opt in non-interactively.",
                code=2,
            )
        _prompt_install_or_die()

    # 5. Touch sentinel BEFORE install so a Ctrl-C / kill during pip
    # leaves the marker behind for the next invocation to surface
    # (§E.5.5 partial-fail recovery).
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    sentinel.touch()

    # 6. Run pip install with pinned versions. Plain subprocess.run
    # — NOT run_with_stderr_redaction (§E.5.6: pip's RAM profile is
    # different from an ML subprocess; the v0.8.2 < 4 GB safety
    # gate would refuse legitimate installs on a memory-tight Mac).
    install_rc = subprocess.run(
        [str(pip_path), "install", *_VIDEO_DEPS_PINNED],
    ).returncode
    if install_rc != 0:
        # Sentinel kept — next invocation tells user about the
        # corrupt state.
        die(
            f"pip install failed (rc={install_rc}). Sentinel kept; "
            "remove .venv-diffusers/ and re-run bootstrap.sh, then "
            "delete the sentinel before retrying.",
            code=2,
        )

    # 7. Verify install via the same probe used at step 3. If the
    # post-install probe fails, the install silently broke — keep
    # sentinel + die.
    if not _video_deps_present(python_path):
        die(
            "pip install reported success but the deps still don't "
            "import. Sentinel kept; remove .venv-diffusers/ and "
            "re-run bootstrap.sh.",
            code=2,
        )

    # 8. Write audit marker + remove sentinel.
    _write_audit_marker(STATE_DIR)
    try:
        sentinel.unlink()
    except OSError:
        # Sentinel removal failure is non-fatal — install succeeded.
        # Worst case: next invocation surfaces the stale sentinel,
        # which is recoverable.
        pass

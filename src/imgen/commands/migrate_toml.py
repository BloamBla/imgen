"""`imgen migrate-toml` — single-shot helper that moves user TOMLs
from the v0.7 ``~/.imgen/backends.d/`` location to the v0.8 canonical
``~/.imgen/models.d/`` location.

Per [[project-v080-design]] §H + §Q commit 10:

* For each ``*.toml`` under ``backends.d/``:
  - If the stem matches a v0.8 built-in Model name, suggest deletion
    ("safe to delete — built-in covers the same recipe"). Per §G.3.
  - Otherwise propose moving it to ``models.d/<stem>.toml``.
* Confirms before each action unless ``--yes`` is passed.
* Best-effort: per-file failures (permission, target-already-exists)
  warn and continue rather than aborting the whole run.

The helper is loud rather than silent — colleagues should see exactly
what would change before agreeing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..colors import C, dim, info, ok, step, warn
from ..models import BUILTIN_MODELS


def _safe_path_display(p: Path) -> str:
    """v0.4 IMP-2 control-byte discipline (round-3 security LOW carry):
    render the path via ``repr()`` so any C0/DEL/C1 byte in a hand-
    crafted directory name renders as a ``\\xNN`` literal instead of
    escaping into the user's terminal. Same pattern used in doctor's
    shadowing warn (commit 10) so the two surfaces are consistent."""
    return repr(str(p))


def _confirm(prompt: str, *, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    try:
        answer = input(f"   {prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def cmd_migrate_toml(args) -> int:
    """Move user TOMLs from ``backends.d/`` → ``models.d/``.

    Returns 0 even when individual files fail (the per-file warn is
    the user-facing signal), 0 when ``backends.d/`` doesn't exist
    (nothing to migrate is success), 0 when every file is already
    redundant (cleanup path).

    The only non-zero return today is 2 for unreadable state dir
    (permission error on ``backends.d/`` itself).
    """
    auto_yes = bool(getattr(args, "yes", False))

    # Late import so tests' monkeypatched STATE_DIR/BACKENDS_D/MODELS_D
    # in conftest.py take effect — module-top imports capture the real
    # ~/.imgen/ paths at import time.
    from ..paths import BACKENDS_D, MODELS_D

    step("imgen migrate-toml (backends.d/ → models.d/)")
    print()

    if not BACKENDS_D.exists():
        ok(f"No legacy {_safe_path_display(BACKENDS_D)} — nothing to migrate.")
        return 0
    if BACKENDS_D.is_symlink():
        warn(
            f"{_safe_path_display(BACKENDS_D)} is a symlink; refusing to "
            "operate on it (v0.4 IMP-3 cross-uid attack class). Remove "
            "the symlink and re-run, or migrate files manually."
        )
        return 2

    try:
        entries = sorted(BACKENDS_D.iterdir())
    except OSError as e:
        warn(f"Couldn't read {_safe_path_display(BACKENDS_D)}: {e}")
        return 2

    tomls = [p for p in entries if p.suffix == ".toml" and p.is_file()]
    if not tomls:
        ok(f"No *.toml files in {_safe_path_display(BACKENDS_D)}.")
        return 0

    info(f"Found {len(tomls)} legacy TOML(s):")
    for path in tomls:
        if path.stem in BUILTIN_MODELS:
            dim(f"   • {path.name}  (shadows built-in Model "
                f"'{path.stem}')")
        else:
            dim(f"   • {path.name}")
    print()

    # Per-file dispatch. Built-in-shadowing files get the deletion
    # path (their recipes are already covered by the built-in
    # registry); novel files get the relocate path.
    moved = 0
    deleted = 0
    skipped = 0
    failed = 0

    for path in tomls:
        if path.stem in BUILTIN_MODELS:
            # Deletion path: built-in covers the same recipe.
            print(
                f"{C.BOLD}{path.name}{C.END}: v0.8 built-in Model "
                f"'{path.stem}' covers the same recipe."
            )
            print(f"   {C.DIM}safe to delete {_safe_path_display(path)}{C.END}")
            if _confirm(f"Delete {path.name}?", auto_yes=auto_yes):
                try:
                    path.unlink()
                    deleted += 1
                    ok(f"deleted {_safe_path_display(path)}")
                except OSError as e:
                    failed += 1
                    warn(f"couldn't delete {_safe_path_display(path)}: {e}")
            else:
                skipped += 1
                dim(f"   kept {_safe_path_display(path)} (no change)")
            print()
            continue

        # Relocate path: move to models.d/.
        target = MODELS_D / path.name
        print(
            f"{C.BOLD}{path.name}{C.END}: move to "
            f"{_safe_path_display(target)}"
        )
        if target.exists():
            warn(
                f"{_safe_path_display(target)} already exists. Refusing "
                "to overwrite — resolve manually (compare contents, "
                "delete the duplicate, re-run)."
            )
            skipped += 1
            print()
            continue
        if not _confirm(f"Move {path.name}?", auto_yes=auto_yes):
            skipped += 1
            dim(f"   kept {_safe_path_display(path)} (no change)")
            print()
            continue

        # mkdir target dir if missing — first migration creates it.
        try:
            MODELS_D.mkdir(mode=0o700, exist_ok=True)
        except OSError as e:
            failed += 1
            warn(f"couldn't create {_safe_path_display(MODELS_D)}: {e}")
            print()
            continue

        try:
            shutil.move(str(path), str(target))
            moved += 1
            ok(f"moved → {_safe_path_display(target)}")
        except OSError as e:
            failed += 1
            warn(f"move failed: {e}")
        print()

    # Summary
    step("migrate-toml summary")
    if moved:
        ok(f"{moved} file(s) moved to models.d/")
    if deleted:
        ok(f"{deleted} redundant file(s) deleted")
    if skipped:
        dim(f"   {skipped} file(s) skipped (kept as-is)")
    if failed:
        warn(f"{failed} file(s) failed — see warnings above")

    return 0

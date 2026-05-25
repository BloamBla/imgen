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

from .._safe import safe_display, safe_path_display
from ..colors import C, dim, info, ok, step, warn
from ..models import BUILTIN_MODELS


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
        ok(f"No legacy {safe_path_display(BACKENDS_D)} — nothing to migrate.")
        return 0
    if BACKENDS_D.is_symlink():
        warn(
            f"{safe_path_display(BACKENDS_D)} is a symlink; refusing to "
            "operate on it (v0.4 IMP-3 cross-uid attack class). Remove "
            "the symlink and re-run, or migrate files manually."
        )
        return 2
    # v0.8.1 LOW-2 closure: mirror the source-side symlink refusal
    # on the DESTINATION. The loader already refuses both (backends.py
    # load_user_backends_dir at line 601), but a migrate run that
    # mkdir's through a same-uid attacker-planted symlink would
    # silently write user TOMLs through the link.
    if MODELS_D.is_symlink():
        warn(
            f"{safe_path_display(MODELS_D)} is a symlink; refusing to "
            "operate on it (v0.4 IMP-3 cross-uid attack class). Remove "
            "the symlink and re-run."
        )
        return 2

    try:
        entries = sorted(BACKENDS_D.iterdir())
    except OSError as e:
        warn(f"Couldn't read {safe_path_display(BACKENDS_D)}: {e}")
        return 2

    tomls = [p for p in entries if p.suffix == ".toml" and p.is_file()]
    if not tomls:
        ok(f"No *.toml files in {safe_path_display(BACKENDS_D)}.")
        return 0

    info(f"Found {len(tomls)} legacy TOML(s):")
    for path in tomls:
        # v0.8.1 LOW-1 closure: wrap path.name/.stem display via
        # safe_display so a hand-crafted filename with embedded ANSI
        # escapes can't corrupt the migration prompt's terminal
        # output. Consistent with the safe_path_display policy used
        # for full paths elsewhere in this command.
        name_safe = safe_display(path.name)
        stem_safe = safe_display(path.stem)
        if path.stem in BUILTIN_MODELS:
            dim(f"   • {name_safe}  (shadows built-in Model {stem_safe})")
        else:
            dim(f"   • {name_safe}")
    print()

    # Per-file dispatch. Built-in-shadowing files get the deletion
    # path (their recipes are already covered by the built-in
    # registry); novel files get the relocate path.
    moved = 0
    deleted = 0
    skipped = 0
    failed = 0

    for path in tomls:
        # v0.8.1 LOW-1: wrap raw .name / .stem via safe_display so any
        # control bytes in the filename render as ``\xNN`` literals
        # rather than escaping into the terminal at the prompt site.
        name_safe = safe_display(path.name)
        stem_safe = safe_display(path.stem)
        if path.stem in BUILTIN_MODELS:
            # Deletion path: built-in covers the same recipe.
            print(
                f"{C.BOLD}{name_safe}{C.END}: v0.8 built-in Model "
                f"{stem_safe} covers the same recipe."
            )
            print(f"   {C.DIM}safe to delete {safe_path_display(path)}{C.END}")
            if _confirm(f"Delete {name_safe}?", auto_yes=auto_yes):
                try:
                    path.unlink()
                    deleted += 1
                    ok(f"deleted {safe_path_display(path)}")
                except OSError as e:
                    failed += 1
                    warn(f"couldn't delete {safe_path_display(path)}: {e}")
            else:
                skipped += 1
                dim(f"   kept {safe_path_display(path)} (no change)")
            print()
            continue

        # Relocate path: move to models.d/.
        target = MODELS_D / path.name
        print(
            f"{C.BOLD}{name_safe}{C.END}: move to "
            f"{safe_path_display(target)}"
        )
        if target.exists():
            warn(
                f"{safe_path_display(target)} already exists. Refusing "
                "to overwrite — resolve manually (compare contents, "
                "delete the duplicate, re-run)."
            )
            skipped += 1
            print()
            continue
        if not _confirm(f"Move {name_safe}?", auto_yes=auto_yes):
            skipped += 1
            dim(f"   kept {safe_path_display(path)} (no change)")
            print()
            continue

        # mkdir target dir if missing — first migration creates it.
        # NB: LOW-2 symlink guard on MODELS_D fires earlier (before
        # the per-file loop), so the mkdir here only ever creates a
        # real directory or no-ops when the dir already exists.
        try:
            MODELS_D.mkdir(mode=0o700, exist_ok=True)
        except OSError as e:
            failed += 1
            warn(f"couldn't create {safe_path_display(MODELS_D)}: {e}")
            print()
            continue

        try:
            shutil.move(str(path), str(target))
            moved += 1
            ok(f"moved → {safe_path_display(target)}")
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

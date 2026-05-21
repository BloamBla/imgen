"""`imgen clean [--all]` — clear stale HF partial downloads, optionally
purge cached models with confirmation. Symlink-safe.
"""
from __future__ import annotations

import datetime
import shutil

from ..colors import C, dim, info, ok, step, warn
from ..paths import HF_CACHE, LOG_RETENTION_DAYS, LOGS_DIR


def _prune_old_batch_logs(args) -> None:
    """Delete ~/.imgen/logs/*.log older than LOG_RETENTION_DAYS.

    Quiet when there's nothing to do (matches the .incomplete pattern).
    Respects --dry-run (only counts, doesn't delete).
    """
    if not LOGS_DIR.exists():
        return
    cutoff = datetime.datetime.now().timestamp() - LOG_RETENTION_DAYS * 86400
    removed = 0
    removed_size = 0
    for log in LOGS_DIR.glob("*.log"):
        try:
            if log.stat().st_mtime < cutoff:
                removed_size += log.stat().st_size
                if not getattr(args, "dry_run", False):
                    log.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        verb = "Would remove" if getattr(args, "dry_run", False) else "Removed"
        ok(f"{verb} {removed} old batch log(s) older than "
           f"{LOG_RETENTION_DAYS} days ({removed_size / 1024:.1f} KB)")


def cmd_clean(args) -> int:
    step("Cleaning HuggingFace cache")
    print()

    if not HF_CACHE.exists():
        ok("Cache is empty")
        # Even if no HF cache, still prune old batch logs.
        _prune_old_batch_logs(args)
        return 0

    # Always: delete .incomplete files older than 24h
    cutoff = datetime.datetime.now().timestamp() - 24 * 3600
    incomplete_removed = 0
    incomplete_size = 0
    for blob in HF_CACHE.rglob("*.incomplete"):
        try:
            if blob.stat().st_mtime < cutoff:
                incomplete_size += blob.stat().st_size
                blob.unlink()
                incomplete_removed += 1
        except OSError:
            pass
    if incomplete_removed:
        ok(f"Removed {incomplete_removed} stale .incomplete files "
           f"({incomplete_size / (1024**3):.1f} GB)")
    else:
        dim("No stale .incomplete files to remove")

    # Also prune old per-batch log files (~/.imgen/logs/*.log > 30 days).
    _prune_old_batch_logs(args)

    if args.all:
        models = sorted(HF_CACHE.glob("models--*"))
        if not models:
            ok("No cached models to remove")
            return 0
        print()
        info("Cached models to be deleted:")
        total = 0
        for m in models:
            try:
                blobs = m / "blobs"
                if blobs.exists():
                    size = sum(p.stat().st_size for p in blobs.iterdir()
                               if p.is_file() and not p.is_symlink())
                else:
                    size = sum(p.lstat().st_size for p in m.rglob("*")
                               if p.is_file() and not p.is_symlink())
                total += size
                name = m.name.replace("models--", "").replace("--", "/")
                print(f"   • {name}: {size / (1024**3):.1f} GB")
            except OSError:
                pass
        print(f"   {C.BOLD}Total: {total / (1024**3):.1f} GB{C.END}")
        print()
        if args.dry_run:
            warn("Dry run — nothing deleted")
            return 0
        try:
            confirm = input("   Delete all? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            confirm = ""
        if confirm != "y":
            dim("   Cancelled")
            return 0
        # Don't follow a symlinked model dir — its target may be outside
        # HF_CACHE (e.g. shared models tree on another volume). rmtree would
        # silently delete the target. Internal symlinks (HF's snapshots/ →
        # blobs/) stay safe — rmtree unlinks them, doesn't follow.
        cleanup_errors: list[tuple[str, BaseException]] = []

        def _log_rmtree_error(func, path, exc):
            cleanup_errors.append((str(path), exc))

        deleted = 0
        skipped_symlinks = []
        for m in models:
            if m.is_symlink():
                skipped_symlinks.append(m)
                continue
            if not m.is_dir():
                continue
            shutil.rmtree(m, onexc=_log_rmtree_error)
            deleted += 1

        if skipped_symlinks:
            warn(f"Skipped {len(skipped_symlinks)} symlinked model dir(s) "
                 f"— refusing to follow:")
            for s in skipped_symlinks:
                print(f"   {C.DIM}• {s}{C.END}")
        if cleanup_errors:
            warn(f"{len(cleanup_errors)} error(s) during deletion:")
            for path, exc in cleanup_errors[:5]:
                print(f"   {C.DIM}• {path}: {exc}{C.END}")
            if len(cleanup_errors) > 5:
                more = len(cleanup_errors) - 5
                print(f"   {C.DIM}  ... and {more} more{C.END}")

        ok(f"Deleted {deleted} model(s) "
           f"({total / (1024**3):.1f} GB freed)")

    return 0

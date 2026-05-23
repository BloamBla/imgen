"""Pure-function tests for the v0.5 ``check_enhance_health`` doctor
helper. Mirrors test_doctor_backends.py for ``check_backend_health``.

The function is pure-ish (one optional import probe, otherwise just
stat'ing HF cache directories). Tests inject:

* ``importable=True/False`` to skip the real mlx_lm import
* an empty/fake HF cache directory (tmp_path) so we never touch the
  real ~/.cache
* a synthetic history list so the recent-runs counter is deterministic

End-to-end exercise of the cmd_doctor printer is left to the existing
``test_doctor_alias.py`` / ``test_doctor_backends.py`` patterns —
this file pins the pure data layer only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.commands.doctor import (
    EnhanceHealth,
    _dir_size_bytes,
    check_enhance_health,
)
from imgen.hf_cache import hf_cache_dir_for


# ── hf_cache_dir_for ───────────────────────────────────────────────────
#
# v0.6.4 (architect v0.6.2 NIT-2): historically imported via the private
# alias ``doctor._hf_cache_dir_for``. After hf_cache.py extraction the
# canonical public name is ``imgen.hf_cache.hf_cache_dir_for``; the
# doctor/parser private aliases were retired alongside this test update.


class TestHFCacheDirFor:
    def test_standard_hf_repo(self, tmp_path):
        result = hf_cache_dir_for(
            "mlx-community/Qwen2.5-7B-Instruct-4bit", tmp_path,
        )
        assert result == tmp_path / "models--mlx-community--Qwen2.5-7B-Instruct-4bit"

    def test_three_slash_path(self, tmp_path):
        """org/sub-name/variant — only the first slash becomes '--' on
        HF's side (snapshot), but huggingface_hub's actual layout uses
        '--' for every '/' in the repo id. Lock-in test."""
        result = hf_cache_dir_for("a/b/c", tmp_path)
        assert result == tmp_path / "models--a--b--c"

    def test_absolute_local_path_passes_through(self, tmp_path):
        """User points --enhance-model at a local checkpoint dir →
        treat the path as-is, don't slap a models-- prefix on."""
        abs_path = "/abs/path/to/local_model"
        result = hf_cache_dir_for(abs_path, tmp_path)
        assert result == Path(abs_path)

    def test_empty_string_returns_cache_root(self, tmp_path):
        # Defensive: if config is empty, fall back to cache root rather
        # than create a weird `models--` directory.
        result = hf_cache_dir_for("", tmp_path)
        assert result == tmp_path


# ── _dir_size_bytes ────────────────────────────────────────────────────


class TestDirSizeBytes:
    def test_missing_dir_returns_zero(self, tmp_path):
        assert _dir_size_bytes(tmp_path / "nonexistent") == 0

    def test_path_is_file_returns_zero(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        # Function takes a directory; a file path returns 0 (defensive).
        assert _dir_size_bytes(f) == 0

    def test_empty_dir_returns_zero(self, tmp_path):
        assert _dir_size_bytes(tmp_path) == 0

    def test_sums_file_sizes_recursively(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.bin").write_bytes(b"y" * 200)
        (sub / "deeper").mkdir()
        (sub / "deeper" / "c.bin").write_bytes(b"z" * 50)
        assert _dir_size_bytes(tmp_path) == 350

    def test_does_not_follow_symlinks(self, tmp_path):
        """HF cache uses snapshots/<sha>/* → blobs/* symlinks. If we
        followed them we'd double-count weights. Defensive test."""
        target = tmp_path / "target.bin"
        target.write_bytes(b"x" * 1000)
        link = tmp_path / "link.bin"
        link.symlink_to(target)
        # Only the real file counts; the symlink is skipped.
        assert _dir_size_bytes(tmp_path) == 1000


# ── check_enhance_health ──────────────────────────────────────────────


def _make_fake_hf_cache(tmp_path: Path, model_size_bytes: int) -> Path:
    """Create a fake HF cache layout for the default Qwen model so
    check_enhance_health treats it as "cached"."""
    cache = tmp_path / "hf_cache"
    cache.mkdir()
    model_dir = (cache
                 / "models--mlx-community--Qwen2.5-7B-Instruct-4bit")
    (model_dir / "blobs").mkdir(parents=True)
    (model_dir / "blobs" / "weights.safetensors").write_bytes(
        b"x" * model_size_bytes
    )
    return cache


class TestCheckEnhanceHealth:
    def test_mlx_lm_not_importable(self, tmp_path):
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=[],
            importable=False,
        )
        assert isinstance(result, EnhanceHealth)
        assert result.mlx_lm_importable is False

    def test_default_config_uses_qwen_2_5_7b(self, tmp_path):
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=[],
            importable=True,
        )
        # Module default — see config._ENHANCE_MODULE_DEFAULTS.
        assert result.model_ref == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        # Default is opt-in (False).
        assert result.enabled_by_default is False

    def test_config_default_true_reflected_in_enabled_by_default(self, tmp_path):
        result = check_enhance_health(
            enhance_cfg={"default": True}, hf_cache=tmp_path, history=[],
            importable=True,
        )
        assert result.enabled_by_default is True

    def test_config_model_override_used(self, tmp_path):
        result = check_enhance_health(
            enhance_cfg={"model": "Qwen/Qwen2.5-3B-Instruct"},
            hf_cache=tmp_path, history=[],
            importable=True,
        )
        assert result.model_ref == "Qwen/Qwen2.5-3B-Instruct"

    def test_model_not_cached_when_dir_missing(self, tmp_path):
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=[],
            importable=True,
        )
        assert result.model_cached is False
        assert result.model_cache_size_bytes is None

    def test_model_cached_when_weights_present(self, tmp_path):
        cache = _make_fake_hf_cache(tmp_path, model_size_bytes=4_000_000_000)
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=cache, history=[],
            importable=True,
        )
        assert result.model_cached is True
        assert result.model_cache_size_bytes == 4_000_000_000

    def test_tiny_cache_treated_as_not_cached(self, tmp_path):
        """A 1 KB cache dir means the download was interrupted before
        the weights landed (config.json + tokenizer files are tiny).
        Report as not-cached so the doctor surfaces the pending
        download for the user."""
        cache = _make_fake_hf_cache(tmp_path, model_size_bytes=1024)
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=cache, history=[],
            importable=True,
        )
        assert result.model_cached is False
        assert result.model_cache_size_bytes is None

    def test_recent_runs_counts_only_actual_enhance_attempts(self, tmp_path):
        """Recent-runs window counts only entries where the user
        actually attempted enhancement (--enhance-prompt). Pre-v0.5
        entries (no `enhanced` field) are skipped; user_opt_out
        entries (intentional --no-enhance) are also skipped — those
        aren't failed attempts, they're "the user explicitly said no"
        and shouldn't drag the success rate down."""
        history = [
            # Pre-v0.5 entry — no `enhanced` field; skipped.
            {"input": "/a.jpg", "prompt": "x"},
            # v0.5 entry, enhanced=True (actual success).
            {"input": "/b.jpg", "prompt": "ENH: x", "enhanced": True,
             "enhance_model": "Qwen/...", "enhance_fallback_reason": None},
            # v0.5 entry, runner-level failure (real attempt that failed).
            {"input": "/c.jpg", "prompt": "x", "enhanced": False,
             "enhance_model": None,
             "enhance_fallback_reason": "invariant_violated"},
            # v0.5 entry, user_opt_out (--no-enhance) — NOT an attempt.
            {"input": "/d.jpg", "prompt": "x", "enhanced": False,
             "enhance_model": None,
             "enhance_fallback_reason": "user_opt_out"},
        ]
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=history,
            importable=True,
        )
        # Only the 2 real-attempt entries count (b succeeded, c failed).
        assert result.recent_runs == 2
        assert result.recent_runs_succeeded == 1


    def test_user_opt_out_entries_alone_yield_zero_recent_runs(self, tmp_path):
        """A user who has only ever run --no-enhance shouldn't see a
        misleading "X% success rate" warning. recent_runs should be 0
        → doctor printer skips the warning entirely."""
        history = [
            {"enhanced": False, "enhance_fallback_reason": "user_opt_out"}
            for _ in range(5)
        ]
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=history,
            importable=True,
        )
        assert result.recent_runs == 0
        assert result.recent_runs_succeeded == 0

    def test_recent_runs_window_is_last_10(self, tmp_path):
        """Window is the last 10 history entries — older runs don't
        skew the success rate."""
        # 15 entries: first 5 are old failures, last 10 are recent
        # successes. Window slices the last 10 → 10 successes, 100%.
        history = (
            [
                {"enhanced": False, "enhance_fallback_reason": "runner_error"}
                for _ in range(5)
            ]
            + [
                {"enhanced": True, "enhance_fallback_reason": None}
                for _ in range(10)
            ]
        )
        result = check_enhance_health(
            enhance_cfg={}, hf_cache=tmp_path, history=history,
            importable=True,
        )
        # Only the last 10 — all successes.
        assert result.recent_runs == 10
        assert result.recent_runs_succeeded == 10


# ── EnhanceHealth dataclass shape ─────────────────────────────────────


def test_enhance_health_is_frozen_with_slots():
    eh = EnhanceHealth(
        mlx_lm_importable=True,
        enabled_by_default=False,
        model_ref="x",
        model_cached=False,
        model_cache_size_bytes=None,
        recent_runs=0,
        recent_runs_succeeded=0,
    )
    # frozen — can't mutate after construction.
    with pytest.raises((AttributeError, TypeError)):
        eh.model_ref = "y"  # type: ignore[misc]
    # slots — typo on attribute set is rejected.
    with pytest.raises((AttributeError, TypeError)):
        eh.nonexistent_typo = 1  # type: ignore[attr-defined]

"""v0.10.0 commit 9 — `--lora <bare-name>` local resolver.

Extends ``parser._lora_ref_arg`` so a trained LoRA can be referenced
by its bare name (``--lora alina``) instead of the full
``~/.imgen/loras/alina.safetensors`` path.

Per [[project-v100-design]] §H.2 + §R.1 ROUND-1 CLOSURES:

Resolution order (locked):
  1. Absolute path (starts with ``/``) — used as-is.
  2. Has ``/`` separator (``author/repo``) — HF repo id, used as-is.
  3. Bare slug ``[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?`` (1-32 chars) —
     probe ``~/.imgen/loras/<slug>.safetensors`` via ``lstat`` +
     ``stat.S_ISREG`` (security H-5: NOT ``is_file`` which follows
     symlinks). Exists → resolve to absolute path. Miss → REJECT
     with a clear error (§R.1 M-4: cleaner than HF fall-through).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.parser import _lora_ref_arg


@pytest.fixture
def loras_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR so ``~/.imgen/loras/`` lands in tmp_path."""
    fake_state = tmp_path / "state"
    (fake_state / "loras").mkdir(parents=True)
    from imgen import paths
    monkeypatch.setattr(paths, "STATE_DIR", fake_state)
    return fake_state / "loras"


def _make_local_lora(loras_dir: Path, name: str) -> Path:
    p = loras_dir / f"{name}.safetensors"
    p.write_bytes(b"fake-lora-weights")
    return p


# ── bare-name → local path ───────────────────────────────────────


class TestBareNameResolvesLocal:
    def test_existing_local_lora_resolves_to_abs_path(self, loras_dir):
        _make_local_lora(loras_dir, "alina")
        ref = _lora_ref_arg("alina")
        assert ref.ref == str(loras_dir / "alina.safetensors")
        assert ref.weight == 1.0

    def test_bare_name_with_weight_resolves_path_and_weight(self, loras_dir):
        _make_local_lora(loras_dir, "alina")
        ref = _lora_ref_arg("alina:0.8")
        assert ref.ref == str(loras_dir / "alina.safetensors")
        assert ref.weight == 0.8

    def test_underscore_and_dash_slugs_resolve(self, loras_dir):
        _make_local_lora(loras_dir, "my_face-2025")
        ref = _lora_ref_arg("my_face-2025")
        assert ref.ref == str(loras_dir / "my_face-2025.safetensors")


# ── bare-name miss → reject (M-4) ────────────────────────────────


class TestBareNameMissRejects:
    def test_missing_local_lora_rejects(self, loras_dir):
        """§R.1 M-4: bare slug w/o local match + no '/' → reject with
        a clear error, NOT fall through to an HF fetch that would
        fail confusingly."""
        import argparse
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _lora_ref_arg("nonexistent")
        msg = str(exc.value).lower()
        assert "nonexistent" in msg
        # Error should point the user at training or HF-id format.
        assert "train" in msg or "author/name" in msg or "hf" in msg

    def test_invalid_slug_grammar_no_slash_rejects(self, loras_dir):
        """Uppercase isn't a valid slug AND has no '/' → can't be a
        local LoRA nor an HF id. Reject."""
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_ref_arg("MyLora")


# ── security H-5: symlink + non-regular-file ─────────────────────


class TestBareNameSecurityH5:
    def test_symlink_at_local_path_rejected(self, loras_dir, tmp_path):
        """Security H-5: lstat + S_ISREG, NOT is_file. A symlink at
        the candidate path must NOT resolve (an attacker with write
        access to ~/.imgen/loras/ could otherwise point a symlink at
        an arbitrary file)."""
        import argparse
        real = tmp_path / "elsewhere.safetensors"
        real.write_bytes(b"x")
        (loras_dir / "alina.safetensors").symlink_to(real)
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_ref_arg("alina")

    def test_directory_at_local_path_rejected(self, loras_dir):
        """A directory named ``alina.safetensors`` is not a regular
        file → reject (don't pass a dir path to mflux --lora-paths)."""
        import argparse
        (loras_dir / "alina.safetensors").mkdir()
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_ref_arg("alina")


# ── existing behaviour preserved ─────────────────────────────────


class TestResolutionOrderUnchangedPaths:
    def test_hf_repo_id_unchanged(self, loras_dir):
        """HF ids (have '/') resolve as-is — never probed locally."""
        ref = _lora_ref_arg("alvarobartt/ghibli-characters-flux-lora")
        assert ref.ref == "alvarobartt/ghibli-characters-flux-lora"

    def test_hf_repo_id_with_weight_unchanged(self, loras_dir):
        ref = _lora_ref_arg("strangerzonehf/Flux-Animeo-v1-LoRA:0.8")
        assert ref.ref == "strangerzonehf/Flux-Animeo-v1-LoRA"
        assert ref.weight == 0.8

    def test_absolute_path_unchanged(self, loras_dir):
        ref = _lora_ref_arg("/Users/me/loras/sketch.safetensors")
        assert ref.ref == "/Users/me/loras/sketch.safetensors"

    def test_absolute_path_with_colon_suffix_unchanged(self, loras_dir):
        """Absolute paths keep the v0.6 IMP-1 no-weight-split posture —
        ``:2024`` is part of the path, not a weight."""
        ref = _lora_ref_arg("/Volumes/.timemachine/disk:2024")
        assert ref.ref == "/Volumes/.timemachine/disk:2024"

    def test_hf_id_does_not_probe_local_even_if_basename_matches(
        self, loras_dir,
    ):
        """An HF id ``author/alina`` must NOT resolve to a local
        ``alina.safetensors`` — the '/' forces HF interpretation."""
        _make_local_lora(loras_dir, "alina")
        ref = _lora_ref_arg("author/alina")
        assert ref.ref == "author/alina"

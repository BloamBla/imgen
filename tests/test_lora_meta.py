"""v0.10.0 commit 10 — trained-LoRA meta.json reader + trigger prepend.

Two layers:

1. :func:`imgen.lora_meta.read_lora_meta` — best-effort reader for the
   ``<name>.meta.json`` sidecar written by ``imgen train`` (commit 6).
   Returns ``(trigger, compat_group)``; any miss / corruption / cap
   violation degrades to ``(None, None)`` (or drops just the bad field)
   so a malformed sidecar never breaks an ``imgen draw`` run.

2. Integration: ``parser._lora_ref_arg`` populates
   ``LoraRef.trigger`` + ``compatible_with`` from the sidecar at
   resolution time, so the EXISTING
   ``build_iteration.prepend_trigger_words`` (word-boundary + dedup)
   auto-prepends the trigger and the EXISTING compat-filter keeps the
   trained LoRA on its base model.

Per [[project-v100-design]] §I + §H.3 + §R.1 ROUND-1 CLOSURES:
* 16 KB cap on meta read (security C-3).
* Trigger length 1..64 re-validated on READ (python H-8).
* Control-byte filter on trigger (defence-in-depth).
* compat-group from meta closes the commit-9 deferral (a klein-4b
  LoRA carries compatible_with=("flux2-klein-4b",) so it isn't
  dropped by filter_compatible_loras under --model flux2-klein-4b).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from imgen.lora_meta import read_lora_meta, scan_trained_loras


def _write_meta(loras_dir: Path, name: str, **overrides) -> Path:
    """Write a <name>.meta.json + a sibling .safetensors. Returns the
    .safetensors path (the read_lora_meta input)."""
    safetensors = loras_dir / f"{name}.safetensors"
    safetensors.write_bytes(b"weights")
    meta = {
        "version": 1,
        "lora_name": name,
        "trigger": "al1na woman",
        "lora_compat_group": "flux2-klein-4b",
        "base_model": "flux2-klein-4b",
    }
    meta.update(overrides)
    (loras_dir / f"{name}.meta.json").write_text(
        json.dumps(meta), encoding="utf-8",
    )
    return safetensors


@pytest.fixture
def loras_dir(tmp_path):
    d = tmp_path / "loras"
    d.mkdir()
    return d


# ── read_lora_meta happy path ────────────────────────────────────


class TestReadLoraMetaHappyPath:
    def test_reads_trigger_and_compat(self, loras_dir):
        st = _write_meta(loras_dir, "alina")
        trigger, compat = read_lora_meta(st)
        assert trigger == "al1na woman"
        assert compat == "flux2-klein-4b"

    def test_strips_trigger_whitespace(self, loras_dir):
        st = _write_meta(loras_dir, "alina", trigger="  ohwx man  ")
        trigger, _ = read_lora_meta(st)
        assert trigger == "ohwx man"


# ── read_lora_meta miss / corruption (degrades to None) ──────────


class TestReadLoraMetaDegrades:
    def test_missing_sidecar_returns_none_none(self, loras_dir):
        st = loras_dir / "nometa.safetensors"
        st.write_bytes(b"weights")
        assert read_lora_meta(st) == (None, None)

    def test_corrupt_json_returns_none_none(self, loras_dir):
        st = loras_dir / "alina.safetensors"
        st.write_bytes(b"weights")
        (loras_dir / "alina.meta.json").write_text("{not valid json")
        assert read_lora_meta(st) == (None, None)

    def test_oversized_meta_returns_none_none(self, loras_dir):
        """Security C-3: 16 KB cap. A meta.json above the cap is a
        DoS / tampering signal → skip entirely."""
        st = loras_dir / "alina.safetensors"
        st.write_bytes(b"weights")
        bloated = {"trigger": "x", "lora_compat_group": "g",
                   "junk": "A" * 20000}
        (loras_dir / "alina.meta.json").write_text(json.dumps(bloated))
        assert read_lora_meta(st) == (None, None)

    def test_invalid_utf8_returns_none_none(self, loras_dir):
        st = loras_dir / "alina.safetensors"
        st.write_bytes(b"weights")
        (loras_dir / "alina.meta.json").write_bytes(b"\xff\xfe not utf8")
        assert read_lora_meta(st) == (None, None)


# ── trigger field validation on READ (python H-8) ────────────────


class TestReadLoraMetaTriggerValidation:
    def test_trigger_over_64_chars_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "alina", trigger="x" * 65)
        trigger, compat = read_lora_meta(st)
        assert trigger is None
        # compat still returned — only the bad field drops.
        assert compat == "flux2-klein-4b"

    def test_empty_trigger_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "alina", trigger="")
        trigger, compat = read_lora_meta(st)
        assert trigger is None
        assert compat == "flux2-klein-4b"

    def test_control_byte_trigger_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "alina", trigger="al1na\x1b[2J")
        trigger, compat = read_lora_meta(st)
        assert trigger is None
        assert compat == "flux2-klein-4b"

    def test_bidi_override_trigger_dropped(self, loras_dir):
        """§R.3 security MEDIUM: read-side re-validation must mirror the
        write-side _trigger_token_arg Cf rejection — a bidi-override
        (U+202E) in a hand-edited sidecar must not reach the prompt."""
        st = _write_meta(loras_dir, "alina", trigger="al1na‮woman")
        trigger, compat = read_lora_meta(st)
        assert trigger is None
        assert compat == "flux2-klein-4b"

    def test_zero_width_trigger_dropped(self, loras_dir):
        """Zero-width space (U+200B, category Cf) dropped on read."""
        st = _write_meta(loras_dir, "alina", trigger="al1na​woman")
        trigger, _ = read_lora_meta(st)
        assert trigger is None

    def test_combining_mark_trigger_dropped(self, loras_dir):
        """Nonspacing combining mark (U+0301, category Mn) dropped on
        read — matches the write-side Mn rejection."""
        st = _write_meta(loras_dir, "alina", trigger="al1náwoman")
        trigger, _ = read_lora_meta(st)
        assert trigger is None

    def test_non_string_trigger_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "alina", trigger=12345)
        trigger, _ = read_lora_meta(st)
        assert trigger is None

    def test_non_string_compat_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "alina", lora_compat_group=999)
        _, compat = read_lora_meta(st)
        assert compat is None

    def test_missing_compat_field_returns_none_for_compat(self, loras_dir):
        st = loras_dir / "alina.safetensors"
        st.write_bytes(b"weights")
        (loras_dir / "alina.meta.json").write_text(
            json.dumps({"trigger": "ohwx man"}),
        )
        trigger, compat = read_lora_meta(st)
        assert trigger == "ohwx man"
        assert compat is None


# ── parser integration: LoraRef gets trigger + compat ────────────


class TestLoraRefArgPopulatesFromMeta:
    @pytest.fixture
    def state_loras(self, tmp_path, monkeypatch):
        fake_state = tmp_path / "state"
        (fake_state / "loras").mkdir(parents=True)
        from imgen import paths
        monkeypatch.setattr(paths, "STATE_DIR", fake_state)
        return fake_state / "loras"

    def test_bare_name_populates_trigger(self, state_loras):
        _write_meta(state_loras, "alina")
        from imgen.parser import _lora_ref_arg
        ref = _lora_ref_arg("alina")
        assert ref.trigger == "al1na woman"

    def test_bare_name_overrides_compatible_with(self, state_loras):
        _write_meta(state_loras, "alina")
        from imgen.parser import _lora_ref_arg
        ref = _lora_ref_arg("alina")
        # §R.1: trained LoRA carries its own compat group, not the
        # broad CLI default — so filter_compatible_loras keeps it on
        # --model flux2-klein-4b.
        assert ref.compatible_with == ("flux2-klein-4b",)

    def test_bare_name_without_meta_keeps_default_compat(
        self, state_loras,
    ):
        """A local .safetensors with NO sidecar (placed manually) →
        trigger None, compatible_with falls back to the broad CLI
        default."""
        (state_loras / "manual.safetensors").write_bytes(b"weights")
        from imgen.parser import _lora_ref_arg
        ref = _lora_ref_arg("manual")
        assert ref.trigger is None
        assert ref.compatible_with == ("flux-1", "flux-dev")

    def test_hf_id_does_not_read_meta(self, state_loras):
        """HF ids never touch the local meta path — trigger stays None,
        compat stays the broad default."""
        from imgen.parser import _lora_ref_arg
        ref = _lora_ref_arg("author/some-lora")
        assert ref.trigger is None
        assert ref.compatible_with == ("flux-1", "flux-dev")


# ── end-to-end via prepend_trigger_words ─────────────────────────


class TestTriggerPrependEndToEnd:
    @pytest.fixture
    def state_loras(self, tmp_path, monkeypatch):
        fake_state = tmp_path / "state"
        (fake_state / "loras").mkdir(parents=True)
        from imgen import paths
        monkeypatch.setattr(paths, "STATE_DIR", fake_state)
        return fake_state / "loras"

    def test_trigger_prepended_when_absent(self, state_loras):
        _write_meta(state_loras, "alina", trigger="al1na woman")
        from imgen.parser import _lora_ref_arg
        from imgen.build_iteration import prepend_trigger_words
        ref = _lora_ref_arg("alina")
        out = prepend_trigger_words("in a samurai outfit", (ref,))
        assert out.startswith("al1na woman")
        assert "in a samurai outfit" in out

    def test_trigger_not_double_prepended(self, state_loras):
        """Word-boundary guard (python H-6): if the trigger is already
        in the prompt, don't prepend again."""
        _write_meta(state_loras, "alina", trigger="al1na woman")
        from imgen.parser import _lora_ref_arg
        from imgen.build_iteration import prepend_trigger_words
        ref = _lora_ref_arg("alina")
        out = prepend_trigger_words(
            "al1na woman in a samurai outfit", (ref,),
        )
        # Only one occurrence — not double-prepended.
        assert out.lower().count("al1na woman") == 1

    def test_multi_lora_triggers_dedup(self, state_loras):
        """python H-7: two LoRAs sharing a trigger don't double-prepend."""
        _write_meta(state_loras, "alina", trigger="al1na woman")
        _write_meta(state_loras, "alina2", trigger="al1na woman")
        from imgen.parser import _lora_ref_arg
        from imgen.build_iteration import prepend_trigger_words
        r1 = _lora_ref_arg("alina")
        r2 = _lora_ref_arg("alina2")
        out = prepend_trigger_words("in a park", (r1, r2))
        assert out.lower().count("al1na woman") == 1


# ── scan_trained_loras (P1: --list-loras discovery of trained LoRAs) ─


class TestScanTrainedLoras:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert scan_trained_loras(tmp_path / "does-not-exist") == []

    def test_empty_dir_returns_empty(self, loras_dir):
        assert scan_trained_loras(loras_dir) == []

    def test_lists_trained_lora_with_trigger_and_group(self, loras_dir):
        _write_meta(loras_dir, "stas", trigger="stas man",
                    lora_compat_group="flux2-klein-4b")
        assert scan_trained_loras(loras_dir) == [
            ("stas", "stas man", "flux2-klein-4b"),
        ]

    def test_bare_safetensors_without_meta_still_listed(self, loras_dir):
        """A .safetensors with no/corrupt sidecar must still appear (so
        the user sees the file) — trigger + group degrade to None."""
        (loras_dir / "orphan.safetensors").write_bytes(b"weights")
        assert scan_trained_loras(loras_dir) == [("orphan", None, None)]

    def test_sorted_by_name(self, loras_dir):
        _write_meta(loras_dir, "zeta", trigger="z man")
        _write_meta(loras_dir, "alpha", trigger="a man")
        names = [n for n, _, _ in scan_trained_loras(loras_dir)]
        assert names == ["alpha", "zeta"]

    def test_ignores_non_safetensors(self, loras_dir):
        """meta.json sidecars and stray files are not LoRA entries."""
        _write_meta(loras_dir, "stas", trigger="stas man")
        (loras_dir / "notes.txt").write_text("hi")
        names = [n for n, _, _ in scan_trained_loras(loras_dir)]
        assert names == ["stas"]

    def test_control_byte_filename_skipped(self, loras_dir):
        """security: a filename with terminal-control bytes prints raw to
        stdout via --list-loras — it must be filtered from discovery,
        consistent with the trigger sanitization."""
        (loras_dir / "evil\x1b[2J.safetensors").write_bytes(b"weights")
        _write_meta(loras_dir, "stas", trigger="stas man")
        names = [n for n, _, _ in scan_trained_loras(loras_dir)]
        assert names == ["stas"]  # the control-byte file is dropped


class TestCompatGroupValidation:
    """security LOW: lora_compat_group also reaches stdout, so it's
    length-capped + control-byte-filtered on read like the trigger."""

    def test_oversized_compat_group_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "x", lora_compat_group="g" * 100)
        _, compat = read_lora_meta(st)
        assert compat is None

    def test_control_byte_compat_group_dropped(self, loras_dir):
        st = _write_meta(loras_dir, "x", lora_compat_group="flux\x07evil")
        _, compat = read_lora_meta(st)
        assert compat is None

    def test_valid_compat_group_kept(self, loras_dir):
        st = _write_meta(loras_dir, "x", lora_compat_group="flux2-klein-4b")
        _, compat = read_lora_meta(st)
        assert compat == "flux2-klein-4b"

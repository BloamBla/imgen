"""v0.6 Phase 2A — CLI flags for the LoRA stack.

* ``--lora REF[:WEIGHT]`` — repeatable; argparse's ``action='append'``
  collects each into an ``args.lora`` list of :class:`LoraRef`.
* ``--no-lora`` — mutex with ``--lora``; drops style-declared LoRAs
  for this run.
* Cross-cutting: both flags on generate + batch subcommands; both
  defended against control bytes + oversized refs + out-of-range
  weights at parse time.
"""
from __future__ import annotations

import argparse

import pytest

from imgen.parser import _lora_ref_arg, build_parser
from imgen.styles import LoraRef


_DUMMY_DEFAULTS = {
    "style": "pixar", "backend": "flux", "quantize": 8, "steps": 20,
    "guidance": 3.5, "strength": 0.55, "mlx_cache_gb": 12, "battery_stop": 20,
}


def _parse(*argv: str):
    return build_parser(_DUMMY_DEFAULTS).parse_args(argv)


# ── _lora_ref_arg validator (unit-level) ──────────────────────────────


class TestLoraRefArg:
    def test_bare_ref_default_weight_1(self):
        ref = _lora_ref_arg("alvarobartt/ghibli-characters-flux-lora")
        assert isinstance(ref, LoraRef)
        assert ref.ref == "alvarobartt/ghibli-characters-flux-lora"
        assert ref.weight == 1.0
        assert ref.compatible_with == ("flux-1",)

    def test_ref_with_explicit_weight(self):
        ref = _lora_ref_arg("strangerzonehf/Flux-Animeo-v1-LoRA:0.8")
        assert ref.ref == "strangerzonehf/Flux-Animeo-v1-LoRA"
        assert ref.weight == 0.8

    def test_ref_with_integer_weight(self):
        ref = _lora_ref_arg("org/model:1")
        assert ref.weight == 1.0  # integer parses as float

    def test_ref_with_negative_weight(self):
        ref = _lora_ref_arg("org/model:-0.5")
        assert ref.ref == "org/model"
        assert ref.weight == -0.5

    def test_ref_without_weight_when_colon_followed_by_non_number(self):
        """A colon followed by a non-numeric tail is treated as part of
        the ref, not as a weight separator. (Real HF repo ids never
        contain colons, but absolute paths might.)"""
        ref = _lora_ref_arg("org/model:variant-a")
        # Rightmost ``:`` followed by non-float → whole string is ref.
        assert ref.ref == "org/model:variant-a"
        assert ref.weight == 1.0

    def test_empty_value_rejected(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
            _lora_ref_arg("   ")

    def test_ref_with_control_bytes_rejected(self):
        """Control bytes (C0/DEL/C1) reach mflux argv → log files →
        terminal display. Same defence as scene_suffix + enhance_model.
        """
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="control bytes"):
            _lora_ref_arg("evil\x1b[2J/model:0.8")

    def test_weight_out_of_range_rejected(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="weight out of range"):
            _lora_ref_arg("org/model:3.0")
        with pytest.raises(argparse.ArgumentTypeError, match="weight out of range"):
            _lora_ref_arg("org/model:-2.5")

    def test_oversized_ref_rejected(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError, match="too long"):
            _lora_ref_arg("x" * 5000)

    def test_weight_boundary_values_accepted(self):
        for boundary in (-2.0, -1.0, 0.0, 1.0, 2.0):
            ref = _lora_ref_arg(f"org/model:{boundary}")
            assert ref.weight == boundary

    # ── v0.6 security-reviewer IMP-1: flag-shaped refs rejected ──────

    def test_flag_shaped_ref_rejected(self):
        """``--lora "--config /etc/passwd"`` would land verbatim on
        mflux's argv and confuse its argparser. Reject any ref starting
        with '-' unless it's an absolute path (starts with '/').
        Symmetric with v0.4's `_validate_binary_field` defence in
        backends.py."""
        with pytest.raises(argparse.ArgumentTypeError, match="flag-shaped"):
            _lora_ref_arg("--config")
        with pytest.raises(argparse.ArgumentTypeError, match="flag-shaped"):
            _lora_ref_arg("--lora-paths")
        with pytest.raises(argparse.ArgumentTypeError, match="flag-shaped"):
            _lora_ref_arg("-x")

    def test_absolute_path_starting_with_slash_accepted(self):
        """``/path/to/lora.safetensors`` IS a valid LoRA ref shape (mflux
        accepts absolute paths via --lora-paths). The flag-shape guard
        must not reject these — the leading '/' distinguishes a path
        from a flag."""
        ref = _lora_ref_arg("/Users/me/loras/sketch.safetensors")
        assert ref.ref == "/Users/me/loras/sketch.safetensors"
        assert ref.weight == 1.0

    # ── v0.6 python-reviewer IMP-1: path:NNNN misparse ───────────────

    def test_absolute_path_with_trailing_colon_digits_not_misparsed(self):
        """v0.5 rightmost-colon split would silently strip a path's
        ``:NNNN`` suffix and treat it as weight. e.g.
        ``/Users/me/lora-v1.0:2024`` (timestamped folder) became
        ``ref="/Users/me/lora-v1.0", weight=2024.0`` → out-of-range
        crash. Worse, ``:0.5`` suffix loaded a DIFFERENT file silently.
        v0.6: absolute paths preserve their colons verbatim; weight
        syntax only applies to HF repo ids (which can't contain ':')."""
        # In-range weight suffix that would have silently misparsed.
        ref = _lora_ref_arg("/Users/me/lora-v1.0:0.5")
        assert ref.ref == "/Users/me/lora-v1.0:0.5"
        assert ref.weight == 1.0  # default — no split happened

        # Out-of-range "weight" suffix that v0.5 would have rejected
        # with a confusing "weight out of range" error.
        ref = _lora_ref_arg("/Volumes/.timemachine/disk:2024")
        assert ref.ref == "/Volumes/.timemachine/disk:2024"
        assert ref.weight == 1.0

    def test_hf_repo_with_weight_still_splits(self):
        """HF repo ids never contain ':' — the weight syntax remains
        unambiguous for the HF case. Lock-in test against accidental
        regression of the HF weight-parsing path."""
        ref = _lora_ref_arg("strangerzonehf/Flux-Animeo-v1-LoRA:0.8")
        assert ref.ref == "strangerzonehf/Flux-Animeo-v1-LoRA"
        assert ref.weight == 0.8


# ── --lora flag end-to-end via build_parser ────────────────────────────


class TestLoraFlagOnGenerate:
    def test_no_flag_defaults_to_none(self):
        args = _parse("generate", "photo.jpg", "-s", "anime")
        assert args.lora is None
        assert args.no_lora is False

    def test_single_lora_creates_one_element_list(self):
        args = _parse(
            "generate", "photo.jpg", "-s", "anime",
            "--lora", "alvarobartt/ghibli-characters-flux-lora",
        )
        assert isinstance(args.lora, list)
        assert len(args.lora) == 1
        assert args.lora[0].ref == "alvarobartt/ghibli-characters-flux-lora"
        assert args.lora[0].weight == 1.0

    def test_multiple_lora_flags_append(self):
        """``action='append'`` collects each --lora into the list in
        the order they appeared on the CLI."""
        args = _parse(
            "generate", "photo.jpg", "-s", "anime",
            "--lora", "a/first:0.8",
            "--lora", "b/second:0.4",
            "--lora", "c/third:0.2",
        )
        assert [r.ref for r in args.lora] == [
            "a/first", "b/second", "c/third",
        ]
        assert [r.weight for r in args.lora] == [0.8, 0.4, 0.2]

    def test_no_lora_sets_true(self):
        args = _parse("generate", "photo.jpg", "-s", "anime", "--no-lora")
        assert args.no_lora is True
        assert args.lora is None

    def test_lora_and_no_lora_mutex(self):
        """``--lora REF`` + ``--no-lora`` together is an argparse mutex
        violation — the user means "either add this or drop all", not
        both."""
        with pytest.raises(SystemExit):
            _parse(
                "generate", "photo.jpg", "-s", "anime",
                "--lora", "x/y", "--no-lora",
            )

    def test_invalid_lora_ref_exits(self):
        """Parser validation fires at parse_args time — control bytes,
        empty value, weight out of range all kick out via
        argparse.ArgumentTypeError → SystemExit."""
        with pytest.raises(SystemExit):
            _parse(
                "generate", "photo.jpg", "-s", "anime",
                "--lora", "evil\x1b/model",
            )

    def test_invalid_lora_weight_exits(self):
        with pytest.raises(SystemExit):
            _parse(
                "generate", "photo.jpg", "-s", "anime",
                "--lora", "org/model:5.0",
            )


class TestLoraFlagOnBatch:
    def test_batch_accepts_lora(self):
        args = _parse(
            "batch", "/some/dir", "-s", "anime",
            "--lora", "x/y:0.6",
        )
        assert len(args.lora) == 1
        assert args.lora[0].weight == 0.6

    def test_batch_accepts_no_lora(self):
        args = _parse("batch", "/some/dir", "-s", "anime", "--no-lora")
        assert args.no_lora is True

    def test_batch_mutex_enforced(self):
        with pytest.raises(SystemExit):
            _parse(
                "batch", "/some/dir", "-s", "anime",
                "--lora", "x/y", "--no-lora",
            )

    def test_batch_multiple_lora_append(self):
        args = _parse(
            "batch", "/some/dir", "-s", "anime",
            "--lora", "a/1:0.7",
            "--lora", "b/2:0.5",
        )
        assert [r.ref for r in args.lora] == ["a/1", "b/2"]
        assert [r.weight for r in args.lora] == [0.7, 0.5]

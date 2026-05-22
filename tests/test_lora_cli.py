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

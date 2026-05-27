"""v0.10.0 commit 4 — `imgen train` parser stanza tests.

Covers:

* :func:`imgen.parser._lora_name_arg` — slug grammar for `--name`
  (filesystem path component + CLI ref + meta-json dict key).
* :func:`imgen.parser._trigger_token_arg` — `--trigger` validator with
  control-byte + Unicode `Cf`/`Mn` rejection (§R.1 security M-2).
* :func:`imgen.parser._add_run_control_args` — new `preview_supported`
  kwarg suppresses ``-p/--preview`` (§R.1 memo M.5).
* :func:`imgen.parser._add_train_args` — flag registration uses the
  shared module constants from :mod:`imgen.models` (§R.1 python H-13).
* ``imgen train --help`` end-to-end via :func:`build_parser` (subparser
  wired + ``cmd_train`` dispatched via cli ``_HANDLERS``).

Strict TDD per CLAUDE.md (pure validators + argparse wiring).
"""
from __future__ import annotations

import argparse

import pytest

from imgen.models import (
    _VALID_LORA_RANKS,
    _VALID_QUANTIZE_TRAIN,
    _VALID_TRAIN_RESOLUTIONS,
)
from imgen.parser import (
    _add_run_control_args,
    _add_train_args,
    _lora_name_arg,
    _trigger_token_arg,
    build_parser,
)


# ── _lora_name_arg ────────────────────────────────────────────────

class TestLoraNameArgAccepts:
    @pytest.mark.parametrize(
        "name",
        [
            "a",  # single char
            "a1",  # min two-char
            "al1na",  # colleague's recipe canonical
            "my-lora",
            "my_lora",
            "a-b-c",
            "a_b_c",
            "lora-1",
            "user-face-2025",
            "x" * 32,  # max length
        ],
    )
    def test_accepts_valid_slug(self, name):
        assert _lora_name_arg(name) == name


class TestLoraNameArgRejects:
    def test_rejects_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty"):
            _lora_name_arg("")

    def test_rejects_too_long(self):
        with pytest.raises(argparse.ArgumentTypeError, match="too long"):
            _lora_name_arg("x" * 33)

    @pytest.mark.parametrize(
        "bad",
        [
            "-",  # single dash
            "--",
            "---",
            "-foo",  # leading dash
            "foo-",  # trailing dash
            "_foo",  # leading underscore (H-3 closure: start+end alnum)
            "foo_",  # trailing underscore
            "_",
            "__",
        ],
    )
    def test_rejects_pure_dash_or_boundary_punct(self, bad):
        """§R.1 python H-3: regex requires start AND end to be alnum
        ``[a-z0-9]``; pure-dash slugs and leading/trailing punctuation
        all reject."""
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_name_arg(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "Foo",  # uppercase
            "MyLora",
            "FOO",
            "café",  # non-ASCII
            "foo.bar",  # dot
            "foo bar",  # space
            "foo/bar",  # path separator (traversal)
            "../etc",
            "foo;bar",  # shell metachar
            "foo|bar",
            "foo&bar",
            "foo$bar",
            "foo\\bar",
            "foo\x00bar",  # NUL
            "foo\nbar",
            "foo\x1bbar",  # ESC
        ],
    )
    def test_rejects_disallowed_chars(self, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            _lora_name_arg(bad)


# ── _trigger_token_arg ────────────────────────────────────────────

class TestTriggerTokenArgAccepts:
    @pytest.mark.parametrize(
        "trigger",
        [
            "al1na",
            "al1na woman",  # colleague's recipe canonical
            "My Person",
            "ohwx man",
            "a",  # single char
            "x" * 64,  # max length
            "Café Style",  # non-ASCII letters (Lu/Ll/Lo) ALLOWED
            "Привет мир",  # Cyrillic
            "日本語",  # CJK
            "foo!",  # punctuation
            "foo's",
            'foo "bar"',
            "trigger-123",
        ],
    )
    def test_accepts_natural_trigger(self, trigger):
        assert _trigger_token_arg(trigger) == trigger


class TestTriggerTokenArgRejects:
    def test_rejects_empty(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty"):
            _trigger_token_arg("")

    def test_rejects_too_long(self):
        with pytest.raises(argparse.ArgumentTypeError, match="too long"):
            _trigger_token_arg("x" * 65)

    @pytest.mark.parametrize(
        "bad",
        [" foo", "foo ", "\tfoo", "foo\t", " foo ", "  foo"],
    )
    def test_rejects_boundary_whitespace(self, bad):
        with pytest.raises(argparse.ArgumentTypeError, match="whitespace"):
            _trigger_token_arg(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "foo\x00bar",  # NUL
            "foo\x1bbar",  # ESC
            "foo\x7fbar",  # DEL
            "foo\x9bbar",  # CSI (C1)
            "foo\nbar",  # newline (C0)
        ],
    )
    def test_rejects_control_bytes(self, bad):
        with pytest.raises(argparse.ArgumentTypeError, match="control bytes"):
            _trigger_token_arg(bad)

    @pytest.mark.parametrize(
        "ch,label",
        [
            ("‮", "RLO bidi override"),
            ("‎", "LRM bidi mark"),
            ("​", "ZWSP zero-width space"),
            ("‍", "ZWJ zero-width joiner"),
            ("﻿", "BOM"),
        ],
    )
    def test_rejects_unicode_format_chars_cf(self, ch, label):
        """§R.1 security M-2: Unicode category ``Cf`` (format) covers
        RLO/LRM/ZWSP/ZWJ/BOM — bidi overrides and zero-width
        spoofing. Rejected so they can't visually impersonate
        a benign LoRA name in CLI/Finder display."""
        with pytest.raises(argparse.ArgumentTypeError, match="Cf"):
            _trigger_token_arg(f"foo{ch}bar")

    @pytest.mark.parametrize(
        "ch,label",
        [
            ("́", "combining acute accent"),
            ("̀", "combining grave accent"),
            ("̧", "combining cedilla"),
        ],
    )
    def test_rejects_unicode_combining_marks_mn(self, ch, label):
        """§R.1 security M-2: Unicode category ``Mn`` (nonspacing mark)
        — combining diacritics. Rejected so a string ``e\\u0301`` (e + acute)
        doesn't visually equal ``é`` from the precomposed form."""
        with pytest.raises(argparse.ArgumentTypeError, match="Mn"):
            _trigger_token_arg(f"foo{ch}bar")


# ── _add_run_control_args preview_supported kwarg ─────────────────

class TestAddRunControlArgsPreviewSupported:
    def test_default_includes_preview_flag(self):
        """Existing surface (draw/refine/video) must keep ``-p/--preview``."""
        p = argparse.ArgumentParser()
        _add_run_control_args(p)
        ns = p.parse_args(["--preview"])
        assert ns.preview is True

    def test_short_p_alias_still_works_by_default(self):
        p = argparse.ArgumentParser()
        _add_run_control_args(p)
        ns = p.parse_args(["-p"])
        assert ns.preview is True

    def test_preview_supported_false_omits_flag(self):
        """§R.1 memo M.5: when ``preview_supported=False`` (cmd_train),
        ``--preview`` must NOT be registered at all (NOT just hidden) —
        train has no preview mode; accepting the flag silently would
        let users believe it does something."""
        p = argparse.ArgumentParser()
        _add_run_control_args(p, preview_supported=False)
        with pytest.raises(SystemExit):
            p.parse_args(["--preview"])

    def test_preview_supported_false_keeps_other_flags(self):
        """``--no-open``, ``-y/--yes``, ``--dry-run``, ``--force`` stay
        registered even when ``preview_supported=False``."""
        p = argparse.ArgumentParser()
        _add_run_control_args(p, preview_supported=False)
        ns = p.parse_args(["--no-open", "-y", "--dry-run", "--force"])
        assert ns.no_open is True
        assert ns.yes is True
        assert ns.dry_run is True
        assert ns.force is True


# ── _add_train_args ───────────────────────────────────────────────

def _build_train_parser():
    """Helper: standalone parser preloaded with the train stanza."""
    p = argparse.ArgumentParser()
    _add_train_args(p, defaults={})
    return p


class TestAddTrainArgsRequiredFlags:
    def test_dataset_required(self, capsys):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--name", "foo", "--trigger", "bar"])
        err = capsys.readouterr().err
        assert "--dataset" in err

    def test_name_required(self, capsys):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--dataset", "/tmp/x", "--trigger", "bar"])
        err = capsys.readouterr().err
        assert "--name" in err

    def test_trigger_required(self, capsys):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--dataset", "/tmp/x", "--name", "foo"])
        err = capsys.readouterr().err
        assert "--trigger" in err

    def test_minimal_required_set_succeeds(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.name == "foo"
        assert ns.trigger == "bar"
        assert str(ns.dataset) == "/tmp/x"


class TestAddTrainArgsOptionalDefaults:
    """Optional flags default to ``None`` — sentinel for "use the
    :class:`TrainingConfig` default at resolve time" (commit 8)."""

    def test_steps_defaults_to_none(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.steps is None

    def test_rank_defaults_to_none(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.rank is None

    def test_quantize_defaults_to_none(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.quantize is None

    def test_max_resolution_defaults_to_none(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.max_resolution is None

    def test_preview_every_defaults_to_none(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.preview_every is None

    def test_base_defaults_to_klein_4b(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.base == "flux2-klein-4b"

    def test_battery_stop_defaults_to_20(self):
        """20% > mflux-train's own 5% default — overnight safety
        margin. Locked in [[project-v100-design]] §F."""
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.battery_stop == 20

    def test_overwrite_defaults_false(self):
        p = _build_train_parser()
        ns = p.parse_args(
            ["--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar"]
        )
        assert ns.overwrite is False


class TestAddTrainArgsChoicesShareConstants:
    """§R.1 python H-13: argparse ``choices=`` must reference the same
    :mod:`imgen.models` module constants used by ``__post_init__``.
    Single source of truth — adding ``rank=128`` later means editing
    one constant, not two places."""

    @pytest.mark.parametrize("rank", sorted(_VALID_LORA_RANKS))
    def test_rank_accepts_every_valid_rank(self, rank):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--rank", str(rank),
            ]
        )
        assert ns.rank == rank

    @pytest.mark.parametrize("bad", [1, 2, 3, 5, 7, 9, 128])
    def test_rank_rejects_outside_set(self, bad):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--rank", str(bad),
                ]
            )

    @pytest.mark.parametrize("q", sorted(_VALID_QUANTIZE_TRAIN))
    def test_quantize_accepts_every_valid_q(self, q):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--quantize", str(q),
            ]
        )
        assert ns.quantize == q

    @pytest.mark.parametrize("bad", [0, 1, 2, 7, 9, 16])
    def test_quantize_rejects_outside_set(self, bad):
        """``0`` is excluded — mflux-train does not accept bf16 for
        training (verified at commit 1; see TrainingConfig docstring)."""
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--quantize", str(bad),
                ]
            )

    @pytest.mark.parametrize("res", sorted(_VALID_TRAIN_RESOLUTIONS))
    def test_max_resolution_accepts_every_valid(self, res):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--max-resolution", str(res),
            ]
        )
        assert ns.max_resolution == res

    @pytest.mark.parametrize("bad", [128, 200, 640, 2048])
    def test_max_resolution_rejects_outside_set(self, bad):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--max-resolution", str(bad),
                ]
            )


class TestAddTrainArgsPreviewEveryRange:
    """§R.1 memo §M.12 (round-2 N-3): mflux-train rejects
    ``monitoring.generate_image_frequency=0`` with
    ``ValueError: Monitoring generate_image_frequency must be > 0`` —
    verified via ``mflux-train --dry-run`` on 2026-05-28. Floor for
    ``--preview-every`` stays at 1."""

    @pytest.mark.parametrize("n", [1, 50, 100, 500, 1000])
    def test_preview_every_accepts_positive(self, n):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--preview-every", str(n),
            ]
        )
        assert ns.preview_every == n

    def test_preview_every_rejects_zero(self):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--preview-every", "0",
                ]
            )

    def test_preview_every_rejects_above_1000(self):
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--preview-every", "1001",
                ]
            )


class TestAddTrainArgsRunControl:
    """Train uses the universal run-control helper with
    ``preview_supported=False`` (§R.1 memo M.5). ``--no-open``, ``-y``,
    ``--dry-run``, ``--force`` are all present. ``--preview`` is
    NOT registered."""

    def test_no_open_present(self):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--no-open",
            ]
        )
        assert ns.no_open is True

    def test_yes_present(self):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "-y",
            ]
        )
        assert ns.yes is True

    def test_dry_run_present(self):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--dry-run",
            ]
        )
        assert ns.dry_run is True

    def test_force_present(self):
        p = _build_train_parser()
        ns = p.parse_args(
            [
                "--dataset", "/tmp/x", "--name", "foo", "--trigger", "bar",
                "--force",
            ]
        )
        assert ns.force is True

    def test_preview_flag_not_registered(self):
        """Train has no preview mode — ``--preview`` must fail."""
        p = _build_train_parser()
        with pytest.raises(SystemExit):
            p.parse_args(
                [
                    "--dataset", "/tmp/x", "--name", "foo",
                    "--trigger", "bar", "--preview",
                ]
            )


# ── build_parser integration ──────────────────────────────────────

class TestBuildParserTrainSubcommand:
    def test_train_subcommand_registered(self):
        parser = build_parser()
        ns = parser.parse_args(
            [
                "train", "--dataset", "/tmp/x", "--name", "foo",
                "--trigger", "bar",
            ]
        )
        assert ns.command == "train"
        assert ns.name == "foo"
        assert ns.trigger == "bar"

    def test_train_help_lists_required_flags(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--help"])
        out = capsys.readouterr().out
        assert "--dataset" in out
        assert "--name" in out
        assert "--trigger" in out

    def test_train_help_documents_optional_flags(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--help"])
        out = capsys.readouterr().out
        assert "--rank" in out
        assert "--quantize" in out
        assert "--max-resolution" in out
        assert "--preview-every" in out
        assert "--battery-stop" in out
        assert "--overwrite" in out
        assert "--base" in out

    def test_train_help_omits_preview_flag(self, capsys):
        """``-p/--preview`` belongs to inference subcommands; the
        train surface deliberately omits it via the
        ``preview_supported=False`` kwarg (§R.1 memo M.5)."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "--help"])
        out = capsys.readouterr().out
        assert "--preview " not in out  # not "preview-every"
        assert "-p," not in out  # short alias also absent


class TestCliHandlerWiredForTrain:
    """``cli._KNOWN_SUBCOMMANDS`` and ``cli._HANDLERS`` must include
    ``"train" -> cmd_train`` so the dispatch lookup in ``main()``
    routes ``imgen train`` to the handler that raises
    ``NotImplementedError`` (commit 3 stub) until commit 8 lands
    the real flow."""

    def test_train_in_known_subcommands(self):
        from imgen.cli import _KNOWN_SUBCOMMANDS
        assert "train" in _KNOWN_SUBCOMMANDS

    def test_train_handler_is_cmd_train(self):
        from imgen.cli import _HANDLERS
        from imgen.commands.train import cmd_train
        assert _HANDLERS.get("train") is cmd_train

"""v0.7.0 (architect §F + §M): `imgen draw` orchestrator + parser
+ helpers integration tests.

Mocks the mflux subprocess at the same seam as test_generate_enhance.py
so no real GPU work happens in the suite. Exercises:

  * parser stanza (positional prompt, --prompt-file mutex, defaults)
  * prompt-slug helper (table-driven per design §D)
  * cmd_draw orchestrator (dry-run + mocked subprocess + history)
  * Backward-compat: existing i2i flows unaffected
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from imgen.cmd_helpers import (
    build_draw_iteration,
    prompt_slug,
)
from imgen.runs import _MAX_COLLISIONS, next_available_path
from imgen.commands.draw import cmd_draw
from imgen.defaults import DEFAULTS


# ── prompt_slug helper (architect §D) ────────────────────────────────


class TestPromptSlug:
    def test_single_word(self):
        assert prompt_slug("samurai") == "samurai"

    def test_multi_word_first_six(self):
        assert (
            prompt_slug("a samurai on a misty mountain at dawn with a sword")
            == "a-samurai-on-a-misty-mountain"
        )

    def test_lowercases(self):
        assert prompt_slug("RED Dragon Breathing FIRE") == "red-dragon-breathing-fire"

    def test_punctuation_collapsed_to_dash(self):
        assert prompt_slug("hello, world! how are you?") == "hello-world-how-are-you"

    def test_unicode_normalized_to_ascii(self):
        """NFKD + ASCII-strip — accented chars decompose to bare letter."""
        assert "naive" in prompt_slug("naïve café")

    def test_emoji_only_fallback_to_draw(self):
        """Emoji + non-ASCII content strips to empty after NFKD; the
        fallback ensures the output filename is always valid."""
        assert prompt_slug("🎨🖼️") == "draw"

    def test_length_capped_at_60(self):
        long_prompt = "a " * 100  # 200 chars of "a a a a ..."
        slug = prompt_slug(long_prompt)
        assert len(slug) <= 60

    def test_empty_string_fallback_to_draw(self):
        assert prompt_slug("") == "draw"

    def test_leading_trailing_dashes_stripped(self):
        assert prompt_slug("---hello---") == "hello"


# ── next_available_path — collision suffix (v0.7.1 generalised) ───────


class TestNextAvailablePath:
    """v0.7.1: helper moved to runs.next_available_path with a
    parametrised `suffix` arg. Default `.png` preserves the v0.7.0
    cmd_draw use case; future video/jsonl callers can pass `.mp4` /
    `.jsonl` without duplicating the collision-suffix loop."""

    def test_first_run_no_suffix(self, tmp_path):
        out = next_available_path(tmp_path, "samurai")
        assert out == tmp_path / "samurai.png"

    def test_collision_appends_2(self, tmp_path):
        (tmp_path / "samurai.png").write_bytes(b"existing")
        out = next_available_path(tmp_path, "samurai")
        assert out == tmp_path / "samurai-2.png"

    def test_multiple_collisions_increment(self, tmp_path):
        (tmp_path / "samurai.png").write_bytes(b"a")
        (tmp_path / "samurai-2.png").write_bytes(b"b")
        (tmp_path / "samurai-3.png").write_bytes(b"c")
        out = next_available_path(tmp_path, "samurai")
        assert out == tmp_path / "samurai-4.png"

    def test_custom_suffix(self, tmp_path):
        """Future v0.7.x callers (video output, history rotation)
        pass a different suffix. Suffix lands before the collision
        index, NOT after — `samurai-2.mp4`, not `samurai.mp4-2`."""
        (tmp_path / "samurai.mp4").write_bytes(b"a")
        out = next_available_path(tmp_path, "samurai", suffix=".mp4")
        assert out == tmp_path / "samurai-2.mp4"

    def test_caps_collisions_at_cap(self, tmp_path, monkeypatch):
        """v0.7.10: collision loop raises RuntimeError once it exceeds
        _MAX_COLLISIONS. Mocks Path.exists() always-True so the loop
        walks to the cap without 1001 real files on disk.

        `monkeypatch.setattr(Path, "exists", ...)` patches the method
        on the class for the duration of this test only; pytest reverts
        it on teardown, so no cross-test leakage."""
        monkeypatch.setattr(Path, "exists", lambda self: True)
        with pytest.raises(
            RuntimeError, match=f"more than {_MAX_COLLISIONS} collisions"
        ):
            next_available_path(tmp_path, "samurai")


# ── Parser stanza ────────────────────────────────────────────────────


def _parse_draw(*argv):
    from imgen.parser import build_parser
    return build_parser({
        "style": "pixar", "backend": "flux", "quantize": 8,
        "steps": 20, "guidance": 3.5, "strength": 0.55,
        "mlx_cache_gb": 12, "battery_stop": 20,
    }).parse_args(["draw", *argv])


class TestDrawParser:
    def test_positional_prompt(self):
        args = _parse_draw("a samurai")
        assert args.command == "draw"
        assert args.prompt == "a samurai"

    def test_default_backend_is_flux_dev(self):
        args = _parse_draw("a samurai")
        assert args.model == "flux-dev"

    def test_default_dimensions_1024x1024(self):
        args = _parse_draw("a samurai")
        assert args.width == 1024
        assert args.height == 1024

    def test_no_scope_flag(self):
        """draw is t2i; --scope is i2i-only. The parser rejects it."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--scope", "scene")

    def test_no_strength_flag(self):
        """t2i: no source photo to interpolate against."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--strength", "0.5")

    def test_no_style_flag(self):
        """v0.7.0: --style deferred to v0.7.1+ for draw."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--style", "anime")

    def test_prompt_file_alternative(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("ignored — parser doesn't read")
        args = _parse_draw("--prompt-file", str(f))
        assert args.prompt is None
        assert args.prompt_file == f

    def test_lora_comma_split_works(self):
        """v0.7.0 step 1: comma-split --lora applies to draw too."""
        args = _parse_draw("a samurai", "--lora", "a/one,b/two:0.5")
        # list[list[LoraRef]] from action='append' + _lora_refs_arg
        assert len(args.lora) == 1
        assert len(args.lora[0]) == 2

    def test_enhance_flags_present(self):
        args = _parse_draw("a samurai", "--enhance-prompt")
        assert args.enhance is True

    # ── v0.7.3: --num-iterations parser stanza ────────────────────────

    def test_num_iterations_default_1(self):
        args = _parse_draw("a samurai")
        assert args.num_iterations == 1

    def test_num_iterations_explicit(self):
        args = _parse_draw("a samurai", "--num-iterations", "5")
        assert args.num_iterations == 5

    def test_num_iterations_short_alias(self):
        args = _parse_draw("a samurai", "-n", "3")
        assert args.num_iterations == 3

    def test_num_iterations_below_1_rejected(self):
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--num-iterations", "0")

    def test_num_iterations_above_cap_rejected(self):
        """Cap 32 protects against accidental `-n 9999`."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--num-iterations", "33")

    # ── v0.7.11 (gap 1): --negative-prompt parser stanza ──────────────

    def test_negative_prompt_default_none(self):
        """No CLI flag → args.negative_prompt is None (sentinel for "no
        CLI override"). Pre-v0.7.11 the attribute didn't exist at all;
        gap 1 closure adds it."""
        args = _parse_draw("a samurai")
        assert args.negative_prompt is None

    def test_negative_prompt_explicit(self):
        """`--negative-prompt "low quality"` parses through to argv.
        Z-Image and FLUX.1-dev model cards recommend negatives;
        pre-v0.7.11 imgen had no way to expose this through `imgen draw`."""
        args = _parse_draw("a samurai", "--negative-prompt", "low quality, blurry")
        assert args.negative_prompt == "low quality, blurry"

    # ── v0.7.11 (gap 2): --guidance accepts 0.0 for distilled models ──

    def test_guidance_zero_accepted(self):
        """Distilled models (Z-Image-Turbo, FLUX-schnell) train with
        classifier-free guidance disabled — argv must accept 0.0.
        Pre-v0.7.11 the parser floor was 0.5, blocking these backends."""
        args = _parse_draw("a samurai", "--guidance", "0.0")
        assert args.guidance == 0.0

    def test_guidance_below_zero_rejected(self):
        """0.0 is the new floor — negative guidance has no physical meaning."""
        with pytest.raises(SystemExit):
            _parse_draw("a samurai", "--guidance", "-0.1")


# ── v0.7.3: build_draw_iterations (plural) ───────────────────────────


class TestBuildDrawIterations:
    """v0.7.3: plural form of build_draw_iteration; seed ladder +
    output naming + explicit_output mutex with N>=2."""

    def test_n_equals_1_single_iteration(self, tmp_path):
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=1,
        )
        assert len(out) == 1
        # N=1 → bare slug.png, no -1 suffix.
        assert "-1.png" not in out[0].output_path.name
        assert out[0].output_path.name.endswith("a-samurai.png")
        # Seed = base_seed.
        seed_idx = out[0].cmd.index("--seed") + 1
        assert out[0].cmd[seed_idx] == "42"

    def test_n_equals_3_seed_ladder(self, tmp_path):
        """Deterministic ladder: base_seed=100, N=3 → seeds 100,101,102."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=100,
            num_iterations=3,
        )
        assert len(out) == 3
        # Each iteration's argv carries its slot's seed.
        seeds = [
            it.cmd[it.cmd.index("--seed") + 1] for it in out
        ]
        assert seeds == ["100", "101", "102"]

    def test_n_equals_3_output_naming(self, tmp_path):
        """N>=2 → <slug>-1.png ... <slug>-N.png (1-indexed)."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=3,
        )
        names = [it.output_path.name for it in out]
        assert names == [
            "a-samurai-1.png",
            "a-samurai-2.png",
            "a-samurai-3.png",
        ]

    # ── v0.7.11 (gap 1): negative-prompt propagation ──────────────────

    def test_negative_prompt_propagates_to_iteration_and_argv(self, tmp_path):
        """`--negative-prompt "low quality"` on the CLI → Iteration.negative
        carries the string AND argv emits `--negative-prompt low quality`
        for backends with supports_negative=True (flux-dev qualifies)."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(negative_prompt="low quality, blurry"),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=1,
        )
        assert out[0].negative == "low quality, blurry"
        assert "--negative-prompt" in out[0].cmd
        idx = out[0].cmd.index("--negative-prompt")
        assert out[0].cmd[idx + 1] == "low quality, blurry"

    def test_negative_prompt_absent_preserves_empty_default(self, tmp_path):
        """No `--negative-prompt` CLI → Iteration.negative == "" and argv
        omits the flag. Locks the v0.7.0 → v0.7.10 behaviour against
        accidental regression now that the plumbing accepts the value."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),  # negative_prompt=None per fixture default
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=1,
        )
        assert out[0].negative == ""
        assert "--negative-prompt" not in out[0].cmd

    def test_negative_prompt_dropped_on_flux2_klein_edit_9b(self, tmp_path):
        """v0.7.11 cross-product defence-in-depth (gap 7 × gap 1 interaction):
        even when the user explicitly passes `--negative-prompt "X"` on
        the CLI, the flag must NOT reach argv when targeting a backend
        with ``supports_negative=False``. The existing gap-7 test in
        test_generate_cmd.py guards the build_mflux_cmd seam; this test
        closes the end-to-end loop from the draw entrypoint."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(negative_prompt="low quality, blurry"),
            prompt="a portrait",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux2-klein-edit-9b"],
            binary=Path("/fake/mflux-generate-flux2-edit"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=1,
        )
        # Iteration still records the user's intent (for replay /
        # history), but argv-level emission is gated by
        # backend.supports_negative=False.
        assert out[0].negative == "low quality, blurry"
        assert "--negative-prompt" not in out[0].cmd

    def test_seed_ladder_wraps_at_2_32(self, tmp_path):
        """base_seed near the cap: ladder modulo 2^32 keeps every
        iteration inside mflux's valid seed range."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        cap = 2**32
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=cap - 1,
            num_iterations=3,
        )
        seeds = [int(it.cmd[it.cmd.index("--seed") + 1]) for it in out]
        # Ladder: cap-1, (cap-1+1)%cap=0, (cap-1+2)%cap=1.
        assert seeds == [cap - 1, 0, 1]

    def test_explicit_output_with_n_gt_1_raises(self, tmp_path):
        """--output PATH single-file is mutex with N>=2 — a single
        FILE can't fan out to N images."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        with pytest.raises(ValueError, match="mutex"):
            build_draw_iterations(
                args=_make_args(),
                prompt="a samurai",
                merged_defaults=DEFAULTS,
                be=BACKENDS["flux-dev"],
                binary=Path("/fake/mflux-generate"),
                width=1024,
                height=1024,
                explicit_output=tmp_path / "mypic.png",
                run_dir=None,
                base_seed=42,
                num_iterations=3,
            )

    def test_num_iterations_zero_raises(self, tmp_path):
        """Parser caps 1..32 but the helper's own contract rejects <1
        for any future programmatic caller."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        with pytest.raises(ValueError):
            build_draw_iterations(
                args=_make_args(),
                prompt="a samurai",
                merged_defaults=DEFAULTS,
                be=BACKENDS["flux-dev"],
                binary=Path("/fake/mflux-generate"),
                width=1024,
                height=1024,
                explicit_output=None,
                run_dir=tmp_path,
                base_seed=42,
                num_iterations=0,
            )

    def test_per_iteration_seed_on_iteration_object(self, tmp_path):
        """v0.7.3 pre-tag CRITICAL fix: Iteration.seed carries the
        per-iteration ladder seed (NOT base_seed for all). Pre-fix,
        run_one_iteration wrote ctx.seed (= base) to every history
        row, breaking replay reproducibility for rows 2..N.

        Lock-in: build N=3 → Iteration.seed values are
        [base, base+1, base+2], matching the argv --seed values."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=500,
            num_iterations=3,
        )
        # Iteration.seed reads independently of cmd argv.
        assert [it.seed for it in out] == [500, 501, 502]
        # Cross-check: argv --seed matches Iteration.seed (no drift
        # between the two surfaces).
        for it in out:
            argv_seed = int(it.cmd[it.cmd.index("--seed") + 1])
            assert argv_seed == it.seed

    def test_all_iterations_share_compatible_loras(self, tmp_path):
        """LoRA resolution runs ONCE outside the loop (same prompt →
        same triggers → same compat-filtered stack); lock the
        invariant that all N Iteration objects carry the same loras
        tuple by reference."""
        from imgen.backends import BACKENDS
        from imgen.cmd_helpers import build_draw_iterations
        out = build_draw_iterations(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            base_seed=42,
            num_iterations=3,
        )
        assert all(it.loras == out[0].loras for it in out)


# ── v0.7.3: cmd_draw enhancer-once optimisation ─────────────────────


class TestCmdDrawEnhancerOnce:
    """v0.7.3 critical optimisation: `--num-iterations N --enhance-prompt`
    fires the LLM ONCE on the unique prompt, then broadcasts the result
    across N iterations. Without this, --num-iterations 5 --enhance
    would pay 5× the LLM cost for 5 identical prompts."""

    def test_enhancer_fires_once_for_n_iterations(
        self, tmp_path, monkeypatch, capsys,
    ):
        from imgen.backends import BACKENDS
        from imgen.enhance import EnhanceResult

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )

        call_count = {"n": 0, "prompts": []}

        def fake_orchestrator(
            *, iteration_prompts, system_prompt, invariants,
            model, temperature, max_tokens, timeout_s,
        ):
            call_count["n"] += 1
            call_count["prompts"].append(list(iteration_prompts))
            return [
                EnhanceResult(
                    final_prompt=f"ENH: {p}",
                    original_prompt=p,
                    was_enhanced=True,
                    fallback_reason=None,
                    was_truncated=False,
                    raw_llm_output=f"ENH: {p}",
                )
                for p in iteration_prompts
            ]

        monkeypatch.setattr(
            "imgen.cmd_helpers.enhance_iteration_prompts", fake_orchestrator,
        )

        args = _make_args(
            prompt="a samurai",
            enhance=True,
            num_iterations=5,
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        # CRITICAL: orchestrator was called ONCE with a list of 1
        # prompt (the unique one), not 5 times or once with 5 copies.
        assert call_count["n"] == 1
        assert call_count["prompts"] == [["a samurai"]]
        # Each of the 5 dry-run commands shows the SAME enhanced prompt.
        out = capsys.readouterr().out
        assert out.count("ENH: a samurai") == 5


# ── build_draw_iteration ─────────────────────────────────────────────


def _make_args(**overrides):
    """SimpleNamespace mirroring the draw parser shape — used to drive
    build_draw_iteration directly in pure tests."""
    defaults = dict(
        prompt="a samurai on a mountain",
        prompt_file=None,
        steps=None,
        quantize=None,
        guidance=None,
        seed=42,
        model="flux-dev",
        preview=False,
        width=1024,
        height=1024,
        num_iterations=1,  # v0.7.3 default
        no_open=True,
        yes=True,
        dry_run=False,
        force=True,
        enhance=False,
        enhance_model=None,
        enhance_temperature=None,
        imgen_config_enhance={},
        output=None,
        output_dir=None,
        lora=None,
        no_lora=False,
        # v0.7.11 (gap 1): argparse always populates this attribute
        # (default=None from the parser stanza), so the fixture
        # mirrors that to keep test args shape-consistent with real
        # parser output. Tests that exercise the absent case can
        # explicitly override via _make_args(negative_prompt=None)
        # or rely on the getattr fallback in build_draw_iterations.
        negative_prompt=None,
        imgen_merged_defaults=DEFAULTS,
        imgen_config_output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildDrawIteration:
    def test_returns_single_iteration(self, tmp_path):
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert it.style_name == "draw"
        assert it.prompt == "a samurai"
        assert it.final_quantize == DEFAULTS["quantize"]
        assert it.final_steps == DEFAULTS["steps"]

    def test_output_path_slug_under_run_dir(self, tmp_path):
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai on a misty mountain",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        # Slug derives from first 6 words; lands inside run_dir.
        assert it.output_path.parent == tmp_path
        assert "samurai" in it.output_path.name
        assert it.output_path.suffix == ".png"

    def test_explicit_output_overrides_slug(self, tmp_path):
        from imgen.backends import BACKENDS
        explicit = tmp_path / "my-pic.png"
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=explicit,
            run_dir=None,
            seed=42,
        )
        assert it.output_path == explicit

    def test_no_image_path_in_cmd(self, tmp_path):
        """t2i: --image-path argv pair must NOT appear."""
        from imgen.backends import BACKENDS
        it = build_draw_iteration(
            args=_make_args(),
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        assert "--image-path" not in it.cmd
        # Sanity: prompt + output still present.
        assert "--prompt" in it.cmd
        assert it.cmd[it.cmd.index("--prompt") + 1] == "a samurai"

    def test_strength_recorded_but_no_argv_emission(self, tmp_path):
        """args.strength is missing on draw Namespace; the getattr
        defence in _resolve_iteration_params falls through to default.
        backend.supports_strength=False so build_mflux_cmd doesn't emit
        --image-strength. Locks the symmetric pass-through."""
        from imgen.backends import BACKENDS
        # Construct args WITHOUT strength attribute (mirror real draw
        # Namespace).
        args = _make_args()
        assert not hasattr(args, "strength")
        it = build_draw_iteration(
            args=args,
            prompt="a samurai",
            merged_defaults=DEFAULTS,
            be=BACKENDS["flux-dev"],
            binary=Path("/fake/mflux-generate"),
            width=1024,
            height=1024,
            explicit_output=None,
            run_dir=tmp_path,
            seed=42,
        )
        # final_strength has a value (from merged_defaults) but argv
        # doesn't carry --image-strength.
        assert "--image-strength" not in it.cmd
        assert it.final_strength == DEFAULTS["strength"]


# ── cmd_draw dry-run path ────────────────────────────────────────────


class TestCmdDrawDryRun:
    """Dry-run hits cmd_draw, exercises the whole pipeline up to
    subprocess execution, then prints + exits clean. Validates the
    parser → build_draw_iteration → cmd-display chain end-to-end
    without needing a real mflux."""

    def test_dry_run_prints_cmd_and_exits_0(
        self, tmp_path, monkeypatch, capsys,
    ):
        # Stub load_backend_and_token so the gated-token / venv-binary
        # checks don't fire in tests.
        def fake_load(args):
            from imgen.backends import BACKENDS
            return ("flux-dev", BACKENDS["flux-dev"], "fake-token",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )

        args = _make_args(
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Dry-run banner.
        assert "Dry run" in out
        # The mflux argv content surfaces in the dry-run output.
        assert "--prompt" in out
        assert "--model" in out
        assert "dev" in out  # --model dev
        # No --image-path (t2i).
        assert "--image-path" not in out

    def test_dry_run_requires_prompt(self, tmp_path, monkeypatch):
        """A draw invocation with neither positional nor --prompt-file
        dies cleanly with exit code 2."""
        args = _make_args(prompt=None, prompt_file=None, dry_run=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_draw(args)
        assert exc_info.value.code == 2

    def test_positional_dash_reads_stdin(
        self, tmp_path, monkeypatch, capsys,
    ):
        """positional '-' reads from stdin (hides prompt from `ps`)."""
        import io
        from imgen.backends import BACKENDS

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )
        monkeypatch.setattr(
            "sys.stdin", io.StringIO("a ninja from stdin"),
        )
        args = _make_args(
            prompt="-",
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "a ninja from stdin" in out


class TestCmdDrawRefineHint:
    """v0.7.7: after a successful single-shot imgen draw, surface a
    one-liner pointing at `imgen refine <output>` so the user
    discovers the explore→refine workflow without reading the README.
    Gated on success-only + N=1 + run-dir layout (not --output FILE)."""

    def _success_path_stubs(self, monkeypatch, tmp_state_dir):
        """Shared monkeypatch setup so the test exercises the real
        cmd_draw success path without spawning mflux."""
        from imgen.backends import BACKENDS

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )
        monkeypatch.setattr(
            "imgen.cmd_helpers.run_with_stderr_redaction",
            lambda cmd, **kw: (
                Path(cmd[cmd.index("--output") + 1]).touch(),
                0,
            )[1],
        )
        monkeypatch.setattr(
            "imgen.cmd_helpers.preflight_resources",
            lambda **kw: None,
        )

    def test_hint_fires_after_successful_single_shot(
        self, tmp_state_dir, monkeypatch, tmp_path, capsys,
    ):
        self._success_path_stubs(monkeypatch, tmp_state_dir)
        args = _make_args(
            prompt="a samurai test",
            output_dir=str(tmp_path),
            dry_run=False,
            yes=True,
            no_open=True,
            force=True,
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "imgen refine" in out
        # Hint advertises the refine scale shape without hard-coding
        # absolute pixel dims (architect IMPORTANT #1 from v0.7.7
        # 3-agent review — a user who ran `imgen draw --width 1280
        # --height 720` shouldn't see "1024² → 1536²/2048²").
        assert "1.5×/2×" in out
        assert "1024²" not in out

    def test_hint_skipped_on_explicit_output_path(
        self, tmp_state_dir, monkeypatch, tmp_path, capsys,
    ):
        """--output FILE bypasses run_dir entirely; hint is gated on
        run_dir presence so this path stays silent."""
        self._success_path_stubs(monkeypatch, tmp_state_dir)
        explicit = tmp_path / "explicit.png"
        args = _make_args(
            prompt="a samurai test",
            output=str(explicit),
            output_dir=None,
            dry_run=False,
            yes=True,
            no_open=True,
            force=True,
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "imgen refine" not in out

    def test_hint_skipped_on_dry_run(
        self, tmp_state_dir, monkeypatch, tmp_path, capsys,
    ):
        """Dry-run produces no actual output file — no point hinting
        about refine of a file that doesn't exist."""
        self._success_path_stubs(monkeypatch, tmp_state_dir)
        args = _make_args(
            prompt="a samurai test",
            output_dir=str(tmp_path),
            dry_run=True,
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "imgen refine" not in out

    def test_hint_skipped_on_num_iterations_2(
        self, tmp_state_dir, monkeypatch, tmp_path, capsys,
    ):
        """N>=2 (--num-iterations explore mode) suppresses the hint —
        with multiple outputs the user picks a winner via Finder first,
        making "refine <output>" ambiguous. Lock-in for the is_batch
        gate so a refactor of the hint condition doesn't quietly re-
        enable it for batch runs."""
        self._success_path_stubs(monkeypatch, tmp_state_dir)
        args = _make_args(
            prompt="a samurai test",
            output_dir=str(tmp_path),
            dry_run=False,
            yes=True,
            no_open=True,
            force=True,
            num_iterations=2,
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "imgen refine" not in out


# ── Enhancer wiring for FLUX.1-dev (architect §K) ────────────────────


class TestCmdDrawEnhancer:
    """v0.7.0 step 6: cmd_draw threads the t2i prompt through the
    LLM enhancer when --enhance-prompt is set. BACKENDS['flux-dev']
    declares enhance_invariants=() (no substring anchor) and a
    t2i-tuned enhance_system_prompt — verify the full chain via
    a mocked orchestrator (same seam as test_generate_enhance.py)."""

    def test_enhanced_prompt_reaches_cmd_argv(
        self, tmp_path, monkeypatch, capsys,
    ):
        from imgen.backends import BACKENDS
        from imgen.enhance import EnhanceResult

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )

        captured_system_prompt = []
        def fake_orchestrator(
            *, iteration_prompts, system_prompt, invariants,
            model, temperature, max_tokens, timeout_s,
        ):
            captured_system_prompt.append(system_prompt)
            return [
                EnhanceResult(
                    final_prompt=f"ENH: {p} with detailed lighting and cinematic composition",
                    original_prompt=p,
                    was_enhanced=True,
                    fallback_reason=None,
                    was_truncated=False,
                    raw_llm_output=f"ENH: {p} with detailed lighting and cinematic composition",
                )
                for p in iteration_prompts
            ]
        monkeypatch.setattr(
            "imgen.cmd_helpers.enhance_iteration_prompts", fake_orchestrator,
        )

        args = _make_args(
            prompt="a samurai",
            enhance=True,
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        # The flux-dev t2i system prompt was passed to the LLM
        # (not Kontext's i2i variant).
        assert len(captured_system_prompt) == 1
        assert "text-to-image diffusion" in captured_system_prompt[0]
        # The enhanced prompt reaches the displayed cmd argv.
        out = capsys.readouterr().out
        assert "ENH: a samurai" in out

    def test_gated_repo_hint_surfaces_on_mflux_failure(
        self, tmp_path, monkeypatch, capsys,
    ):
        """v0.7.0 post-tag UX-gap fix: cold-install colleague whose HF
        token works for Kontext but never accepted FLUX.1-dev's license
        sees a 401 GatedRepoError buried in mflux's stack trace. cmd_draw
        appends a friendly hint pointing at the per-model license page
        — read from ``Backend.hf_gated_repo``."""
        from imgen.backends import BACKENDS

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )
        # Force mflux subprocess to "fail" (return non-zero rc) so
        # cmd_draw routes through the failure-summary path with the
        # hint block.
        monkeypatch.setattr(
            "imgen.cmd_helpers.run_with_stderr_redaction",
            lambda cmd, **kw: 1,  # non-zero rc
        )
        monkeypatch.setattr(
            "imgen.cmd_helpers.preflight_resources",
            lambda **kw: None,
        )

        args = _make_args(
            prompt="a samurai",
            enhance=False,
            dry_run=False,
            yes=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        # Non-zero exit propagates from the failed iteration.
        assert rc != 0
        out = capsys.readouterr().out
        # The hint surfaces the HF model URL.
        assert "huggingface.co/black-forest-labs/FLUX.1-dev" in out
        assert "GatedRepoError" in out or "401" in out

    def test_invariants_empty_means_no_substring_check(
        self, tmp_path, monkeypatch, capsys,
    ):
        """flux-dev's enhance_invariants=() short-circuits the
        check_invariants path — an enhancement that drops "samurai"
        entirely would be REJECTED on flux (Kontext) but ACCEPTED on
        flux-dev. Locks the t2i contract (architect §K: t2i prompt-
        fidelity is weaker by design)."""
        from imgen.backends import BACKENDS
        from imgen.enhance import EnhanceResult

        def fake_load(args):
            return ("flux-dev", BACKENDS["flux-dev"], "tok",
                    Path("/fake/mflux-generate"), None)
        monkeypatch.setattr(
            "imgen.commands.draw.load_backend_and_token", fake_load,
        )

        # The "enhanced" prompt has none of the original substrings —
        # a Kontext-style invariant tuple would reject this.
        def fake_orchestrator(
            *, iteration_prompts, system_prompt, invariants,
            model, temperature, max_tokens, timeout_s,
        ):
            # Lock-in: the orchestrator received the empty invariants
            # tuple from flux-dev's Backend.enhance_invariants.
            assert invariants == ()
            return [
                EnhanceResult(
                    final_prompt="completely different output text",
                    original_prompt=p,
                    was_enhanced=True,
                    fallback_reason=None,
                    was_truncated=False,
                    raw_llm_output="completely different output text",
                )
                for p in iteration_prompts
            ]
        monkeypatch.setattr(
            "imgen.cmd_helpers.enhance_iteration_prompts", fake_orchestrator,
        )

        args = _make_args(
            prompt="a samurai",
            enhance=True,
            dry_run=True,
            output_dir=str(tmp_path),
        )
        rc = cmd_draw(args)
        assert rc == 0
        out = capsys.readouterr().out
        # The unrelated enhanced text reaches argv unchallenged.
        assert "completely different output text" in out

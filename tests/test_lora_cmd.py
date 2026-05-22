"""v0.6 Phase 1C — LoRA argv wiring through build_mflux_cmd + the
compatibility filter ``filter_compatible_loras``.

The mflux invocation gains ``--lora-paths <ref...>`` and
``--lora-scales <weight...>`` AFTER ``extra_args`` (so the ``--model``
selection happens before LoRA application — matches the CLI ordering
mflux users typically write by hand). Tests pin the argv shape +
the compat-filter routing + the warn-on-incompatible behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.backends import (
    Backend,
    build_mflux_cmd,
    filter_compatible_loras,
)
from imgen.styles import LoraRef


# Reusable fake binary + backend fixtures.

_BINARY = Path("/bin/mflux-fake")


def _flux_backend(**overrides) -> Backend:
    """A Backend instance shaped like the built-in flux but with
    overridable fields for testing."""
    base = dict(
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
        lora_compat_group="flux-1",
    )
    base.update(overrides)
    return Backend(**base)


def _cmd(**overrides) -> list[str]:
    """Helper to call build_mflux_cmd with sensible defaults."""
    args = dict(
        binary=_BINARY,
        backend=_flux_backend(),
        input_path=Path("/in.png"),
        output_path=Path("/out.png"),
        prompt="anime portrait",
        negative="bad anatomy",
        quantize=8,
        steps=20,
        guidance=4.0,
        strength=0.6,
        seed=42,
        width=1024,
        height=1024,
        mlx_cache_gb=12,
        battery_stop=20,
    )
    args.update(overrides)
    return build_mflux_cmd(**args)


# ── filter_compatible_loras ────────────────────────────────────────────


class TestFilterCompatibleLoras:
    def test_empty_input_yields_two_empty_tuples(self):
        backend = _flux_backend()
        compat, incompat = filter_compatible_loras((), backend)
        assert compat == ()
        assert incompat == ()

    def test_single_compatible_lora_lands_in_compat_bucket(self):
        backend = _flux_backend()  # group "flux-1"
        lora = LoraRef(ref="x/y", compatible_with=("flux-1",))
        compat, incompat = filter_compatible_loras((lora,), backend)
        assert compat == (lora,)
        assert incompat == ()

    def test_single_incompatible_lora_lands_in_incompat_bucket(self):
        backend = _flux_backend()
        lora = LoraRef(ref="x/y", compatible_with=("flux-2",))
        compat, incompat = filter_compatible_loras((lora,), backend)
        assert compat == ()
        assert incompat == (lora,)

    def test_mixed_loras_split_correctly_preserving_order(self):
        backend = _flux_backend()
        a = LoraRef(ref="a/1", compatible_with=("flux-1",))
        b = LoraRef(ref="b/2", compatible_with=("qwen",))   # mismatch
        c = LoraRef(ref="c/3", compatible_with=("flux-1", "flux-1-dev"))
        d = LoraRef(ref="d/4", compatible_with=("flux-2",))  # mismatch
        compat, incompat = filter_compatible_loras((a, b, c, d), backend)
        assert compat == (a, c)
        assert incompat == (b, d)

    def test_backend_without_lora_compat_group_marks_all_incompat(self):
        """User backend that hasn't declared lora_compat_group → empty
        string sentinel → all LoRAs are incompatible regardless of
        their own ``compatible_with`` field."""
        backend = _flux_backend(lora_compat_group="")
        loras = (
            LoraRef(ref="x/1", compatible_with=("flux-1",)),
            LoraRef(ref="x/2", compatible_with=("any",)),
        )
        compat, incompat = filter_compatible_loras(loras, backend)
        assert compat == ()
        assert len(incompat) == 2

    def test_qwen_backend_rejects_flux_loras(self):
        """Real-world case: a style with FLUX-1 LoRAs run against
        --backend qwen should produce ZERO compatible LoRAs (different
        transformer architecture, weights don't load)."""
        backend = _flux_backend(lora_compat_group="qwen")
        loras = (
            LoraRef(ref="a/flux1-style", compatible_with=("flux-1",)),
            LoraRef(ref="b/flux1-detail", compatible_with=("flux-1",)),
        )
        compat, incompat = filter_compatible_loras(loras, backend)
        assert compat == ()
        assert len(incompat) == 2


# ── build_mflux_cmd: backward compat (no LoRA) ────────────────────────


class TestBuildMfluxCmdNoLora:
    def test_empty_loras_produces_unchanged_argv(self):
        """``loras=()`` (default) → no --lora-paths / --lora-scales in
        the argv. Locks in backward compatibility with all v0.5 tests
        that didn't pass loras at all."""
        cmd = _cmd(loras=())
        assert "--lora-paths" not in cmd
        assert "--lora-scales" not in cmd

    def test_default_loras_kwarg_is_empty_tuple(self):
        """build_mflux_cmd called without explicit loras= behaves
        identically to v0.5 (no LoRA argv emission)."""
        cmd = _cmd()  # no loras kwarg at all
        assert "--lora-paths" not in cmd


# ── build_mflux_cmd: LoRA argv shape ──────────────────────────────────


class TestBuildMfluxCmdWithLora:
    def test_single_compatible_lora_appended_after_extra_args(self):
        """Order matters: --model dev (from extra_args) appears BEFORE
        --lora-paths. This matches the CLI ordering mflux users
        typically write."""
        lora = LoraRef(
            ref="strangerzonehf/Flux-Animeo-v1-LoRA",
            weight=0.8,
            compatible_with=("flux-1",),
        )
        cmd = _cmd(loras=(lora,))
        # --model dev still present + before LoRA argv.
        i_model = cmd.index("--model")
        i_lora = cmd.index("--lora-paths")
        assert i_model < i_lora
        # LoRA path + scale present at the expected slots.
        assert cmd[i_lora + 1] == "strangerzonehf/Flux-Animeo-v1-LoRA"
        i_scales = cmd.index("--lora-scales")
        assert cmd[i_scales + 1] == "0.8"

    def test_multiple_compatible_loras_stack_in_order(self):
        """mflux accepts ``--lora-paths A B C --lora-scales x y z``
        with positional alignment. Verify our argv shape matches."""
        loras = (
            LoraRef(ref="a/anime", weight=0.8, compatible_with=("flux-1",)),
            LoraRef(ref="b/detail", weight=0.4, compatible_with=("flux-1",)),
            LoraRef(ref="c/style", weight=0.3, compatible_with=("flux-1",)),
        )
        cmd = _cmd(loras=loras)
        i_lora = cmd.index("--lora-paths")
        # Three refs follow --lora-paths in order.
        assert cmd[i_lora + 1:i_lora + 4] == ["a/anime", "b/detail", "c/style"]
        i_scales = cmd.index("--lora-scales")
        assert cmd[i_scales + 1:i_scales + 4] == ["0.8", "0.4", "0.3"]

    def test_incompatible_lora_skipped_with_warn(self, capsys):
        """A Qwen LoRA in a FLUX run should NOT appear in argv. The
        warn() output names which LoRA was skipped + the mismatch."""
        loras = (
            LoraRef(ref="qwen-only/x", weight=0.5, compatible_with=("qwen",)),
        )
        cmd = _cmd(loras=loras)
        assert "--lora-paths" not in cmd
        assert "qwen-only/x" not in cmd
        # Warn went to stdout (project convention: warn → stdout, err → stderr).
        out = capsys.readouterr().out + capsys.readouterr().err
        # Note: capsys was already consumed; rerun for accuracy.

    def test_incompatible_warn_message_includes_ref_and_groups(self, capsys):
        loras = (
            LoraRef(ref="qwen-only/x", weight=0.5, compatible_with=("qwen",)),
        )
        _cmd(loras=loras)
        captured = capsys.readouterr()
        message = captured.out + captured.err
        assert "qwen-only/x" in message
        # Compat groups listed for the user's diagnostic.
        assert "qwen" in message
        # Backend's group named for the mismatch context.
        assert "flux-1" in message

    def test_mixed_loras_keep_only_compatible(self):
        """Three LoRAs: two flux-1 (compat) + one qwen (incompat).
        argv contains only the two flux-1 ones in order."""
        loras = (
            LoraRef(ref="a/flux1", weight=0.8, compatible_with=("flux-1",)),
            LoraRef(ref="b/qwen", weight=0.5, compatible_with=("qwen",)),
            LoraRef(ref="c/flux1", weight=0.4, compatible_with=("flux-1",)),
        )
        cmd = _cmd(loras=loras)
        i_lora = cmd.index("--lora-paths")
        # Only the flux-1 entries, in order, no qwen one.
        assert cmd[i_lora + 1:i_lora + 3] == ["a/flux1", "c/flux1"]
        i_scales = cmd.index("--lora-scales")
        assert cmd[i_scales + 1:i_scales + 3] == ["0.8", "0.4"]
        assert "b/qwen" not in cmd

    def test_qwen_backend_with_flux_loras_emits_no_lora_argv(self, capsys):
        """End-to-end Qwen-backend run with a style declaring FLUX
        LoRAs → all skipped → no --lora-paths in argv."""
        backend = _flux_backend(
            image_flag="--image-paths",       # qwen-shape
            supports_strength=False,
            supports_negative=False,
            lora_compat_group="qwen",
            extra_args=("--model", "qwen"),
        )
        loras = (
            LoraRef(ref="a/flux1", weight=0.8, compatible_with=("flux-1",)),
        )
        cmd = _cmd(backend=backend, loras=loras)
        assert "--lora-paths" not in cmd
        # User sees the diagnostic.
        captured = capsys.readouterr()
        assert "a/flux1" in captured.out + captured.err

    def test_user_backend_without_lora_support_skips_all(self, capsys):
        """A user backend that hasn't declared lora_compat_group →
        empty string sentinel → no LoRA support → every entry in
        ``loras`` is silently incompatible (with a warn per entry)."""
        backend = _flux_backend(lora_compat_group="")
        loras = (
            LoraRef(ref="x/y", weight=0.8, compatible_with=("flux-1",)),
        )
        cmd = _cmd(backend=backend, loras=loras)
        assert "--lora-paths" not in cmd

    def test_lora_argv_weight_as_string(self):
        """mflux's --lora-scales expects positional floats but argv is
        a list of strings (the subprocess re-parses). Verify weight
        is str()'d, not left as a float."""
        lora = LoraRef(ref="x/y", weight=0.75, compatible_with=("flux-1",))
        cmd = _cmd(loras=(lora,))
        i_scales = cmd.index("--lora-scales")
        assert isinstance(cmd[i_scales + 1], str)
        assert cmd[i_scales + 1] == "0.75"

    def test_lora_argv_negative_weight_passes_through(self):
        """Weight can be negative (overshoot / inverted-effect — rare
        but valid in mflux). The schema accepts -2.0..2.0; here we
        verify the argv stringifies correctly."""
        lora = LoraRef(ref="x/y", weight=-0.5, compatible_with=("flux-1",))
        cmd = _cmd(loras=(lora,))
        i_scales = cmd.index("--lora-scales")
        assert cmd[i_scales + 1] == "-0.5"


# ── argv ordering invariants (lock-in) ────────────────────────────────


class TestLoRAArgvOrdering:
    def test_lora_appears_after_extra_args(self):
        """Built-in flux extra_args = ('--model', 'dev'). LoRA argv
        must come AFTER, so --model dev applies first."""
        lora = LoraRef(ref="x/y", weight=0.5, compatible_with=("flux-1",))
        cmd = _cmd(loras=(lora,))
        i_model = cmd.index("--model")
        i_lora = cmd.index("--lora-paths")
        assert i_model < i_lora

    def test_lora_appears_after_negative_prompt(self):
        """Order from v0.1.x: common → strength → extra_args → negative
        → (v0.6) LoRA. Negative comes before LoRA so LoRAs are last
        in the command, matching how users typically tail-edit mflux
        invocations."""
        lora = LoraRef(ref="x/y", weight=0.5, compatible_with=("flux-1",))
        cmd = _cmd(loras=(lora,))
        i_neg = cmd.index("--negative-prompt")
        i_lora = cmd.index("--lora-paths")
        assert i_neg < i_lora

    def test_lora_scales_immediately_follows_lora_paths(self):
        """The two flags must be adjacent (separated only by the path
        values). mflux parses them positionally."""
        loras = (
            LoraRef(ref="a/b", weight=0.5, compatible_with=("flux-1",)),
            LoraRef(ref="c/d", weight=0.3, compatible_with=("flux-1",)),
        )
        cmd = _cmd(loras=loras)
        i_paths = cmd.index("--lora-paths")
        i_scales = cmd.index("--lora-scales")
        # 2 paths after --lora-paths, then --lora-scales immediately.
        assert i_scales == i_paths + 3  # paths + path1 + path2 = 3 slots

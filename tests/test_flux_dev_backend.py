"""v0.7.0 (architect §A): FLUX.1-dev backend entry shape lock-in.

The new t2i backend lands as a peer of FLUX-Kontext + Qwen in
BUILTIN_BACKENDS. Field-by-field lock-in here so any future schema
shift surfaces loudly. Coupled with the CLI `_lora_ref_arg` default
widening to `("flux-1", "flux-dev")` — locked here too so a user's
`--lora foo/bar` works on both FLUX backends.
"""
from __future__ import annotations

from imgen.backends import BACKENDS, BUILTIN_BACKENDS, _FLUX_DEV_DRAW_ENHANCE_SYS
from imgen.parser import _lora_ref_arg


class TestFluxDevBackendEntry:
    def test_registered_in_builtin_backends(self):
        assert "flux-dev" in BUILTIN_BACKENDS
        # BACKENDS is the read-only alias pointing at BUILTIN_BACKENDS for
        # call sites that don't need user-TOML extensions.
        assert "flux-dev" in BACKENDS

    def test_binary_is_mflux_generate(self):
        """Plain `mflux-generate` (t2i), NOT `mflux-generate-kontext`
        (which is i2i Kontext)."""
        assert BACKENDS["flux-dev"].binary == "mflux-generate"

    def test_needs_token_true(self):
        """Same HF gated repo class as Kontext — shares the
        `~/.imgen/hf_token` file. load_backend_and_token routes both
        through the same token-validation path."""
        assert BACKENDS["flux-dev"].needs_token is True

    def test_supports_strength_false(self):
        """t2i: no input photo, no strength parameter to apply."""
        assert BACKENDS["flux-dev"].supports_strength is False

    def test_supports_negative_true(self):
        """FLUX.1-dev accepts --negative-prompt via mflux's
        `mflux-generate` binary."""
        assert BACKENDS["flux-dev"].supports_negative is True

    def test_extra_args_model_dev(self):
        """mflux convention — same as Kontext entry. Distinguishes
        from a future schnell variant which would pass
        ('--model', 'schnell')."""
        assert BACKENDS["flux-dev"].extra_args == ("--model", "dev")

    def test_image_flag_populated(self):
        """image_flag stays populated for dataclass-shape consistency
        with the i2i entries. build_mflux_cmd will gate the actual
        argv emission on input_path being not None (step 4)."""
        assert BACKENDS["flux-dev"].image_flag == "--image-path"

    def test_enhance_invariants_empty(self):
        """t2i has no identity anchor to preserve (no input photo). The
        Kontext/Qwen i2i entries use _IDENTITY_ANCHOR_INVARIANTS; the
        draw entry deliberately ships empty per architect §K."""
        assert BACKENDS["flux-dev"].enhance_invariants == ()

    def test_enhance_system_prompt_present(self):
        """t2i-tuned system prompt for the LLM enhancer."""
        sys_prompt = BACKENDS["flux-dev"].enhance_system_prompt
        assert sys_prompt is not None
        assert len(sys_prompt) > 100

    def test_enhance_system_prompt_t2i_specific(self):
        """The system prompt explicitly states 'text-to-image diffusion
        model' and references generating-from-scratch (no input
        photo). Locks the t2i framing — a regression that copy-pasted
        Kontext's i2i prompt would be caught."""
        sys = BACKENDS["flux-dev"].enhance_system_prompt
        assert sys is not None
        assert "text-to-image diffusion" in sys
        assert "no input photo" in sys
        # Must NOT reference Kontext-specific phrasing about preserving
        # the user's "while preserving …" clause — that's i2i-only.
        assert "while preserving" not in sys

    def test_enhance_system_prompt_constant_exported(self):
        """The module-level constant is importable so tests + future
        replay debug can reference exact text."""
        assert (
            _FLUX_DEV_DRAW_ENHANCE_SYS
            == BACKENDS["flux-dev"].enhance_system_prompt
        )

    def test_lora_compat_group_unique(self):
        """flux-dev MUST NOT share the lora_compat_group="flux-1" tag
        with FLUX-Kontext. Until per-LoRA verification proves a given
        Kontext-trained LoRA loads on plain FLUX.1-dev t2i (mirror of
        the v0.6.1 lesson), we keep the compat groups separate."""
        assert BACKENDS["flux-dev"].lora_compat_group == "flux-dev"
        # Sibling groups for comparison — defence against accidental
        # rename in either direction.
        assert BACKENDS["flux"].lora_compat_group == "flux-1"
        assert BACKENDS["qwen"].lora_compat_group == "qwen"

    def test_emit_gated_repo_hint_no_op_on_success(self, capsys):
        """v0.7.1 architect NIT: direct unit tests for the extracted
        helper. Coverage today is incidental via test_draw.py; pure
        functions warrant strict TDD-matrix coverage. Case 1:
        ``failed=[]`` → no-op, no output."""
        from imgen.cmd_helpers import emit_gated_repo_hint_if_failed
        emit_gated_repo_hint_if_failed(
            failed=[], backend_obj=BACKENDS["flux-dev"],
        )
        assert capsys.readouterr().out == ""

    def test_emit_gated_repo_hint_no_op_on_non_gated_backend(self, capsys):
        """Case 2: backend has ``hf_gated_repo=None`` (qwen). Even with
        a non-empty failed list, no hint surfaces — the hint is
        gated-repo-specific."""
        from imgen.cmd_helpers import emit_gated_repo_hint_if_failed
        from pathlib import Path
        emit_gated_repo_hint_if_failed(
            failed=[("draw", Path("/tmp/out.png"), 1)],
            backend_obj=BACKENDS["qwen"],
        )
        assert capsys.readouterr().out == ""

    def test_emit_gated_repo_hint_prints_url_when_both_truthy(self, capsys):
        """Case 3: failed AND hf_gated_repo set → URL surfaces."""
        from imgen.cmd_helpers import emit_gated_repo_hint_if_failed
        from pathlib import Path
        emit_gated_repo_hint_if_failed(
            failed=[("draw", Path("/tmp/out.png"), 1)],
            backend_obj=BACKENDS["flux-dev"],
        )
        out = capsys.readouterr().out
        assert "https://huggingface.co/black-forest-labs/FLUX.1-dev" in out
        assert "GatedRepoError" in out or "401" in out

    def test_hf_gated_repo_populated(self):
        """v0.7.0 post-tag UX-gap: flux-dev declares the HF model URL
        path so cmd_draw can surface the per-repo license hint on
        first-run 401 failures. Cold-install colleagues whose token
        works for Kontext but haven't accepted FLUX.1-dev's separate
        license hit this gap; the cmd_draw post-failure block reads
        ``hf_gated_repo`` to point them at the right page."""
        assert BACKENDS["flux-dev"].hf_gated_repo == "black-forest-labs/FLUX.1-dev"
        # FLUX-Kontext also declares its gated repo for symmetry —
        # accepting Kontext's license is the i2i prerequisite.
        assert BACKENDS["flux"].hf_gated_repo == "black-forest-labs/FLUX.1-Kontext-dev"
        # qwen is open (needs_token=False), no gated repo to declare.
        assert BACKENDS["qwen"].hf_gated_repo is None


class TestLoraRefArgCompatibleWithWidening:
    """v0.7.0 (architect §A): CLI `--lora` default compatible_with
    widens from `("flux-1",)` to `("flux-1", "flux-dev")` so a user's
    `--lora foo/bar` reaches both FLUX backends. User-style TOMLs that
    explicitly declare `compatible_with = ["flux-1"]` stay restrictive
    (locked in test_styles.py / TOML schema path)."""

    def test_cli_default_includes_flux_dev(self):
        ref = _lora_ref_arg("alvarobartt/ghibli-characters-flux-lora")
        assert ref.compatible_with == ("flux-1", "flux-dev")

    def test_cli_default_includes_flux_1(self):
        """Backward-compat: existing Kontext-targeted CLI LoRAs still
        match the Kontext backend's `lora_compat_group="flux-1"` group."""
        ref = _lora_ref_arg("strangerzonehf/Flux-Animeo-v1-LoRA:0.8")
        assert "flux-1" in ref.compatible_with

    def test_cli_default_does_not_include_qwen(self):
        """Qwen LoRAs are a separate ecosystem (different transformer
        shape); the CLI default tuple does NOT widen to include them."""
        ref = _lora_ref_arg("some/qwen-lora")
        assert "qwen" not in ref.compatible_with

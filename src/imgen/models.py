"""v0.8.0 — Model dataclass + canonical BUILTIN_MODELS registry.

Per [[project-v080-design]] §F + §G.1 + §Q commit 4b. A ``Model`` row is
the per-recipe property surface: which engine routes it, its default
steps/guidance, RAM math, LoRA compat group, gated-repo URL. The Engine
(mflux / diffusers_mps) reads ``Model + GenParams`` and dispatches.

This module is the **canonical v0.8 registry source-of-truth**:

* ``BUILTIN_MODELS`` (v0.8.0 commit 4b): literal-declared dict keyed by
  v0.8 canonical names (``flux-kontext``, ``qwen-image-edit-v1``,
  ``flux-dev``, ``flux2-klein-edit-9b``). Replaces the commit-2-3
  derive-from-BUILTIN_BACKENDS path.
* ``_V07_TO_V08_MODEL_RENAMES``: pure data about v0.7 → v0.8 name
  renames. Used by parser ``_resolve_v07_alias`` (for CLI input
  validation) AND by ``backends.get_backend`` (for v0.7-name input
  translation in the back-compat shim). Lives here as the canonical
  registry module to avoid circular import (parser → backends →
  parser). (4b design pre-vet python-reviewer HIGH-1.)

``backends.py`` becomes a thin v0.7-compat facade at 4b: ``BUILTIN_BACKENDS``
is BACKWARD-DERIVED from ``BUILTIN_MODELS`` (keyed by v0.7 names for
test-fixture compatibility per memo §Q + architect HIGH-1), and
``get_backend()`` accepts both v0.7 and v0.8 names. The Model surface
itself is engine-aware (commit 1) and supports the new diffusers_mps
engine path (commit 6).

Field surface locked by ``tests/test_models.py::TestModelDataclassShape``.
Engine-conditional invariants enforced at every instantiation site via
``__post_init__`` — built-in registry, user TOMLs (commit 6+ when
user-TOML schema gains v0.8 fields), test fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "BUILTIN_MODELS",
    "Model",
    "_V07_TO_V08_MODEL_RENAMES",
    "get_model",
    "list_models",
]


# ── v0.7 → v0.8 model name renames (canonical home, 4b moved from parser.py) ──
#
# Pure data; no imports of imgen modules — anyone can import this freely.
# parser.py (CLI `--model` validation) and backends.py (back-compat shim
# `get_backend()`) both consume this; locating here avoids circular
# import (4b design pre-vet python-reviewer HIGH-1).
#
# Only TWO built-ins moved in v0.8.0: `flux` → `flux-kontext` (honest:
# it's FLUX.1-Kontext-dev, not generic flux) and `qwen` →
# `qwen-image-edit-v1` (honest: v1 of Qwen-Image-Edit, distinct from
# the 2512 family). Other built-ins (`flux-dev`, `flux2-klein-edit-9b`)
# already had honest v0.8-style names so no rename row needed.
_V07_TO_V08_MODEL_RENAMES: Mapping[str, str] = MappingProxyType({
    "flux": "flux-kontext",
    "qwen": "qwen-image-edit-v1",
})


# ── Model dataclass (commit 1, unchanged at 4b) ────────────────────────


@dataclass(frozen=True, slots=True)
class Model:
    """Per-recipe property surface.

    Engine routing is mandatory; engine-specific fields (``binary`` for
    mflux, ``repo`` for diffusers_mps) are optional at dataclass level
    but enforced as required by ``__post_init__`` based on engine.

    All v0.8.0 NEW fields default to safe values so v0.7-shaped
    instantiation calls (via the ``Backend = Model`` facade alias in
    backends.py) keep working through the v0.8.x deprecation window.

    Tied design memo: [[project-v080-design]] §F.
    """

    # — Engine routing (required) —
    engine: str

    # — Engine-conditional optional fields (validated in __post_init__) —
    binary: str | None = None
    repo: str | None = None
    extra_args: tuple[str, ...] = ()
    image_flag: str | None = None
    cpu_offload_threshold_mp: float = 2.0

    # — Capability flags (engine-agnostic) —
    supports_strength: bool = False
    supports_negative: bool = False
    needs_token: bool = False
    lora_compat_group: str = ""
    hf_gated_repo: str | None = None

    # — v0.8 NEW: per-model param defaults (replaces v0.7.14 uniform numbers) —
    default_steps: int = 20
    default_guidance: float = 3.5
    min_guidance: float = 0.0
    max_guidance: float = 10.0
    supported_quants: tuple[int, ...] = (3, 4, 5, 6, 8)
    omit_quantize: bool = False
    # Tuple-of-tuples rather than dict — immutable so frozen=True actually
    # IS frozen. Engine code converts to dict at the call boundary.
    param_overrides: tuple[tuple[str, object], ...] = ()

    # — v0.8 NEW: per-model RAM math (replaces v0.7.14 uniform 5 GB/MP) —
    # ram_baseline_gb=0.0 / ram_slope_gb_per_mp=0.0 are SENTINELS that
    # fail loudly in __post_init__ — Model rows must declare them.
    ram_baseline_gb: float = 0.0
    ram_slope_gb_per_mp: float = 0.0
    encoder_ram_gb: float = 0.0

    # — Enhancer (carried from v0.5) —
    enhance_system_prompt: str | None = None
    enhance_invariants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Engine-conditional invariants. Fires at every Model
        instantiation — built-in registry, user TOMLs, test fixtures —
        so the guarantee is uniform across registration sources."""
        if self.engine == "mflux":
            if self.binary is None:
                raise ValueError(
                    f"Model with engine='mflux' requires binary= (got None)"
                )
        elif self.engine == "diffusers_mps":
            if self.repo is None:
                raise ValueError(
                    f"Model with engine='diffusers_mps' requires repo= (got None)"
                )
        else:
            raise ValueError(
                f"engine={self.engine!r} not in {{'mflux', 'diffusers_mps'}}"
            )
        if self.ram_baseline_gb <= 0.0:
            raise ValueError(
                f"Model row missing ram_baseline_gb (got {self.ram_baseline_gb}) "
                "— registry author must declare; sentinel 0.0 fails loudly "
                "rather than silently letting preflight under-estimate."
            )
        if self.ram_slope_gb_per_mp <= 0.0:
            raise ValueError(
                f"Model row missing ram_slope_gb_per_mp (got {self.ram_slope_gb_per_mp}) "
                "— registry author must declare."
            )


# ── Enhancer system prompts (moved from backends.py at 4b) ─────────────
#
# These are FLUX-/Qwen-specific tuning text used by the LLM prompt
# enhancer (v0.5+). They live with the Model registry now because they
# are properties of the model, not of the v0.7 "Backend" surface.
# backends.py continues to re-export them for any v0.7 test that
# imported them directly.

_FLUX_KONTEXT_ENHANCE_SYS = (
    "You expand image-editing prompts for FLUX.1 Kontext, an image-"
    "conditioning model that restyles input photos while preserving "
    "identity, pose, and composition. Take the user prompt and expand "
    "it to 40-60 tokens. "
    "CRITICAL: you MUST preserve the entire 'while preserving …' "
    "clause from the user prompt VERBATIM. Keep every word inside that "
    "clause exactly as written — particularly identity anchors such as "
    "'facial identity', 'exact facial features', or 'recognizable "
    "expression'. Do NOT replace these anchors with synonyms or "
    "alternative preservation language (e.g. NEVER substitute 'overall "
    "composition' or 'relative position of subjects' for the identity "
    "anchor). "
    "Add specific stylistic descriptors (lighting, color palette, art "
    "technique, materials) at the START or END, not inside the "
    "preserving clause. Do NOT invent objects, scenes, or characters "
    "not in the user prompt — expand existing details only. NEVER "
    "describe the input photo's content — Kontext sees it directly. "
    "Output ONLY the expanded prompt with no preamble, no quotes, "
    "no explanation."
)

_QWEN_EDIT_ENHANCE_SYS = (
    "You expand instruction-style edit prompts for Qwen-Image-Edit. "
    "Use imperative verbs ('transform', 'restyle', 'apply'). Keep the "
    "output under 40 tokens — Qwen-Edit prefers shorter directives "
    "than FLUX. "
    "CRITICAL: preserve the entire 'while preserving …' clause from "
    "the user prompt VERBATIM, including identity anchors like "
    "'facial identity', 'exact facial features', or 'recognizable "
    "expression'. Do NOT swap these for synonyms. "
    "Do NOT invent objects, scenes, or characters not in the user "
    "prompt — expand existing details only. NEVER describe the input "
    "photo's content. Output ONLY the expanded prompt with no preamble, "
    "no quotes, no explanation."
)

_FLUX_DEV_DRAW_ENHANCE_SYS = (
    "You are a prompt engineer for FLUX.1, a text-to-image diffusion "
    "model. Expand the user's brief description into a richer, "
    "visually-detailed image prompt suitable for generation from "
    "scratch (no input photo). "
    "Add concrete detail to: subject (specific appearance, clothing, "
    "expression, posture), composition (camera angle, framing, "
    "subject placement), lighting (source, quality, direction, color "
    "temperature), color palette (dominant tones, accents, mood), "
    "and art style (medium, technique, era, named artists or schools "
    "where the user's prompt implies one). "
    "Stay faithful to the user's intent — do NOT invent a different "
    "subject, swap genders, change species, or relocate the scene. "
    "Expand the existing details; don't replace them. "
    "Target 40-70 tokens. Output ONLY the expanded prompt with no "
    "preamble, no quotes, no explanation, no 'Here is the expanded "
    "prompt:' framing."
)

# Multi-substring identity-anchor invariants — see v0.5 enhance.py.
# Per-style-family anchor: one of these substrings must survive the
# enhance pass for the i2i identity preservation contract to hold.
_IDENTITY_ANCHOR_INVARIANTS: tuple[str, ...] = (
    "facial identity",
    "exact facial features",
    "recognizable expression",
)


# ── BUILTIN_MODELS — literal-declared registry (v0.8.0 commit 4b) ──────
#
# Replaces the commit-2-3 derived view. Each row is the v0.8 canonical
# name keyed to a literal Model() construction. RAM values populated
# inline from the v0.7.14 baseline table (preserved in commit-history
# context for `_V07_BACKEND_RAM_DEFAULTS`); commit 8 (per §Q) will
# tune per-Model ram_baseline_gb / slope based on real measurements.
# Per-Model param defaults (default_steps / guidance / min_guidance)
# stay at dataclass defaults at 4b; commit 7 wires the per-Model
# values into the resolver path per §G.1 and §M.

BUILTIN_MODELS: dict[str, Model] = {
    # FLUX.1-Kontext-dev — i2i style transfer (v0.7 name: flux).
    "flux-kontext": Model(
        engine="mflux",
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
        enhance_system_prompt=_FLUX_KONTEXT_ENHANCE_SYS,
        enhance_invariants=_IDENTITY_ANCHOR_INVARIANTS,
        lora_compat_group="flux-1",
        hf_gated_repo="black-forest-labs/FLUX.1-Kontext-dev",
        ram_baseline_gb=9.0,
        ram_slope_gb_per_mp=5.0,
        encoder_ram_gb=0.0,
        # v0.8.0 commit 7 (§M): per-Model param defaults applied
        # through the resolver. FLUX.1-Kontext needs CFG > 0 to
        # produce non-blurry output; min_guidance=1.0 hard-floor.
        default_steps=20,
        default_guidance=3.5,
        min_guidance=1.0,
        max_guidance=10.0,
    ),

    # Qwen-Image-Edit-2509 — open-license i2i (v0.7 name: qwen).
    "qwen-image-edit-v1": Model(
        engine="mflux",
        binary="mflux-generate-qwen-edit",
        needs_token=False,
        image_flag="--image-paths",
        supports_strength=False,
        supports_negative=False,
        extra_args=("--model", "qwen"),
        enhance_system_prompt=_QWEN_EDIT_ENHANCE_SYS,
        enhance_invariants=_IDENTITY_ANCHOR_INVARIANTS,
        lora_compat_group="qwen",
        hf_gated_repo=None,
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=5.0,
        encoder_ram_gb=7.0,  # Qwen2.5-VL encoder ~7 GB peak
        # Qwen-Image-Edit converges slower than FLUX (instruction-
        # following architecture, denser cross-attention). 30 steps
        # is the model-card recommended floor for quality.
        default_steps=30,
        default_guidance=4.0,
        min_guidance=0.0,
        max_guidance=10.0,
    ),

    # FLUX.1-dev — t2i default for `imgen draw` (name unchanged at 4b).
    "flux-dev": Model(
        engine="mflux",
        binary="mflux-generate",
        needs_token=True,
        image_flag="--image-path",  # dataclass-shape consistency; build_cmd
                                    # gates emission on input_path being set.
        supports_strength=False,
        supports_negative=True,
        extra_args=("--model", "dev"),
        enhance_system_prompt=_FLUX_DEV_DRAW_ENHANCE_SYS,
        enhance_invariants=(),  # t2i: no identity-anchor contract
        lora_compat_group="flux-dev",
        hf_gated_repo="black-forest-labs/FLUX.1-dev",
        ram_baseline_gb=9.0,
        ram_slope_gb_per_mp=5.0,
        encoder_ram_gb=0.0,
        # FLUX.1-dev canonical: 20 steps, 3.5 guidance. min_guidance=1.0
        # because dev is NOT a distilled model — needs real CFG.
        default_steps=20,
        default_guidance=3.5,
        min_guidance=1.0,
        max_guidance=10.0,
    ),

    # FLUX.2-klein-edit-9b — Hires-Fix refine default (name unchanged at 4b).
    "flux2-klein-edit-9b": Model(
        engine="mflux",
        binary="mflux-generate-flux2-edit",
        needs_token=True,
        image_flag="--image-paths",
        supports_strength=False,
        supports_negative=False,  # FLUX.2 family deliberately dropped CFG/neg
        extra_args=("-m", "flux2-klein-9b"),
        enhance_system_prompt=None,
        enhance_invariants=(),
        lora_compat_group="flux2-klein-9b",
        hf_gated_repo="black-forest-labs/FLUX.2-klein-9B",
        ram_baseline_gb=14.0,
        ram_slope_gb_per_mp=5.5,
        encoder_ram_gb=0.0,
        # v0.8.0 commit 7: FLUX.2-klein distilled — mflux 0.17.5's
        # `mflux-generate-flux2-edit` ONLY accepts `--guidance 1.0`
        # and dies with a usage error on anything else. Pre-commit-7
        # cmd_refine had a hardcoded ``args.guidance = 1.0`` override
        # (refine.py:238); commit 7 removes that pin and replaces it
        # with min_guidance=max_guidance=1.0 → MfluxEngine.validate
        # rejects mismatches. The new approach scales: any future
        # FLUX.2 variant gets the same enforcement without per-binary
        # cmd_* edits.
        default_steps=20,
        default_guidance=1.0,
        min_guidance=1.0,
        max_guidance=1.0,
    ),
}


# ── Strict v0.8 accessors (canonical Engine-layer API) ─────────────────
#
# Per architect M-1 (4b design pre-vet): get_model / list_models are the
# new canonical API and are STRICTLY v0.8-only — they take v0.8 keys
# and raise on anything else. The v0.7-name translation surface lives
# in backends.get_backend (v0.7-compat shim) so the alias layer is
# single-source. New Engine-layer code (commit 6+) uses get_model;
# existing call sites continue to use get_backend through 4b.


def get_model(name: str) -> Model:
    """Strict v0.8-only Model lookup. Raises KeyError on unknown names
    or v0.7 spellings. v0.7-name translation lives in
    ``backends.get_backend`` (see architect 4b pre-vet M-1)."""
    if name not in BUILTIN_MODELS:
        available = ", ".join(sorted(BUILTIN_MODELS.keys()))
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return BUILTIN_MODELS[name]


def list_models() -> list[str]:
    """Sorted v0.8 canonical names of built-in Models. User-TOML Models
    are NOT included at 4b — user TOMLs go through the v0.7 Backend
    facade (see backends.py ``list_backends``). Commit 6+ Engine layer
    wires user-TOML-derived Models into this list."""
    return sorted(BUILTIN_MODELS.keys())

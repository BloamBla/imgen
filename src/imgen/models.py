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
from typing import Literal, Mapping

__all__ = [
    "BUILTIN_MODELS",
    "Model",
    "TrainingConfig",
    "TrainingTargetSpec",
    "VideoConfig",
    "_KLEIN_4B_TARGET_MODULES",
    "_V07_TO_V08_MODEL_RENAMES",
    "_VALID_LORA_RANKS",
    "_VALID_QUANTIZE_TRAIN",
    "_VALID_TRAIN_RESOLUTIONS",
    "get_model",
    "list_models",
]


# ── VideoConfig (v0.9 commit 1, [[project-v090-design]] §C) ────────────
#
# Nested video-specific config on ``Model.video``. Absent (None) ⇒
# image Model. Present ⇒ video Model. Per architect §R.1 HIGH-1:
# flat field expansion (7 new top-level Model fields) was rejected
# because (a) image-only user TOMLs would carry noise fields
# meaningless for image; (b) future audio/3d Models would compound
# the flat bloat. Nested keeps Model at fixed top-level cardinality.
#
# frozen + slots so Model can stay frozen+hashable with this nested.
# All fields are hashable types (int / float / bool / tuple of str).


# v0.9.3 C2 (B-1 closure): video-pipeline allowlist. The Engine reads
# ``model.video.pipeline_class`` and threads it into the runner
# payload; both the parent-side construction (here) and the runner-
# side trust boundary (``_diffusers_runner._PIPELINE_CLASS_ALLOWLIST``)
# enforce a literal allowlist to avoid introspection-style attacks.
#
# This parent-side set is a STRICT subset of the runner-side set:
# the runner also allows ``"DiffusionPipeline"`` for the v0.8 image
# fallback, but a VideoConfig declaring an image pipeline class would
# mis-route inside the runner — reject one layer earlier here.
#
# To add a new video pipeline: extend BOTH this set and the runner
# set, plus add the matching ``from diffusers import …`` line in
# ``_resolve_pipeline_class``. The test
# ``test_v093_pipeline_class.TestAllowlistSubsetInvariant`` locks
# the subset relationship.
_VIDEO_PIPELINE_CLASS_ALLOWLIST: frozenset[str] = frozenset({
    "LTXPipeline",                 # v0.9.0 t2v
    "LTXImageToVideoPipeline",     # v0.9.3 i2v
})


@dataclass(frozen=True, slots=True)
class VideoConfig:
    """v0.9 video-specific Model config. Absent on a Model ⇒ image;
    present ⇒ video output.

    Tied design memo: [[project-v090-design]] §C.
    """

    default_num_frames: int           # LTX: 25 (~1 sec @ 24 fps)
    default_fps: int                  # LTX: 24
    max_num_frames: int               # LTX: 257 (~10.7 sec @ 24 fps; paper cap)
    num_frames_alignment: int = 8     # LTX: 8 (must be 8k+1 frames)
    num_frames_offset: int = 1        # LTX: 1 (the +1 in 8k+1)
    supports_video_codecs: tuple[str, ...] = ("libx264",)
    force_cpu_offload: bool = True    # video defaults to forced offload (T5-XXL pressure)
    encoder_ram_gb: float = 3.0       # T5-XXL transient peak when offloaded; not optional for video
    # v0.9.3 C2 (B-1 closure): pipeline class threaded into the runner
    # payload. Default ``"LTXPipeline"`` preserves v0.9.0 t2v
    # behaviour for existing Model rows. v0.9.3 i2v opts in by
    # constructing a derivative VideoConfig with
    # ``pipeline_class="LTXImageToVideoPipeline"``. Validated against
    # :data:`_VIDEO_PIPELINE_CLASS_ALLOWLIST` in __post_init__.
    pipeline_class: str = "LTXPipeline"

    def __post_init__(self) -> None:
        if self.default_num_frames < 9:
            raise ValueError(
                f"VideoConfig.default_num_frames={self.default_num_frames} "
                "too low (minimum 9 for usable temporal sampling)"
            )
        if self.default_fps not in {24, 25, 30}:
            raise ValueError(
                f"VideoConfig.default_fps={self.default_fps} not in "
                "{24, 25, 30} — supported video rates for v0.9.0"
            )
        if self.max_num_frames < self.default_num_frames:
            raise ValueError(
                f"VideoConfig.max_num_frames={self.max_num_frames} must be "
                f">= default_num_frames={self.default_num_frames}"
            )
        if self.num_frames_alignment < 1:
            raise ValueError(
                f"VideoConfig.num_frames_alignment={self.num_frames_alignment} "
                "must be >= 1"
            )
        if not self.supports_video_codecs:
            raise ValueError(
                "VideoConfig.supports_video_codecs must list at least one "
                "codec (typically 'libx264')"
            )
        if self.encoder_ram_gb <= 0:
            raise ValueError(
                f"VideoConfig.encoder_ram_gb={self.encoder_ram_gb} must be > 0 "
                "(text encoder peak RAM is load-bearing for video preflight)"
            )
        if self.pipeline_class not in _VIDEO_PIPELINE_CLASS_ALLOWLIST:
            raise ValueError(
                f"VideoConfig.pipeline_class={self.pipeline_class!r} not in "
                f"video allowlist {sorted(_VIDEO_PIPELINE_CLASS_ALLOWLIST)!r}. "
                "Adding a new video pipeline requires extending the parent "
                "and runner allowlists in lockstep (B-1 closure pattern)."
            )


# ── TrainingConfig + TrainingTargetSpec (v0.10 commit 1, [[project-v100-design]] §C) ──
#
# Nested training config on ``Model.training``. Absent (None) ⇒ Model
# cannot be a LoRA-training target. Present ⇒ Model is a valid
# ``imgen train --base <name>`` target. Mirrors the v0.9 VideoConfig
# nested-config pattern.
#
# §R.1 ROUND-1 CLOSURES (canonical post-amendment shape):
# * NO ``mflux_train_model`` field — derived from registry key in
#   cmd_train (architect C-2 closure).
# * NO ``_TRAINING_BASE_ALLOWLIST`` constant — ``Model.training is not
#   None`` IS the signal for trainability.
# * ``target_modules: tuple[TrainingTargetSpec, ...]`` matches real
#   mflux ``lora_layers.targets[]`` JSON shape (verified by reading
#   ``mflux/models/common/training/_example/train.json``). Each spec
#   has its OWN rank (NOT a single top-level rank).
# * ``default_epochs`` not ``default_steps_per_image`` — semantically
#   correct per architect M-1 (total_steps = epochs × len(dataset) ×
#   batch_size).
# * ``_KLEIN_4B_TARGET_MODULES`` lifted to module-level constant for
#   the klein-4b BUILTIN_MODELS row (commit 2) to reference by NAME —
#   single source of truth, closes the B-1 anti-pattern that v0.9.3
#   fixed for ``pipeline_class``.


# Shared rank/quant/resolution sets — consumed by BOTH ``__post_init__``
# AND argparse ``choices=`` (commit 4) so a future v0.10.x extension
# (e.g. rank=128) lands in one place without drift. §R.1 python H-13
# closure.
_VALID_LORA_RANKS: frozenset[int] = frozenset({4, 8, 16, 32, 64})
_VALID_QUANTIZE_TRAIN: frozenset[int] = frozenset({3, 4, 5, 6, 8})
_VALID_TRAIN_RESOLUTIONS: frozenset[int] = frozenset(
    {256, 384, 512, 768, 1024}
)


# Module-path regex: starts with letter/underscore, contains
# alphanumeric/underscore/dot/optional ``{block}`` placeholder. The
# placeholder appears at most once and is bracketed by dotted-path
# segments. Excludes control bytes and shell metachars by construction
# — anything outside ``[\w.{}]`` rejects.
import re as _re  # noqa: E402 — local import to avoid widening the
# module's top-level import surface (this is the only ``re`` consumer
# in models.py and it stays scoped to the TrainingTargetSpec validator).
_TRAINING_TARGET_MODULE_PATH_RE = _re.compile(
    r"^[a-zA-Z_][\w.]*(?:\{block\}[\w.]*)?$"
)


@dataclass(frozen=True, slots=True)
class TrainingTargetSpec:
    """v0.10 — one entry in ``TrainingConfig.target_modules``.

    Matches mflux ``lora_layers.targets[]`` JSON shape verified at
    ``mflux/models/common/training/_example/train.json``: each entry is
    ``{module_path, blocks: {start, end}, rank}`` and renders to mflux
    via the ``{block}``-templated module path substituted across the
    block range.

    Frozen + slots + hashable so the parent ``TrainingConfig`` can stay
    frozen+hashable too. All fields are hashable types: ``str`` /
    ``tuple[int, int] | None`` / ``int``.

    Tied design memo: [[project-v100-design]] §C.
    """

    module_path: str               # e.g. "transformer_blocks.{block}.attn.to_q"
    blocks: tuple[int, int] | None # (start, end) inclusive-exclusive, or None
    rank: int                      # per-target LoRA rank; must be in _VALID_LORA_RANKS

    def __post_init__(self) -> None:
        # 1. Empty / control-byte / shape rejection on module_path.
        if not self.module_path:
            raise ValueError(
                "TrainingTargetSpec.module_path is empty"
            )
        if not _TRAINING_TARGET_MODULE_PATH_RE.fullmatch(self.module_path):
            raise ValueError(
                f"TrainingTargetSpec.module_path={self.module_path!r} "
                "must match ^[a-zA-Z_][\\w.]*(\\{block\\}[\\w.]*)?$ — "
                "alphanumeric/underscore/dot path with optional single "
                "``{block}`` placeholder; control bytes + shell "
                "metachars excluded by construction."
            )

        # 2. Cross-rule: {block} placeholder iff blocks is not None.
        #    The validator catches BOTH directions of inconsistency.
        has_placeholder = "{block}" in self.module_path
        if has_placeholder and self.blocks is None:
            raise ValueError(
                f"TrainingTargetSpec.module_path={self.module_path!r} "
                "contains ``{block}`` placeholder but blocks=None — "
                "spec is unrenderable. Provide blocks=(start, end)."
            )
        if not has_placeholder and self.blocks is not None:
            raise ValueError(
                f"TrainingTargetSpec.module_path={self.module_path!r} "
                f"has no ``{{block}}`` placeholder but blocks={self.blocks!r} "
                "— no placeholder to substitute into. Either add "
                "``{block}`` to the path or set blocks=None."
            )

        # 3. Validate blocks range when present.
        if self.blocks is not None:
            start, end = self.blocks
            if start < 0:
                raise ValueError(
                    f"TrainingTargetSpec.blocks={self.blocks!r} has negative "
                    f"start; valid range is start >= 0"
                )
            if start > end:
                raise ValueError(
                    f"TrainingTargetSpec.blocks={self.blocks!r} has "
                    f"start > end; valid range is start <= end"
                )

        # 4. Rank allowlist.
        if self.rank not in _VALID_LORA_RANKS:
            raise ValueError(
                f"TrainingTargetSpec.rank={self.rank} not in "
                f"_VALID_LORA_RANKS={sorted(_VALID_LORA_RANKS)!r}"
            )


# ── Klein-4b canonical target modules (module-level constant) ─────────
#
# Per [[project-colleague-lora-training-2026-05-27]] proven recipe +
# FLUX.2 transformer architecture grep
# (``mflux/models/flux2/weights/flux2_lora_mapping.py:336`` for the
# ``transformer_blocks.{block}.attn.*`` path shape;
# ``mflux/models/flux2/model/flux2_transformer/transformer.py:20`` for
# the num_single_layers=20 constant).
#
# Klein-4b architecture:
# * transformer_blocks #38 (double-stream) with attn q/k/v/to_out targets
# * single_transformer_blocks #20 (single-stream) with attn.to_out target
#
# BUILTIN_MODELS klein-4b row (commit 2) will reference this by NAME so
# the schema-and-content split lives in one place. Adding a 2nd
# training base = new module-level constant + new BUILTIN_MODELS row.
_KLEIN_4B_TARGET_MODULES: tuple[TrainingTargetSpec, ...] = (
    TrainingTargetSpec(
        module_path="transformer_blocks.{block}.attn.to_q",
        blocks=(0, 38),
        rank=16,
    ),
    TrainingTargetSpec(
        module_path="transformer_blocks.{block}.attn.to_k",
        blocks=(0, 38),
        rank=16,
    ),
    TrainingTargetSpec(
        module_path="transformer_blocks.{block}.attn.to_v",
        blocks=(0, 38),
        rank=16,
    ),
    TrainingTargetSpec(
        module_path="transformer_blocks.{block}.attn.to_out.0",
        blocks=(0, 38),
        rank=16,
    ),
    TrainingTargetSpec(
        module_path="single_transformer_blocks.{block}.attn.to_out",
        blocks=(0, 20),
        rank=16,
    ),
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """v0.10 training-specific Model config. Absent (Model.training is
    None) ⇒ Model cannot be a LoRA-training target. Present ⇒ Model is
    a valid ``imgen train --base <name>`` target.

    Tied design memo: [[project-v100-design]] §C.
    """

    # ── Per-Model training peak (preflight gate) ──
    # Sentinel 0.0 fails loudly — preflight gate is load-bearing on
    # this value. Calibrated from smoke runs; klein-4b on M5 Pro 48 GB
    # at q4/rank-16/low_ram peaks ~26-30 GB resident (colleague). M2 Pro
    # 32 GB smoke at pre-tag gate refines this.
    training_peak_ram_gb: float

    # ── Training defaults (mflux-train JSON config seed) ──
    default_lora_rank: int = 16
    default_max_resolution: int = 512
    default_quantize: int = 4
    default_epochs: int = 80        # total_steps = default_epochs × len(dataset) × batch_size
    default_low_ram: bool = True

    # ── Optimizer (AdamW lr=1e-4 colleague-validated default) ──
    optimizer_name: str = "AdamW"
    optimizer_lr: float = 1e-4

    # ── Per-Model target modules (canonical klein-4b set lives in
    #    _KLEIN_4B_TARGET_MODULES module constant) ──
    target_modules: tuple[TrainingTargetSpec, ...] = ()

    # ── Monitoring frequency ──
    # Colleague's recipe used preview_frequency=10 which BLOATED wall
    # from a theoretical ~5h to actual 10h. imgen default is 100
    # (sparse) to keep the M2 Pro 32 GB overnight envelope reasonable.
    default_preview_frequency: int = 100

    def __post_init__(self) -> None:
        if self.training_peak_ram_gb <= 0.0:
            raise ValueError(
                f"TrainingConfig.training_peak_ram_gb={self.training_peak_ram_gb} "
                "must be > 0 — sentinel 0 means Model author forgot to declare; "
                "preflight gate is load-bearing on this value."
            )
        if self.default_lora_rank not in _VALID_LORA_RANKS:
            raise ValueError(
                f"TrainingConfig.default_lora_rank={self.default_lora_rank} "
                f"not in _VALID_LORA_RANKS={sorted(_VALID_LORA_RANKS)!r}"
            )
        if self.default_quantize not in _VALID_QUANTIZE_TRAIN:
            raise ValueError(
                f"TrainingConfig.default_quantize={self.default_quantize} "
                "not in mflux-train's quantize choices "
                f"_VALID_QUANTIZE_TRAIN={sorted(_VALID_QUANTIZE_TRAIN)!r} "
                "(note 0 NOT in set — mflux-train doesn't accept bf16)"
            )
        if self.default_max_resolution not in _VALID_TRAIN_RESOLUTIONS:
            raise ValueError(
                f"TrainingConfig.default_max_resolution={self.default_max_resolution} "
                f"not in _VALID_TRAIN_RESOLUTIONS={sorted(_VALID_TRAIN_RESOLUTIONS)!r}"
            )
        # §R.1 architect M-1 closure: default_epochs (renamed from
        # default_steps_per_image). Floor 10 — under-trains below.
        if self.default_epochs < 10:
            raise ValueError(
                f"TrainingConfig.default_epochs={self.default_epochs} "
                "too low — under-trains; minimum 10 (colleague baseline 80)"
            )
        if self.optimizer_name not in {"AdamW", "Adafactor"}:
            raise ValueError(
                f"TrainingConfig.optimizer_name={self.optimizer_name!r} "
                "not in {'AdamW', 'Adafactor'}"
            )
        # §R.1 python M-5 closure: tightened from 0..1 to 1e-6..1e-2.
        if not (1e-6 <= self.optimizer_lr <= 1e-2):
            raise ValueError(
                f"TrainingConfig.optimizer_lr={self.optimizer_lr} "
                "out of [1e-6, 1e-2] range — realistic LoRA-finetune envelope"
            )
        if not self.target_modules:
            raise ValueError(
                "TrainingConfig.target_modules is empty — mflux-train "
                "requires at least one TrainingTargetSpec. Klein-4b's "
                "validated set lives in module constant "
                "_KLEIN_4B_TARGET_MODULES."
            )
        if self.default_preview_frequency < 1:
            raise ValueError(
                f"TrainingConfig.default_preview_frequency={self.default_preview_frequency} "
                "must be >= 1"
            )


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

    # — v0.9 commit 1: nested video config (None ⇒ image Model) —
    # Forward-typed via PEP 563 (`from __future__ import annotations`
    # at top of file) so this annotation references the class declared
    # above without import-order pain.
    video: VideoConfig | None = None

    # — v0.10 commit 1: nested training config (None ⇒ not trainable) —
    # Mirrors the v0.9 video pattern. ``Model.training is not None``
    # IS the signal for "this Model is a LoRA-training target" (no
    # separate _TRAINING_BASE_ALLOWLIST constant per §R.1 architect
    # C-2 closure).
    training: TrainingConfig | None = None

    @property
    def output_type(self) -> Literal["image", "video"]:
        """Derived: ``"video"`` if a VideoConfig is attached, else
        ``"image"``. Used by iteration_dryrun_display (§H), runner
        payload routing (§F), history command field (§J)."""
        return "video" if self.video is not None else "image"

    @property
    def training_supported(self) -> bool:
        """Derived: ``True`` if this Model has a TrainingConfig
        attached, else ``False``. Used by ``imgen train --base``
        validator + ``imgen doctor`` listing (commits 4, 8).
        [[project-v100-design]] §C."""
        return self.training is not None

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
        # v0.9 commit 1 — Model × VideoConfig cross-rules per §C
        if self.video is not None:
            if self.engine != "diffusers_mps":
                raise ValueError(
                    f"VideoConfig only supported with engine='diffusers_mps' "
                    f"(got engine={self.engine!r})"
                )
            # v0.9 video Models bypass the cpu_offload_threshold_mp
            # mechanism — force_cpu_offload on VideoConfig is the
            # single source of truth. Non-default threshold on a
            # video Model is double-encoded behaviour (footgun).
            if self.cpu_offload_threshold_mp != 2.0:
                raise ValueError(
                    "video Models must use VideoConfig.force_cpu_offload; "
                    "cpu_offload_threshold_mp is image-only "
                    "(leave at the default 2.0 for video Models)"
                )

        # v0.10 commit 1 — Model × TrainingConfig cross-rules per
        # [[project-v100-design]] §C. v0.10.0 ships only mflux-based
        # training; diffusers_mps Models (video etc.) stay
        # inference-only. Adding training to a non-mflux engine
        # requires re-running design review per the locked surface.
        if self.training is not None:
            if self.engine != "mflux":
                raise ValueError(
                    f"TrainingConfig only supported with engine='mflux' "
                    f"(got engine={self.engine!r}). v0.10.0 does not "
                    "train via diffusers_mps engine; video/diffusers "
                    "Models stay inference-only."
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
        # v0.8.0 commit 8 (§L): RAM math separated into weights (scales
        # with quantize) + activations (scales with megapixels) +
        # encoder + engine overhead. Calibration anchor: Q8 1MP target
        # ≈ 18 GB matching v0.7.7 real measurement on M2 Pro 32 GB →
        # 13.5*1 + 4.0*1 + 0 + 0.5 = 18.0 ✓
        ram_baseline_gb=13.5,
        ram_slope_gb_per_mp=4.0,
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
        # v0.8.0 commit 8 (§L): Qwen-Image-Edit has the Qwen2.5-VL
        # encoder loaded one-time during prompt-encode (~7 GB peak).
        # Activations slope slightly higher than FLUX due to denser
        # cross-attention in the instruction-following arch.
        # Q8 1MP target: 13*1 + 4.5*1 + 7 + 0.5 = 25.0 (matches the
        # v0.7.14 calibration row for ("qwen", 8): 25).
        ram_baseline_gb=13.0,
        ram_slope_gb_per_mp=4.5,
        encoder_ram_gb=7.0,
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
        # FLUX.1-dev shares the FLUX.1 transformer envelope with
        # flux-kontext; same calibration applies.
        ram_baseline_gb=13.5,
        ram_slope_gb_per_mp=4.0,
        encoder_ram_gb=0.0,
        # FLUX.1-dev canonical: 20 steps, 3.5 guidance. min_guidance=1.0
        # because dev is NOT a distilled model — needs real CFG.
        default_steps=20,
        default_guidance=3.5,
        min_guidance=1.0,
        max_guidance=10.0,
    ),

    # LTX-Video — v0.9.0 t2v built-in (§K + §L). Pulled forward from
    # commit 9 in lockstep with cmd_video so the parser's
    # ``--model ltx-video`` default resolves. Doctor RAM-forecast row
    # for video Models still lands in commit 9; commit 7 keeps this
    # row minimal (no doctor extension yet).
    #
    # Trust note (security §R.1 MED-5): Lightricks Ltd. is a real
    # corporate publisher; LTX-Video has been in diffusers ≥0.32 with
    # significant community surface. Trust assumption same as
    # Tencent/Qwen rows: open-weights research model from an
    # established source.
    #
    # supported_quants=() — LTX is bf16-only at v0.9.0; diffusers
    # doesn't ship MLX-style quantization for LTX. Empty tuple keeps
    # the doctor table from rendering meaningless q3/q4/etc rows.
    "ltx-video": Model(
        engine="diffusers_mps",
        repo="Lightricks/LTX-Video",
        needs_token=False,
        supports_negative=True,
        cpu_offload_threshold_mp=2.0,  # image-only default — ignored for video
        # §L LTX envelope on M2 Pro 32 GB: baseline=10 (T5 offloaded
        # transformer+VAE+activations); slope=4 GB/MP. No quantize
        # term in the video formula — flat bf16 weights.
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=4.0,
        encoder_ram_gb=18.0,  # ignored when model.video is set
        # Per-Model param defaults (§K).
        default_steps=50,
        default_guidance=3.0,
        min_guidance=1.0,
        max_guidance=10.0,
        supported_quants=(),
        enhance_system_prompt=None,  # §S.4 — LTX has no enhancer at v0.9.0
        enhance_invariants=(),
        lora_compat_group="ltx-video",  # reserved for v0.9.x video LoRA
        hf_gated_repo=None,
        video=VideoConfig(
            default_num_frames=25,
            default_fps=24,
            max_num_frames=257,
            num_frames_alignment=8,
            num_frames_offset=1,
            supports_video_codecs=("libx264",),
            force_cpu_offload=True,  # mandatory on M2 Pro 32 GB
            encoder_ram_gb=3.0,      # T5-XXL transient peak when offloaded
        ),
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
        # v0.8.0 commit 8 (§L): calibrated from v0.7.7 real-mflux
        # measurements on M2 Pro 32 GB:
        #   Q4 1536² (2.36 MP) → 23 GB resident peak
        #   Q4 2048² (4.19 MP) → 30 GB total memory pressure
        # Solving the formula `baseline*0.5 + slope*MP + 0.5` = both:
        #   slope = (30 - 23) / (4.19 - 2.36) ≈ 4.0
        #   baseline = (23 - 4*2.36 - 0.5) / 0.5 ≈ 27.0
        # Verified: 27*0.5 + 4*2.36 + 0 + 0.5 = 23.4 GB ≈ 23 ✓
        #           27*0.5 + 4*4.19 + 0 + 0.5 = 30.76 GB ≈ 30 ✓
        ram_baseline_gb=27.0,
        ram_slope_gb_per_mp=4.0,
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

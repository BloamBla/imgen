"""v0.8.0 — Model dataclass replacing the v0.7 Backend.

Per [[project-v080-design]] §F. A `Model` row is the per-recipe property
surface: which engine routes it, what its default steps/guidance are,
what RAM it needs, and so on. The Engine (mflux / diffusers_mps) reads
the Model + GenParams and dispatches.

Field surface is locked by ``tests/test_models.py::TestModelDataclassShape``.
Engine-conditional invariants are enforced at every instantiation site
(BUILTIN_MODELS dict, user TOMLs, test fixtures) via ``__post_init__``,
so the registry source-of-truth flip in commit 4b can't introduce a
silently-malformed row.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Model"]


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

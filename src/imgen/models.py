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

__all__ = ["BUILTIN_MODELS", "Model", "_model_from_backend"]


# v0.8 commit-2 scaffold: RAM defaults per legacy v0.7 backend name.
# Exists ONLY to satisfy Model.__post_init__ during the derive-from-
# Backend phase (commits 2-3). Commit 4b promotes BUILTIN_MODELS to
# the live registry with literal per-model values declared inline (per
# §G.1), and this table goes away. Values are calibrated from the
# v0.7.14 RAM_REQUIRED_GB table + v0.7.7 real-mflux measurements +
# 2026-05-25 Qwen-2512 swap-thrash session, encoded per [[project-v080-
# design]] §G.1.
_V07_BACKEND_RAM_DEFAULTS: dict[str, tuple[float, float, float]] = {
    # name → (ram_baseline_gb, ram_slope_gb_per_mp, encoder_ram_gb)
    "flux":                 (9.0,  5.0,  0.0),
    "flux-dev":             (9.0,  5.0,  0.0),
    "qwen":                 (10.0, 5.0,  7.0),  # Qwen2.5-VL ~7 GB
    "flux2-klein-edit-9b":  (14.0, 5.5,  0.0),
}


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


def _model_from_backend(backend, name: str) -> Model:
    """Derive a v0.8 Model from a v0.7 Backend row.

    Used to populate ``BUILTIN_MODELS`` while the live registry is still
    ``BUILTIN_BACKENDS`` (commits 2-3 per §Q). Once commit 4b flips the
    source-of-truth, BUILTIN_MODELS is declared inline per §G.1 and this
    helper is no longer needed at module load (it stays exported for
    user TOML migration tests).

    The ``name`` parameter is needed for the per-backend RAM lookup —
    Backend rows don't carry RAM info (it lived in v0.7.14's separate
    ``checks.RAM_REQUIRED_GB`` table). The lookup is exhaustive for the
    4 current built-ins; unknown names raise (forces explicit table
    update if a new built-in lands during the v0.8 arc).
    """
    if name not in _V07_BACKEND_RAM_DEFAULTS:
        raise KeyError(
            f"_V07_BACKEND_RAM_DEFAULTS missing entry for {name!r}. "
            "Add (baseline_gb, slope_per_mp, encoder_gb) row in models.py "
            "before deriving a Model for this backend."
        )
    baseline, slope, encoder = _V07_BACKEND_RAM_DEFAULTS[name]
    return Model(
        engine="mflux",
        binary=backend.binary,
        extra_args=backend.extra_args,
        image_flag=backend.image_flag,
        supports_strength=backend.supports_strength,
        supports_negative=backend.supports_negative,
        needs_token=backend.needs_token,
        lora_compat_group=backend.lora_compat_group,
        hf_gated_repo=backend.hf_gated_repo,
        enhance_system_prompt=backend.enhance_system_prompt,
        enhance_invariants=backend.enhance_invariants,
        # v0.8 NEW fields — populated from the per-backend lookup
        # table above. Commit 4b replaces this with inline declarations
        # per §G.1 and the table is deleted.
        ram_baseline_gb=baseline,
        ram_slope_gb_per_mp=slope,
        encoder_ram_gb=encoder,
        # default_steps / default_guidance / min_guidance / max_guidance
        # stay at dataclass defaults for the derived-view phase —
        # explicit per-model values land alongside the §G.1 inline
        # declarations in commit 4b.
    )


# Derived registry view. Built from BUILTIN_BACKENDS at module load —
# single source of derivation, no manual sync between the two dicts.
# When commit 4b flips source-of-truth, this becomes the live literal
# declaration and BUILTIN_BACKENDS becomes the derived view going
# backward (`{name: _backend_from_model(m) for ...}`).
def _build_builtin_models() -> dict[str, Model]:
    """Local import to avoid module-load circularity — backends.py
    imports `Model` from us in its facade (post-§D), but at this commit
    backends.py only exports legacy `Backend` + `BUILTIN_BACKENDS`."""
    from .backends import BUILTIN_BACKENDS
    return {
        name: _model_from_backend(b, name)
        for name, b in BUILTIN_BACKENDS.items()
    }


BUILTIN_MODELS: dict[str, Model] = _build_builtin_models()

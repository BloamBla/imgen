"""v0.8.0 — Engine Protocol + GenParams.

Per [[project-v080-design]] §C. ``Engine`` is a structural-typing
Protocol (NOT abc.ABC) — only 2 subclasses will ever exist in v0.8.0,
runtime ``@abstractmethod`` enforcement is overkill against a closed
set, and Protocol gives cleaner test doubles. Mirrors the
[[project-v07-backlog]] FL-1 precedent that picked Protocol for
``IterationGroup``.

``@runtime_checkable`` so ``isinstance(eng, Engine)`` works in lock-in
tests (structural conformance check at runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

__all__ = ["Engine", "GenParams"]


@dataclass(frozen=True, slots=True)
class GenParams:
    """Resolved parameters for one generation call. Pure data, no I/O.

    Engines take a GenParams + Model and produce either subprocess argv
    (mflux) or in-process pipeline invocation (diffusers_mps).
    """
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    quantize: int
    strength: float
    input_path: Path | None
    output_path: Path
    # tuple[LoraRef, ...] — typed as plain tuple to avoid circular import
    # with styles.py at this layer. Engine implementations re-type at the
    # call boundary.
    loras: tuple
    # — mflux-engine-specific (default to v0.7 DEFAULTS values so
    # diffusers_mps callers can omit them safely). These are per-call
    # values today (user can override via --mlx-cache-limit-gb /
    # --battery-stop-limit) but defaults match merged_defaults from
    # config.py to keep MfluxEngine argv bit-identical with v0.7.17. —
    mlx_cache_gb: int = 12
    battery_stop: int = 20


@runtime_checkable
class Engine(Protocol):
    """Dispatch contract. Each Engine implementation owns ONE runtime
    (subprocess vs in-process). Multiple Models route to one Engine.
    """

    name: str

    def build_cmd(self, model, params: GenParams) -> list[str]:
        """For subprocess-based Engines (mflux), return argv list.

        For in-process Engines (diffusers_mps), raise NotImplementedError
        — those engines override ``run()`` directly with their own
        dispatch mechanism (stdin-JSON to a static runner script per §E).
        """
        ...

    def run(
        self,
        model,
        params: GenParams,
        env: Mapping[str, str] | None = None,
    ) -> int:
        """Execute generation. Returns exit code (0 success, non-zero
        failure).

        For subprocess Engines: routes through
        ``subprocess_helpers.run_with_stderr_redaction`` (or its v0.8.0
        sibling) to preserve HF-token redaction + chunk-streamed
        logging. The redaction wrapper is mandatory for both engines —
        diffusers' ``from_pretrained`` 401/403 tracebacks include auth
        headers that must be filtered on the way out (see §E.1 security
        round-2 fix).
        """
        ...

    def validate(self, model, params: GenParams) -> list[str]:
        """Return list of error messages for invalid (model, params)
        combinations. Empty list = OK to proceed.

        Example rejections:
          - z-image-turbo + guidance != 0 → ['turbo requires guidance=0']
          - flux2-klein-edit + negative_prompt → ['FLUX.2 family does not
            support negative']

        Lives on Engine (not Model) because validation rules are facts
        about the RUNTIME accepting that combination, not facts about
        the weights.
        """
        ...

    def ram_estimate_gb(self, model, params: GenParams) -> float:
        """Peak RAM estimate for the preflight gate.

        Reads ``model.ram_baseline_gb`` + ``model.ram_slope_gb_per_mp`` +
        ``model.encoder_ram_gb`` + adds Engine-specific overhead
        constants (mflux subprocess startup vs diffusers_mps import
        cost). Replaces v0.7.14's uniform 5 GB/MP slope table.
        """
        ...

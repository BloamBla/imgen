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
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Mapping,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Forward reference: TrainingParams lives in engines/_training.py
    # which imports from models.py. Importing it at type-check time
    # keeps base.py's runtime import surface unchanged (no new
    # transitive imports for callers that don't touch training).
    from ._training import TrainingParams

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
    # — v0.9 commit 2 — video output extensions [[project-v090-design]] §D.
    # APPENDED AT END so positional construction is byte-additive for
    # v0.8 image callers (num_frames default 1 ⇒ image; fps default 24
    # is ignored when num_frames == 1). Reading these requires keyword
    # access; v0.8 positional argv stops at battery_stop. —
    num_frames: int = 1
    fps: int = 24


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
        *,
        env: Mapping[str, str] | None = None,
        log_file: BinaryIO | None = None,
    ) -> int:
        """Execute generation. Returns exit code (0 success, non-zero
        failure).

        For subprocess Engines: routes through
        ``subprocess_helpers.run_with_stderr_redaction`` to preserve
        HF-token redaction + chunk-streamed logging. The redaction
        wrapper is mandatory for both engines — diffusers'
        ``from_pretrained`` 401/403 tracebacks include auth headers
        that must be filtered on the way out (see §E.1 security
        round-2 fix).

        ``log_file`` (v0.8.2 architect CRITICAL-1 closure): optional
        BatchLogger-borrowed fd. When non-None the redacted stderr
        tee writes into this fd alongside the in-memory copy, so
        multi-style runs preserve their ``~/.imgen/logs/<batch_id>.log``
        layout. Engine implementations pass it through to
        ``run_with_stderr_redaction(log_file=...)`` unchanged.

        ``env`` and ``log_file`` are keyword-only — positional
        passthrough would silently drift if a future field landed
        between them.

        KeyboardInterrupt must propagate UNWRAPPED. The orchestration
        layer (``engine_dispatch.run_one_iteration``) owns the
        cancel-history-marker side effect; if Engine.run caught and
        swallowed (or re-raised as a different exception), the marker
        couldn't fire.
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

    def train(
        self,
        model,
        params: "TrainingParams",
        env: Mapping[str, str] | None = None,
    ) -> int:
        """Execute LoRA training. Returns exit code (0 success,
        non-zero failure).

        ``env`` is the allowlisted subprocess environment from
        ``build_mflux_env(token=hf_token)`` (security H-4 — never the
        raw inherited environment at the ``Popen`` boundary). Part of
        the Protocol signature so implementations stay uniform with
        ``MfluxEngine.train``; engines that raise ``NotImplementedError``
        accept it without use.

        v0.10.0 commit 1: Protocol verb added per
        [[project-v100-design]] §R.1 round-1 architect H-3 closure —
        re-decided from a separate ``MfluxTrainer`` class to a method
        on the existing Engine Protocol to preserve the v0.9.5 M-2
        Engine-registry as single source of truth for runtime
        dispatch. ``Engine.validate`` and ``Engine.ram_estimate_gb``
        already coexist as parallel verbs on the same Protocol;
        ``train`` continues the pattern.

        v0.10.0 commit 5: ``params`` is :class:`TrainingParams`
        (NOT ``GenParams``). Training has a fundamentally different
        parameter shape from inference (dataset_dir + target_modules
        + optimizer settings vs prompt + width/height + steps), and
        sharing a single envelope would either inflate ``GenParams``
        with training-only fields or leave training callers stuffing
        sentinel values into inference-only fields. Separate
        dataclasses keep both surfaces honest.

        **Convention** (§R.1 round-2 N-1 closure): engines that don't
        support training MUST raise ``NotImplementedError`` with the
        engine name in the message — same posture as
        ``abc.abstractmethod`` conventions. v0.10.0 ships:

        * ``MfluxEngine.train`` — raises ``NotImplementedError`` at
          commits 1-6; commit 7 wires the real
          ``mflux-train --config <FILE>`` subprocess dispatch via
          ``run_with_stderr_redaction`` + ``build_mflux_env``.
        * ``DiffusersMpsEngine.train`` — raises ``NotImplementedError``
          PERMANENTLY (v0.10.0 doesn't train via diffusers_mps; video
          Models stay inference-only per §B.4).
        """
        ...

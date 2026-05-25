"""v0.8.0 — MfluxEngine: subprocess dispatch for mflux-generate-* binaries.

Per [[project-v080-design]] §D. Ports v0.7's ``build_mflux_cmd`` (from
``backends.py``) to the Engine Protocol surface. Argv output is
bit-identical to v0.7.17 for every current built-in backend — locked
by ``tests/test_engines.py::TestMfluxEngineBuildCmdMatchesV07_17``.

Commit 2 scope: ``build_cmd`` only (the argv-construction path).
``run`` / ``validate`` / ``ram_estimate_gb`` are stubbed minimally so
``isinstance(MfluxEngine(), Engine)`` works; their full implementations
land in commits 6 (run via redaction wrapper), 7 (validate), 8 (RAM).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..backends import filter_compatible_loras
from .base import GenParams

__all__ = ["MfluxEngine"]


class MfluxEngine:
    """Wraps the mflux-generate-* CLI binaries.

    Implements the ``Engine`` Protocol from
    ``src/imgen/engines/base.py``. Stateless dispatch — instantiate once
    at module load and reuse across calls. No ``__init__`` because
    there's no state to capture.
    """

    name = "mflux"

    def build_cmd(
        self,
        model,
        params: GenParams,
        *,
        binary: Path | None = None,
    ) -> list[str]:
        """Build the mflux argv for ``model`` from ``params``.

        Pure: no I/O, no env reads, no subprocess. Order preserved from
        v0.7.17 ``build_mflux_cmd``:

          1. binary + ``--quantize N``
          2. ``model.image_flag <input>`` (if ``params.input_path`` is not None)
          3. ``--prompt`` + ``--steps`` + ``--guidance`` + ``--seed``
             + ``--width`` + ``--height`` + ``--mlx-cache-limit-gb``
             + ``--battery-percentage-stop-limit`` + ``--output``
          4. ``--image-strength X`` (if ``model.supports_strength``)
          5. ``model.extra_args`` (e.g. ``('--model', 'dev')``)
          6. ``--negative-prompt X`` (if ``model.supports_negative`` and
             ``params.negative``)
          7. LoRA paths + scales (compatible-only)

        ``binary`` kwarg is the resolved mflux entry-point path. If
        None, looks up ``VENV_BIN / model.binary`` — matches v0.7's
        cmd_helpers behaviour. Tests pass an explicit path so they
        don't depend on the real venv layout.
        """
        if binary is None:
            from ..paths import VENV_BIN
            binary = VENV_BIN / model.binary

        cmd = [
            str(binary),
            "--quantize", str(params.quantize),
        ]
        if params.input_path is not None:
            cmd += [model.image_flag, str(params.input_path)]
        cmd += [
            "--prompt", params.prompt,
            "--steps", str(params.steps),
            "--guidance", str(params.guidance),
            "--seed", str(params.seed),
            "--width", str(params.width),
            "--height", str(params.height),
            "--mlx-cache-limit-gb", str(params.mlx_cache_gb),
            "--battery-percentage-stop-limit", str(params.battery_stop),
            "--output", str(params.output_path),
        ]
        if model.supports_strength:
            cmd += ["--image-strength", str(params.strength)]
        cmd += list(model.extra_args)
        if model.supports_negative and params.negative:
            cmd += ["--negative-prompt", params.negative]
        if params.loras:
            compatible, _incompatible = filter_compatible_loras(
                params.loras, model,
            )
            if compatible:
                cmd += ["--lora-paths", *(lora.ref for lora in compatible)]
                cmd += ["--lora-scales", *(str(lora.weight) for lora in compatible)]
        return cmd

    def run(
        self,
        model,
        params: GenParams,
        env: Mapping[str, str] | None = None,
    ) -> int:
        """Stub for commit 2. Full implementation in commit 6 routes
        through ``run_with_stderr_redaction``. Today's call sites
        (cmd_generate / cmd_batch / cmd_draw / cmd_refine) continue
        to invoke the legacy paths until the CLI rename in commit 4."""
        raise NotImplementedError(
            "MfluxEngine.run wired in commit 6 — call sites use legacy "
            "subprocess paths through commit 5."
        )

    def validate(self, model, params: GenParams) -> list[str]:
        """Stub for commit 2 — returns empty list (no validation).
        Real implementation in commit 7 enforces ``supported_quants``
        / ``min_guidance`` / ``max_guidance`` / ``supports_negative``
        from the Model fields."""
        return []

    def ram_estimate_gb(self, model, params: GenParams) -> float:
        """Stub for commit 2 — minimal formula using Model fields so
        the value is non-zero (commit 8 wires this into the doctor
        RAM table)."""
        mp = params.width * params.height / 1_000_000
        return (
            model.ram_baseline_gb
            + model.ram_slope_gb_per_mp * mp
            + model.encoder_ram_gb
        )

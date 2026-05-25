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

        # v0.8.0 commit 7 (§M): skip --quantize when the Model is a
        # prequantized recipe (e.g. mlx-community/*-4bit). Built-ins
        # at commit 7 ship with omit_quantize=False; the field is
        # forward-compat for user TOMLs declaring prequant repos.
        cmd = [str(binary)]
        if not model.omit_quantize:
            cmd += ["--quantize", str(params.quantize)]
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
        *,
        env: Mapping[str, str] | None = None,
        log_file = None,  # BinaryIO | None — BatchLogger-borrowed fd
    ) -> int:
        """Execute an mflux generation subprocess. v0.8.2 M-1B closure
        of the v0.8.1 M-1 backlog item.

        Argv comes from ``self.build_cmd(model, params)`` — same
        argv-shape locked to v0.7.17 by ``test_engines.py::
        TestMfluxEngineBuildCmdMatchesV07_17`` AND now also byte-
        identical with the legacy ``backends.build_mflux_cmd``
        (architect CRITICAL-2 lock-in:
        ``test_mflux_engine_build_cmd_matches_legacy_build_mflux_cmd``).

        The argv is dispatched through
        ``subprocess_helpers.run_with_stderr_redaction`` which:
          * Tees stderr to the parent process's stderr + the optional
            ``log_file`` BinaryIO (BatchLogger-borrowed fd).
          * Applies HF-token redaction on the stream so a 401 traceback
            doesn't leak ``hf_<token>`` into either destination.
          * Streams chunk-by-chunk so 5+ minute mflux runs surface
            progress instead of buffering.

        ``env`` is passed through verbatim — the caller
        (``cmd_helpers.run_one_iteration``) builds it via
        ``build_mflux_env`` to populate the allowlisted env vars
        (HF_TOKEN, PYTHONPATH, etc.).

        KeyboardInterrupt propagates unwrapped (architect HIGH-2):
        ``run_with_stderr_redaction`` already re-raises on its own
        catch; this method doesn't swallow either. The cancel-history-
        marker side effect lives in the orchestrator
        (``run_one_iteration``).
        """
        from ..subprocess_helpers import (
            build_mflux_env,
            run_with_stderr_redaction,
        )

        cmd = self.build_cmd(model, params)
        # env=None defensive fallback: build a minimal allowlisted env
        # (no HF_TOKEN — gated-repo runs require the caller to pass
        # env=build_mflux_env(token=...) explicitly). Production path
        # via cmd_helpers.run_one_iteration always passes ctx.env;
        # this branch only fires for tests / direct Engine.run calls.
        return run_with_stderr_redaction(
            cmd,
            dict(env) if env is not None else build_mflux_env(),
            log_file=log_file,
        )

    def validate(self, model, params: GenParams) -> list[str]:
        """Return list of error messages for (Model, GenParams)
        combinations that mflux would reject at argv-parse time.

        v0.8.0 commit 7 (§M): real implementation. Replaces the
        pre-commit-7 hardcoded special-cases scattered across cmd_*
        (e.g. refine.py:238 `if backend == "flux2-klein-edit-9b":
        args.guidance = 1.0`) with a centralised per-Model contract
        that scales to any future backend without per-binary cmd_*
        edits.

        Checks:

        * ``params.quantize ∈ model.supported_quants`` — built-ins
          ship the full set (3,4,5,6,8); user TOMLs that restrict
          quants get enforced here. Model rows with
          ``supported_quants=()`` (engines that don't quantize at
          all) skip this check.
        * ``model.min_guidance ≤ params.guidance ≤ model.max_guidance``
          — flux2-klein-edit-9b pins min=max=1.0 (mflux 0.17.5 rejects
          anything else at argv); FLUX.1-Kontext / FLUX.1-dev hard-
          floor at 1.0 (CFG=0 produces blurry/uninstructable output
          on non-distilled FLUX).

        Returns empty list when params pass — caller proceeds. Non-
        empty list → caller dies with each error on its own line
        (clean exit-2 path via the cmd_helpers validate-or-die
        helper).
        """
        errors: list[str] = []
        if model.supported_quants and params.quantize not in model.supported_quants:
            allowed = sorted(model.supported_quants)
            errors.append(
                f"--quantize {params.quantize} not supported by "
                f"{model.binary}; allowed: {allowed}"
            )
        if not (
            model.min_guidance <= params.guidance <= model.max_guidance
        ):
            errors.append(
                f"--guidance {params.guidance} out of range "
                f"[{model.min_guidance}, {model.max_guidance}] "
                f"for {model.binary}"
            )
        return errors

    def ram_estimate_gb(self, model, params: GenParams) -> float:
        """v0.8.0 commit 8 (§L): peak RAM estimate (GB) for the
        (Model, GenParams) combination. Replaces v0.7.14's per-
        (backend, quant) ``RAM_REQUIRED_GB`` lookup table with per-
        Model math.

        Formula (per memo §L):

          weights_gb     = baseline * (quantize / 8)   — rough Q-scale
          activations_gb = slope * mp
          encoder_gb     = one-time peak from VLM encoder load
          overhead_gb    = 0.5  — mflux subprocess + MLX cache headroom
          total          = weights + activations + encoder + overhead

        ``quantize / 8.0`` is the rough weight-memory scaling. Q8 →
        full baseline; Q4 → half; Q3 → 3/8 (slightly underestimates
        because int3 unpacking overhead is non-linear, but for
        preflight estimation purposes the linear approximation is
        within the noise of real measurements).

        Calibration anchors (locked by tests):
          * flux-kontext Q8 1MP ≈ 18 GB → matches v0.7.7 real
            measurement on M2 Pro 32 GB.
          * flux2-klein-edit-9b Q4 1536² ≈ 23 GB and Q4 2048² ≈ 30 GB
            → both match v0.7.7 real-mflux smoke run.
        """
        mp = params.width * params.height / 1_000_000.0
        weights_gb = model.ram_baseline_gb * (params.quantize / 8.0)
        activations_gb = model.ram_slope_gb_per_mp * mp
        encoder_gb = model.encoder_ram_gb
        overhead_gb = 0.5
        return weights_gb + activations_gb + encoder_gb + overhead_gb

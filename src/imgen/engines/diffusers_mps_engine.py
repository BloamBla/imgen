"""v0.8.0 commit 6 — DiffusersMpsEngine: out-of-process diffusers dispatch.

Per [[project-v080-design]] §E.1. Wraps a SEPARATE Python venv
(``.venv-diffusers/``) so the heavy diffusers + torch stack stays out
of the main mflux venv. Invokes the static runner module in
:mod:`imgen.engines._diffusers_runner` via:

    .venv-diffusers/bin/python -m imgen.engines._diffusers_runner

GenParams + Model fields cross the process boundary as JSON-on-stdin.
The argv is STATIC — no user data ever flows into the command line.
This is the locked design pattern that round-1 review's CRITICAL
findings (script-injection, path traversal, HF-token leak) all
hardened against.

At commit 6 ZERO built-in Models route to this engine. The
``qwen-image-2512-bf16.toml`` opt-in template lands at commit 10 per
§Q (M3-Ultra-and-up only; the 32 GB Mac stays on mflux MLX-Q4 per
[[project-qwen-2512-findings-2026-05-25]]). The engine ships in
commit 6 as foundation for future HF day-0 models that mflux hasn't
ported.
"""
from __future__ import annotations

import json
from typing import Mapping

from .base import GenParams

__all__ = ["DiffusersMpsEngine"]


class DiffusersMpsEngine:
    """Out-of-process diffusers dispatch via a separate-venv subprocess.

    Implements the Engine Protocol from
    :mod:`imgen.engines.base`. Stateless dispatch — instantiate once
    at module load and reuse across calls. No ``__init__`` because
    there's no state to capture.
    """

    name = "diffusers_mps"

    def build_cmd(self, model, params: GenParams) -> list[str]:
        """Not applicable: diffusers_mps dispatches via JSON-on-stdin
        to a static runner module, not an argv-shaped command line.

        Engine Protocol contract per
        [[project-v080-design]] §C: subprocess-based engines return
        argv from ``build_cmd``; in-process engines (this one, where
        all user data goes via stdin) raise NotImplementedError and
        override ``run`` directly.
        """
        raise NotImplementedError(
            "diffusers_mps dispatches via in-process call to "
            "_diffusers_runner subprocess; no argv-shaped command. "
            "Engine routes through `run()` instead."
        )

    def run(
        self,
        model,
        params: GenParams,
        *,
        env: Mapping[str, str] | None = None,
        log_file = None,  # BinaryIO | None — BatchLogger-borrowed fd
    ) -> int:
        """Spawn ``.venv-diffusers/bin/python -m
        imgen.engines._diffusers_runner`` with JSON payload on stdin.

        Path resolution rules (security pre-vet C1 + memo §E.1):

        * ``.venv-diffusers/`` is resolved from
          :data:`imgen.paths.IMGEN_INSTALL_ROOT` — NEVER from cwd. So
          ``cd /tmp/attacker && imgen draw ...`` cannot exec a
          planted python.
        * ``venv_python.is_file()`` (not ``exists()``) before exec —
          mirror of v0.4 ``_validate_binary_field`` discipline (a
          directory at the path would otherwise crash ``subprocess``
          with an opaque ``IsADirectoryError``).
        * Friendly error if the venv is missing — points the user at
          bootstrap.sh's diffusers opt-in prompt, not a bare
          ``FileNotFoundError``.

        Payload construction:

        * Every value comes from ``model`` (registry-controlled) or
          ``params`` (resolver-validated at the parser layer). The
          runner re-validates at the trust boundary (defense in
          depth: parent is trusted, JSON shape is an explicit
          contract).
        * ``model.param_overrides`` is a ``tuple[tuple[str, object],
          ...]`` at the Model dataclass level (frozen+slot); the
          runner expects a dict. Convert at this boundary.

        Subprocess plumbing:

        * ``run_with_stderr_redaction(stdin_data=...)`` (extended at
          commit 6 with keyword-only stdin_data) writes the JSON
          payload to the child's stdin BEFORE entering the stderr-
          read loop. The HF-token redaction pattern catches
          ``hf_<token>`` leaks in diffusers' 401/403 tracebacks
          (auth headers in HTTP error dumps).
        * Environment via :func:`build_diffusers_env` (sibling of
          build_enhance_env; allowlist with HF_TOKEN forwarded for
          gated-repo support).
        """
        from ..colors import die
        from ..paths import IMGEN_INSTALL_ROOT
        from ..subprocess_helpers import (
            build_diffusers_env,
            run_with_stderr_redaction,
        )

        venv_python = (
            IMGEN_INSTALL_ROOT / ".venv-diffusers" / "bin" / "python"
        )
        if not venv_python.is_file():
            die(
                "diffusers_mps engine selected but the diffusers venv "
                "is not installed.\n"
                f"  Expected: {venv_python}\n"
                "Run bootstrap.sh and answer 'y' at the diffusers "
                "prompt, OR set IMGEN_INSTALL_DIFFUSERS=1 to install "
                "non-interactively. Alternatively, comment out the "
                "model's TOML to fall back to an mflux model.",
                code=3,
            )

        payload = {
            "repo": model.repo,
            "prompt": params.prompt,
            "negative": params.negative,
            "steps": params.steps,
            "guidance": params.guidance,
            "width": params.width,
            "height": params.height,
            "seed": params.seed,
            "output_path": str(params.output_path),
            "input_path": (
                str(params.input_path) if params.input_path else None
            ),
            "cpu_offload_threshold_mp": model.cpu_offload_threshold_mp,
            # tuple-of-tuples → dict at the boundary; runner re-
            # filters through the allowlist.
            "param_overrides": dict(model.param_overrides),
        }

        # STATIC argv — no user data interpolated. Security pre-vet
        # confirmed: str(Path) is a stable stringification; no shell
        # expansion possible because subprocess uses shell=False.
        argv = [
            str(venv_python),
            "-m",
            "imgen.engines._diffusers_runner",
        ]

        return run_with_stderr_redaction(
            argv,
            dict(env) if env is not None else build_diffusers_env(),
            stdin_data=json.dumps(payload).encode("utf-8"),
            log_file=log_file,
        )

    def validate(self, model, params: GenParams) -> list[str]:
        """Intentionally stubbed by design — payload validation lives
        at the subprocess trust boundary in
        ``_diffusers_runner._validate_payload_shape`` per memo §E.1.

        v0.8.1 §R.4 M-4 / architect docstring-drift closure: pre-v0.8.1
        comment promised wiring "in commit 7" — that wiring never
        landed across commits 7-11, and on reflection it shouldn't.
        The runner re-validates EVERY field at the JSON-stdin boundary
        with deny-by-default discipline; mirroring that here would
        double-validate without catching anything new and risk drift
        between the two checks. The Engine.validate surface stays
        intentionally empty for diffusers_mps; MfluxEngine.validate
        (which IS wired) catches per-Model invariants at the in-
        process resolver layer where the runner has no presence.

        Returns ``[]`` unconditionally — callers treat as "no errors".
        """
        return []

    def ram_estimate_gb(self, model, params: GenParams) -> float:
        """v0.8.0 commit 8 (§L): peak RAM estimate (GB) for the
        diffusers_mps engine. Same shape as
        :meth:`MfluxEngine.ram_estimate_gb` but with a heavier
        overhead constant — diffusers + torch + transformers cold
        import is ~1.5-2 GB vs mflux's ~0.5 GB.

        Formula:
          total = baseline * (quantize / 8) + slope * mp + encoder + 2.0
        """
        mp = params.width * params.height / 1_000_000.0
        weights_gb = model.ram_baseline_gb * (params.quantize / 8.0)
        activations_gb = model.ram_slope_gb_per_mp * mp
        encoder_gb = model.encoder_ram_gb
        overhead_gb = 2.0  # diffusers + torch import footprint
        return weights_gb + activations_gb + encoder_gb + overhead_gb

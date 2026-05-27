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
import os
from typing import TYPE_CHECKING, Mapping

from .base import GenParams

if TYPE_CHECKING:
    from ._training import TrainingParams

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
        # v0.9 commit 7.1 (§R.2 security HIGH-1) + commit 11.2 hotfix:
        # symlink guard on execution path. The original FIX-1 rejected
        # ALL symlinks, which broke the canonical Python venv layout
        # (`python -> python3.12` relative same-dir symlink — standard
        # `python3 -m venv` output). Hotfix narrows the guard:
        #
        # * Allow first-level symlinks whose target is a same-dir peer
        #   (no `/` in the readlink). Catches `python -> python3.12`.
        # * Reject any symlink with `/` in target — absolute paths,
        #   `..` traversal, or any non-peer reference. Catches the
        #   attacker scenario (`python -> /tmp/evil_python`).
        #
        # The python3.12 chain may end at /opt/homebrew/... or
        # /Library/Frameworks/Python... — a same-uid attacker with
        # write access to those system paths is out of scope (they
        # could replace the binary directly anyway). The first-level
        # peer check defends against the realistic threat of a
        # planted-symlink-into-venv attack without rejecting normal
        # venv layouts.
        # v0.9.1 B-14: wrap readlink() in try/except so a TOCTOU race
        # (symlink vanishing between is_symlink() and readlink()) falls
        # through to the is_file() check below rather than propagating
        # an unhandled OSError to the user.
        guard_triggers = False
        if venv_python.is_symlink():
            try:
                target = os.readlink(venv_python)
                guard_triggers = "/" in target
            except OSError:
                pass
        if guard_triggers:
            die(
                f".venv-diffusers/bin/python is a symlink with "
                "non-peer target — refusing to exec. Canonical "
                "Python venv layout uses relative same-dir symlinks "
                "(e.g. python -> python3.12); an absolute target "
                "or path traversal is a plant-attack signal. "
                "Remove .venv-diffusers/ and re-run bootstrap.sh.",
                code=3,
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
        # v0.9 commit 4 — video payload extensions (§F). Image Models
        # keep the v0.8 payload shape (no output_type key, runner
        # defaults to image). Video Models add the v0.9 triple +
        # force_cpu_offload from VideoConfig.
        if model.video is not None:
            payload["output_type"] = "video"
            payload["num_frames"] = params.num_frames
            payload["fps"] = params.fps
            payload["force_cpu_offload"] = model.video.force_cpu_offload
            # v0.9.3 C2 (B-1 closure): read from VideoConfig typed
            # field instead of hardcoding. v0.9.0 t2v rows keep working
            # via the default "LTXPipeline"; v0.9.3 i2v constructs a
            # derivative VideoConfig with
            # ``pipeline_class="LTXImageToVideoPipeline"``. The runner
            # re-validates against its own allowlist before the
            # diffusers import (security §R.1 HIGH-1).
            payload["pipeline_class"] = model.video.pipeline_class

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
        """v0.9 commit 3 ([[project-v090-design]] §E.0 + §E.1): video
        Models get parent-side validation that mirrors what the
        runner-side ``_validate_payload_shape`` will re-check at the
        trust boundary (commit 4). Image Models keep the v0.8.1 §R.4
        M-4 no-op pattern — image payload checks live solely at the
        runner boundary.

        Reopening for video is justified because video inputs have a
        ~50ms parser-gate cost vs ~3-5s cold-import cost on the runner
        side — a 60× speedup at the rejection path makes
        the double-validation tradeoff worthwhile here.
        """
        if model.video is None:
            # Image path — preserve v0.8.1 §R.4 M-4 no-op behaviour.
            return []
        return self._validate_video(model, params)

    def _validate_video(self, model, params: GenParams) -> list[str]:
        """§E.1 — num_frames range, alignment (8k+1 for LTX), fps
        allowlist. Error messages include nearest-valid suggestion
        for alignment violations per architect §R.1 MED-2.
        """
        errors: list[str] = []
        vc = model.video
        nf = params.num_frames

        # Range
        min_frames = vc.default_num_frames // 2
        if nf < min_frames:
            errors.append(
                f"num_frames={nf} below model minimum {min_frames} "
                f"(default_num_frames={vc.default_num_frames})"
            )
        if nf > vc.max_num_frames:
            errors.append(
                f"num_frames={nf} exceeds model cap {vc.max_num_frames}"
            )

        # Alignment: (nf - offset) % alignment == 0
        if vc.num_frames_alignment > 1:
            remainder = (nf - vc.num_frames_offset) % vc.num_frames_alignment
            if remainder != 0:
                # Floor to nearest valid <= nf so the suggestion never
                # increases the user-requested duration.
                k = (nf - vc.num_frames_offset) // vc.num_frames_alignment
                nearest = k * vc.num_frames_alignment + vc.num_frames_offset
                if nearest < vc.num_frames_offset:
                    nearest = vc.num_frames_offset
                errors.append(
                    f"num_frames={nf} violates alignment "
                    f"{vc.num_frames_alignment}k+{vc.num_frames_offset}; "
                    f"nearest valid: {nearest}"
                )

        # FPS allowlist — v0.9.0 ships {24, 25, 30}.
        if params.fps not in {24, 25, 30}:
            errors.append(
                f"fps={params.fps} not in {{24, 25, 30}}"
            )

        # v0.9 commit 7.1 (§R.2 architect HIGH-2): quantize gate.
        # Video Models with supported_quants=() (LTX-Video at v0.9.0
        # is bf16-only) must reject non-zero quantize. Without this
        # gate, merged_defaults["quantize"]=4 silently propagates
        # through the resolver to the GenParams of every LTX
        # iteration. Functionally inert (runner ignores quantize for
        # video) but semantically wrong — future user-TOML video
        # Model could misconfigure engine=diffusers_mps without
        # video= and inherit silent quantize behaviour.
        if not model.supported_quants and params.quantize > 0:
            errors.append(
                f"quantize={params.quantize} but model.supported_quants "
                "is empty (model is bf16-only; pass --quantize 0 or "
                "omit the flag)"
            )

        return errors

    def ram_estimate_gb(self, model, params: GenParams) -> float:
        """v0.8.0 commit 8 (§L) image branch + v0.9 commit 3 (§L)
        video branch. Image Models keep the pre-v0.9 formula; video
        Models use a different shape because (a) LTX is bf16-only —
        quantize doesn't scale weights; (b) baseline ALREADY includes
        diffusers cold-import overhead per §L "T5-offloaded baseline"
        definition (no separate +2.0 term); (c) encoder source is
        ``model.video.encoder_ram_gb`` (transient T5-XXL peak), not
        ``model.encoder_ram_gb`` (image-only field).
        """
        if model.video is None:
            return self._ram_estimate_image(model, params)
        return self._ram_estimate_video(model, params)

    def _ram_estimate_image(self, model, params: GenParams) -> float:
        """Image branch — preserved pre-v0.9 formula:
          total = baseline * (quantize / 8) + slope * mp + encoder + 2.0
        """
        mp = params.width * params.height / 1_000_000.0
        weights_gb = model.ram_baseline_gb * (params.quantize / 8.0)
        activations_gb = model.ram_slope_gb_per_mp * mp
        encoder_gb = model.encoder_ram_gb
        overhead_gb = 2.0  # diffusers + torch import footprint
        return weights_gb + activations_gb + encoder_gb + overhead_gb

    def _ram_estimate_video(self, model, params: GenParams) -> float:
        """Video branch (§L):
          baseline + slope*mp + video.encoder_ram_gb + 0.1*num_frames

        No quantize term (LTX bf16-only); no +2.0 overhead (baseline
        already includes cold-import footprint per §L definition).
        """
        mp = (params.width * params.height) / 1_000_000.0
        baseline = model.ram_baseline_gb
        mp_term = model.ram_slope_gb_per_mp * mp
        encoder = model.video.encoder_ram_gb  # transient T5-XXL peak
        frame_term = 0.1 * params.num_frames
        return baseline + mp_term + encoder + frame_term

    def train(self, model, params: "TrainingParams") -> int:
        """v0.10.0 — diffusers_mps does NOT support LoRA training.

        v0.10.0 ships mflux-train-based training only;
        ``DiffusersMpsEngine.train`` raises ``NotImplementedError``
        PERMANENTLY per [[project-v100-design]] §B.4. Video Models
        (ltx-video and future LTX variants) stay inference-only.

        Per the Engine.train docstring convention (§R.1 round-2 N-1):
        engines that don't support training raise NotImplementedError
        with the engine name in the message.

        Lifting training onto diffusers_mps would require:
          * A separate training-side runner subprocess in
            ``.venv-diffusers/`` (training peak RAM is incompatible
            with the inference runner's mid-process state)
          * A DiffusionPipeline-shaped training adapter (no equivalent
            of mflux-train's targeted-layer LoRA injection)
          * Per-Model target-module specs covering the diffusers
            transformer attribute paths (NOT mflux's `transformer_blocks`
            naming)
        — a v0.11+ design call, NOT a v0.10.x extension.
        """
        raise NotImplementedError(
            "DiffusersMpsEngine.train: diffusers_mps does not support "
            "LoRA training. v0.10.0 ships training via the mflux engine "
            "only; video/diffusers Models stay inference-only. "
            "See [[project-v100-design]] §B.4."
        )

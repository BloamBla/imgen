"""v0.8.0 commit 6 — STATIC runner module for the diffusers_mps engine.

Per [[project-v080-design]] §E.1. The diffusers_mps Engine runs in a
SEPARATE Python venv (``.venv-diffusers/``) because the diffusers +
torch stack is too heavy to install in the main mflux venv. This
module is the static entry point invoked via:

    .venv-diffusers/bin/python -m imgen.engines._diffusers_runner

GenParams + Model fields cross the process boundary as **JSON on
stdin** — no ``-c "<string>"``, no ``.format()`` of user data into
code, no dynamic Python source. This is the locked design pattern
that round-1 review's CRITICAL findings (script-injection via prompt
concatenation, path traversal via cwd, HF-token leak via stderr) all
hardened against.

Security boundaries (locked by tests in
``tests/test_diffusers_mps_engine.py``):

1. **Bounded stdin read** — refuse oversized payloads BEFORE
   ``json.loads`` to prevent memory-inflation DoS.
2. **Strict payload-shape validation** — explicit required + optional
   key allowlist with type/range/regex checks. Unknown top-level keys
   reject with EX_USAGE (deny-by-default per security pre-vet M3).
3. **HF repo-id regex** — ``[A-Za-z0-9._-]+/[A-Za-z0-9._-]+`` — reject
   path-like values (``../etc``, absolute paths) before they reach
   ``from_pretrained`` (security pre-vet HIGH).
4. **Output/input path checks** — absolute path + safe image extension
   (security pre-vet HIGH defense-in-depth at the trust boundary).
5. **param_overrides allowlist** — only two diffusers kwargs may flow
   from user TOML into ``pipe(**kwargs)``: ``true_cfg_scale`` and
   ``cfg_normalization``. New entries require reviewer approval.
6. **PYTORCH_ENABLE_MPS_FALLBACK=1 set BEFORE torch import** — so the
   colleague's shell env can't sabotage diffusers_mps with strict-MPS
   mode (architect pre-vet M4).
"""
from __future__ import annotations

import json
import os
import re
import sys

__all__ = ["main"]


# ── Tunables (security-pre-vet locked) ─────────────────────────────────

# Bounded stdin read — 4× the realistic max payload (~15 KB prompt +
# config). Memo §E.1 round-2 security HIGH lock-in.
_RUNNER_STDIN_MAX_BYTES = 65_536

# HF repo slug — `author/name`. Each segment MUST start with an
# alphanumeric character; `.`, `_`, `-` allowed in the body. Reject
# path-like values (``../etc/passwd``, ``./local``, ``/abs/path``)
# BEFORE from_pretrained (security commit-6 pre-vet HIGH). The
# start-with-alphanumeric rule blocks the `.` / `..` traversal
# vector — a bare `[A-Za-z0-9._-]+` would accept `.` as a valid
# segment, and `./local` would parse as `.` + `local`.
_HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)

# Safe extensions for the output PNG/JPG/WebP and the optional input
# image. Mirrors paths.SAFE_OUTPUT_EXTS but defined locally so the
# runner has zero non-stdlib imports during validation (defense-in-
# depth: even if imgen.paths fails to import inside .venv-diffusers,
# validation still works).
_SAFE_OUTPUT_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_SAFE_IMAGE_INPUT_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif",
})

# Map _SAFE_OUTPUT_EXTS → PIL format keyword. When PIL.save is given
# an open file object (instead of a path string) it cannot infer the
# format from the extension and needs ``format=...`` explicitly.
_PIL_FORMAT_BY_EXT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}

# param_overrides allowlist — only these keys may flow from user TOML
# into ``pipe(**kwargs)``. Memo §E.1 round-2 security HIGH; symmetric
# to ``backends.py:_DANGEROUS_ENV_VARS`` deny-by-default discipline.
# Reviewer approval required for additions.
_DIFFUSERS_PIPE_KWARG_ALLOWLIST: frozenset[str] = frozenset({
    "true_cfg_scale",        # Qwen-Image-2512 — non-CFG-guidance scale
    "cfg_normalization",     # Z-Image family — CFG normalisation toggle
})

# Top-level payload schema — required + optional keys. Unknown keys
# reject with EX_USAGE (deny-by-default per security pre-vet M3).
_PAYLOAD_REQUIRED_KEYS: frozenset[str] = frozenset({
    "repo",
    "prompt",
    "negative",
    "steps",
    "guidance",
    "width",
    "height",
    "seed",
    "output_path",
})
_PAYLOAD_OPTIONAL_KEYS: frozenset[str] = frozenset({
    "input_path",
    "cpu_offload_threshold_mp",
    "param_overrides",
})

# POSIX EX_USAGE (sysexits.h). Returned for every input-validation
# failure so the caller can grep by code without parsing messages.
_EX_USAGE = 64


# ── Schema validation (pure, no imports beyond stdlib) ─────────────────


def _ex_usage(message: str) -> int:
    """Write a static diagnostic to stderr + return EX_USAGE.

    The MESSAGE parameter is a STATIC string literal at every call
    site — never an f-string of user-controlled values. Security pre-
    vet C2 + memo §I round-1 MEDIUM: error messages from the runner
    must not echo payload contents (control-byte injection vector
    into the caller's terminal via stderr forwarding).
    """
    sys.stderr.write(message + "\n")
    return _EX_USAGE


def _validate_payload_shape(payload: object) -> int:
    """Strict schema check on the JSON-decoded payload. Returns
    EX_USAGE on any violation, 0 if the shape passes.

    Required keys + types + range/regex per
    :data:`_PAYLOAD_REQUIRED_KEYS` and the locked memo §E.1 spec.
    Unknown top-level keys reject (deny-by-default).
    """
    if not isinstance(payload, dict):
        return _ex_usage("runner: payload must be a JSON object")

    keys = set(payload)
    missing = _PAYLOAD_REQUIRED_KEYS - keys
    if missing:
        return _ex_usage(
            "runner: payload missing required keys"
        )
    unknown = keys - _PAYLOAD_REQUIRED_KEYS - _PAYLOAD_OPTIONAL_KEYS
    if unknown:
        # Security pre-vet M3 deny-by-default: any unknown key signals
        # a parent-side bug OR a compromised parent. Reject without
        # echoing the key name (which would be attacker-controllable).
        return _ex_usage("runner: payload has unknown top-level keys")

    # Type + range / regex checks. Booleans subclass int in Python; we
    # explicitly reject bool where int is expected.
    if not isinstance(payload["repo"], str) or not _HF_REPO_RE.match(
        payload["repo"]
    ):
        return _ex_usage(
            "runner: invalid `repo` — must match `author/name` "
            "HF slug pattern"
        )

    for str_key in ("prompt", "negative", "output_path"):
        if not isinstance(payload[str_key], str):
            return _ex_usage(
                "runner: string fields must be JSON strings"
            )

    for int_key in ("steps", "width", "height", "seed"):
        v = payload[int_key]
        if not isinstance(v, int) or isinstance(v, bool):
            return _ex_usage("runner: int fields must be JSON ints")
    if not (1 <= payload["steps"] <= 500):
        return _ex_usage("runner: steps out of range [1, 500]")
    for dim_key in ("width", "height"):
        if not (64 <= payload[dim_key] <= 8192):
            return _ex_usage(
                f"runner: {dim_key} out of range [64, 8192]"
            )

    guidance = payload["guidance"]
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)):
        return _ex_usage("runner: guidance must be a number")
    if not (0.0 <= float(guidance) <= 30.0):
        return _ex_usage("runner: guidance out of range [0.0, 30.0]")

    # output_path: absolute, safe ext. Security pre-vet HIGH: this is
    # where pipe.images[0].save(...) writes — must be in the user's
    # output tree, not /etc/passwd or similar.
    if not payload["output_path"].startswith("/"):
        return _ex_usage("runner: output_path must be absolute")
    out_ext = _splitext_lower(payload["output_path"])
    if out_ext not in _SAFE_OUTPUT_EXTS:
        return _ex_usage(
            "runner: output_path extension not in allowlist"
        )

    # input_path: optional. If present, absolute + safe image ext.
    input_path = payload.get("input_path")
    if input_path is not None:
        if not isinstance(input_path, str):
            return _ex_usage(
                "runner: input_path must be a JSON string or null"
            )
        if not input_path.startswith("/"):
            return _ex_usage("runner: input_path must be absolute")
        in_ext = _splitext_lower(input_path)
        if in_ext not in _SAFE_IMAGE_INPUT_EXTS:
            return _ex_usage(
                "runner: input_path extension not in image allowlist"
            )

    # cpu_offload_threshold_mp: optional, positive number.
    threshold = payload.get("cpu_offload_threshold_mp")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float)
        ):
            return _ex_usage(
                "runner: cpu_offload_threshold_mp must be a number"
            )
        if float(threshold) <= 0.0:
            return _ex_usage(
                "runner: cpu_offload_threshold_mp must be positive"
            )

    # param_overrides: optional, dict[str, primitive], allowlist-keyed.
    overrides = payload.get("param_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            return _ex_usage(
                "runner: param_overrides must be a JSON object"
            )
        for k in overrides:
            if not isinstance(k, str):
                return _ex_usage(
                    "runner: param_overrides keys must be strings"
                )
            if k not in _DIFFUSERS_PIPE_KWARG_ALLOWLIST:
                # Security pre-vet HIGH: any key outside the allowlist
                # is a potential injection vector into pipe(**kwargs).
                # Reject without echoing the key name.
                return _ex_usage(
                    "runner: param_overrides key not in allowlist"
                )
    return 0


def _splitext_lower(path: str) -> str:
    """``os.path.splitext`` returns (root, ext). Return only ext,
    lowercased. No tilde / glob expansion; pure string op."""
    idx = path.rfind(".")
    if idx == -1:
        return ""
    return path[idx:].lower()


def _open_output_for_save(output_path: str):
    """Open ``output_path`` with ``O_NOFOLLOW`` and return ``(file_obj,
    pil_format)``. The caller wraps file_obj in a ``with`` block so the
    fd is closed even if ``PIL.Image.save`` raises.

    v0.8.3 M-NEW-1 defence-in-depth: a pre-existing symlink at
    ``output_path`` (planted by a same-uid attacker) would otherwise
    let PIL dereference and write image bytes to whatever the symlink
    points at. Same-uid attacker already has direct file-write, so
    impact is bounded; this matches the rest of the runner's
    deny-by-default discipline.

    Raises ``OSError`` with ``errno.ELOOP`` if the path is a symlink
    (POSIX behaviour of ``O_NOFOLLOW``). The caller in ``main`` catches
    + emits a static stderr line + returns rc=1.

    The extension is already in ``_SAFE_OUTPUT_EXTS`` per
    ``_validate_payload_shape``, so the format lookup never raises
    KeyError in production — but a stand-alone test that bypasses
    validation would, hence the explicit ``ValueError`` fallback.
    """
    ext = _splitext_lower(output_path)
    pil_format = _PIL_FORMAT_BY_EXT.get(ext)
    if pil_format is None:
        raise ValueError(f"unsupported output extension: {ext!r}")
    fd = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_TRUNC,
        0o644,
    )
    return os.fdopen(fd, "wb"), pil_format


# ── Entry point ────────────────────────────────────────────────────────


def main() -> int:
    """Read payload from stdin, validate, dispatch to diffusers
    pipeline, write result image.

    Order (security pre-vet N2):
      1. Read up to ``_RUNNER_STDIN_MAX_BYTES + 1`` bytes; reject
         if over cap (DoS guard).
      2. ``json.loads`` (catch decode error).
      3. ``_validate_payload_shape`` (strict schema + repo regex +
         path checks + allowlist).
      4. Lazy import torch + diffusers (~3-5s cold-import cost —
         deferred so validation errors return in <100 ms).
      5. Pipeline load + run + save.
    """
    # Security pre-vet M4: set MPS-fallback flag BEFORE torch is
    # imported (which happens in step 4). Once torch is imported the
    # env var is captured and post-hoc setting has no effect. A
    # colleague's shell env could otherwise force strict-MPS mode
    # via PYTORCH_ENABLE_MPS_FALLBACK=0 and silently break
    # diffusers_mps on first unsupported op.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # 1. Bounded stdin read. Read MAX+1 so the strict-greater check
    # fires on exactly-MAX payloads (security pre-vet N1).
    try:
        raw = sys.stdin.buffer.read(_RUNNER_STDIN_MAX_BYTES + 1)
    except OSError as e:
        sys.stderr.write(f"runner: stdin read failed: {e}\n")
        return _EX_USAGE
    if len(raw) > _RUNNER_STDIN_MAX_BYTES:
        sys.stderr.write(
            f"runner: stdin payload exceeded "
            f"{_RUNNER_STDIN_MAX_BYTES} bytes — refusing oversize\n"
        )
        return _EX_USAGE

    # 2. JSON decode.
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _ex_usage("runner: stdin is not valid JSON")

    # 3. Strict schema validation.
    rc = _validate_payload_shape(payload)
    if rc != 0:
        return rc

    # 4. Lazy imports (~3-5s cold-import; deferred past validation
    # so invalid payloads fail fast).
    try:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import load_image
    except ImportError as e:
        sys.stderr.write(
            f"runner: diffusers stack import failed: {e}. "
            "Re-run bootstrap.sh and answer 'y' at the diffusers "
            "prompt (or set IMGEN_INSTALL_DIFFUSERS=1 for "
            "non-interactive install).\n"
        )
        return 3

    # 5. Pipeline load.
    pipe = DiffusionPipeline.from_pretrained(
        payload["repo"],
        torch_dtype=torch.bfloat16,
    )

    # CPU-offload above the per-model MP threshold; else direct MPS
    # transfer. Architect pre-vet H2 noted disk-write cost for offload
    # — the doctor reports free disk under ~/.cache/huggingface when
    # diffusers_mps models are declared; this runner trusts the
    # preflight gate at the parent (no point re-checking after
    # validation has passed).
    mp = (payload["width"] * payload["height"]) / 1_000_000.0
    threshold = payload.get("cpu_offload_threshold_mp", 2.0)
    if mp > float(threshold):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("mps")

    pipe_kwargs = dict(
        prompt=payload["prompt"],
        negative_prompt=payload["negative"] or None,
        num_inference_steps=payload["steps"],
        guidance_scale=payload["guidance"],
        width=payload["width"],
        height=payload["height"],
        generator=torch.Generator(device="mps").manual_seed(
            payload["seed"],
        ),
    )

    # param_overrides flows through as a flat dict, ALREADY FILTERED
    # by `_validate_payload_shape` against the allowlist. The schema
    # check is the single trust boundary.
    pipe_kwargs.update(payload.get("param_overrides") or {})

    if payload.get("input_path"):
        pipe_kwargs["image"] = load_image(payload["input_path"])

    result = pipe(**pipe_kwargs)

    # M-NEW-1 (v0.8.3): O_NOFOLLOW-guarded write. See
    # _open_output_for_save() for the threat-model rationale.
    try:
        out_fp, pil_format = _open_output_for_save(payload["output_path"])
    except OSError as e:
        # Static stderr — don't echo the user-controlled path back
        # (memo §I round-1 MEDIUM: control-byte avoidance in stderr).
        sys.stderr.write(
            f"runner: refused to write output_path (errno {e.errno}); "
            "path may be a symlink (O_NOFOLLOW) or permission denied\n"
        )
        return 1
    with out_fp:
        result.images[0].save(out_fp, format=pil_format)
    return 0


if __name__ == "__main__":
    sys.exit(main())

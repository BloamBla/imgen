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

# Safe extensions for the output PNG/JPG/WebP/MP4 and the optional
# input image. Mirrors paths.SAFE_OUTPUT_EXTS but defined locally so
# the runner has zero non-stdlib imports during validation (defense-
# in-depth: even if imgen.paths fails to import inside
# .venv-diffusers, validation still works). v0.9 commit 4 adds .mp4
# for VideoEngine output; the lock-in test
# ``test_runner_safe_output_exts_equals_paths_safe_output_exts``
# pins both sets identical.
_SAFE_OUTPUT_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".mp4"}
)
_SAFE_IMAGE_INPUT_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif",
})

# Per-output_type extension allowlist. The schema's matrix rule:
# image payloads MUST use the image set; video payloads MUST use the
# video set. .mp4 in an image payload (or vice versa) rejects before
# reaching the save path.
_SAFE_IMAGE_OUTPUT_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp"}
)
_SAFE_VIDEO_OUTPUT_EXTS: frozenset[str] = frozenset({".mp4"})

# Map _SAFE_OUTPUT_EXTS → PIL format keyword. When PIL.save is given
# an open file object (instead of a path string) it cannot infer the
# format from the extension and needs ``format=...`` explicitly.
_PIL_FORMAT_BY_EXT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}

# v0.9 commit 4 — pipeline class allowlist for the runner-side
# dispatch. Security §R.1 HIGH-1: validate as a LITERAL allowlist
# BEFORE the diffusers import so introspection / path-traversal /
# dunder strings never reach getattr-on-module. The actual class
# objects resolve via literal dict in _resolve_pipeline_class().
#
# v0.9.3 C2 (B-1 closure): expanded to include LTXImageToVideoPipeline
# for i2v dispatch. This runner set is a STRICT superset of the
# parent-side ``models._VIDEO_PIPELINE_CLASS_ALLOWLIST`` (the parent
# omits image-only ``"DiffusionPipeline"`` because it's wrong on a
# video Model). The subset invariant is locked by
# ``tests/test_v093_pipeline_class.TestAllowlistSubsetInvariant``.
_PIPELINE_CLASS_ALLOWLIST: frozenset[str] = frozenset({
    "DiffusionPipeline",         # generic v0.8 image fallback
    "LTXPipeline",               # v0.9.0 LTX-Video t2v class
    "LTXImageToVideoPipeline",   # v0.9.3 LTX-Video i2v class
})

# v0.9 commit 4 — fps allowlist for video payloads. Mirrors
# DiffusersMpsEngine._validate_video (parent side) per §E.0 drift
# lock-in.
_VIDEO_FPS_ALLOWLIST: frozenset[int] = frozenset({24, 25, 30})

# Output type allowlist. "image" is the default for legacy v0.8
# payloads that omit the key.
_OUTPUT_TYPE_ALLOWLIST: frozenset[str] = frozenset({"image", "video"})

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
    # v0.9 commit 4 — video payload extensions. All optional at the
    # top-level schema; output_type=="video" triggers conditional
    # required-ness for num_frames + fps + pipeline_class via the
    # per-output-type matrix in _validate_payload_shape.
    "num_frames",
    "fps",
    "output_type",
    "pipeline_class",
    "force_cpu_offload",
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

    v0.9 commit 4 adds the video-payload branch (§F). When
    ``output_type=="video"`` the additional triple (num_frames, fps,
    pipeline_class) becomes required and the output_path extension
    is gated to ``.mp4`` only. ``pipeline_class`` is validated via
    a LITERAL allowlist BEFORE the diffusers import (security §R.1
    HIGH-1 — getattr-on-module is the anti-pattern this avoids).
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

    # v0.9 commit 4 — output_type validation. Optional key; defaults
    # to "image" when absent (v0.8 payload compat). Allowlist
    # enforced BEFORE the per-type matrix below.
    if "output_type" in payload:
        ot = payload["output_type"]
        if not isinstance(ot, str) or ot not in _OUTPUT_TYPE_ALLOWLIST:
            return _ex_usage(
                "runner: output_type must be in {'image', 'video'}"
            )
        output_type = ot
    else:
        output_type = "image"

    # output_path: absolute, safe ext per output_type. Security
    # pre-vet HIGH: this is where pipe.images[0].save(...) writes —
    # must be in the user's output tree, not /etc/passwd or similar.
    # v0.9 commit 4: the output_type/extension matrix prevents a
    # video payload from silently writing a .png (or vice versa).
    if not payload["output_path"].startswith("/"):
        return _ex_usage("runner: output_path must be absolute")
    out_ext = _splitext_lower(payload["output_path"])
    if output_type == "video":
        if out_ext not in _SAFE_VIDEO_OUTPUT_EXTS:
            return _ex_usage(
                "runner: output_path extension not in video allowlist"
            )
    else:
        if out_ext not in _SAFE_IMAGE_OUTPUT_EXTS:
            return _ex_usage(
                "runner: output_path extension not in image allowlist"
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

    # ── v0.9 commit 4 — video-conditional schema branch ────────────
    #
    # When output_type=="video", three additional keys become required
    # (num_frames, fps, pipeline_class) with strict type/allowlist
    # checks. force_cpu_offload stays optional with a bool-only type.
    if output_type == "video":
        for required_video_key in ("num_frames", "fps", "pipeline_class"):
            if required_video_key not in payload:
                return _ex_usage(
                    "runner: video payload missing required key"
                )

        nf = payload["num_frames"]
        if isinstance(nf, bool) or not isinstance(nf, int):
            return _ex_usage("runner: num_frames must be a JSON int")
        if not (1 <= nf <= 1024):
            return _ex_usage("runner: num_frames out of range [1, 1024]")

        fps = payload["fps"]
        if isinstance(fps, bool) or not isinstance(fps, int):
            return _ex_usage("runner: fps must be a JSON int")
        if fps not in _VIDEO_FPS_ALLOWLIST:
            return _ex_usage("runner: fps not in {24, 25, 30}")

        # Security §R.1 HIGH-1: literal allowlist BEFORE any diffusers
        # import. The check fails closed for dunder strings, path
        # traversal, empty strings, and any non-allowlisted class name.
        pc = payload["pipeline_class"]
        if not isinstance(pc, str) or pc not in _PIPELINE_CLASS_ALLOWLIST:
            return _ex_usage(
                "runner: pipeline_class not in allowlist"
            )

        # force_cpu_offload: optional, bool-not-int discipline.
        if "force_cpu_offload" in payload:
            fco = payload["force_cpu_offload"]
            if not isinstance(fco, bool):
                return _ex_usage(
                    "runner: force_cpu_offload must be a JSON bool"
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


# ── Pipeline class resolver (v0.9 commit 4 §F) ────────────────────────


def _resolve_pipeline_class(name: str):
    """Literal dict dispatch from allowlisted name → class object.

    Security §R.1 HIGH-1: this is the SECOND of two layers around the
    getattr-on-module anti-pattern. The first layer is the schema
    check in :func:`_validate_payload_shape`, which rejects any
    non-allowlisted ``pipeline_class`` BEFORE the diffusers import
    runs. This function provides defence-in-depth — even if the
    schema check were bypassed by a future refactor, the literal
    dict lookup here fails closed for any name not in
    :data:`_PIPELINE_CLASS_ALLOWLIST`.

    The diffusers import is lazy (deferred until allowlist check
    passes) so a fail-closed call costs ~0ms, not the ~3-5s of cold
    diffusers import.
    """
    if name not in _PIPELINE_CLASS_ALLOWLIST:
        raise ValueError("pipeline_class not in allowlist")
    # Lazy diffusers import — only reached after allowlist passes.
    # v0.9.3 C2 (B-1 closure): LTXImageToVideoPipeline added for i2v
    # dispatch. LTXImageToVideoPipeline has been in diffusers ≥0.32
    # alongside LTXPipeline — same checkpoint, image-conditioning
    # frontend.
    from diffusers import (
        DiffusionPipeline,
        LTXImageToVideoPipeline,
        LTXPipeline,
    )
    classes = {
        "DiffusionPipeline": DiffusionPipeline,
        "LTXPipeline": LTXPipeline,
        "LTXImageToVideoPipeline": LTXImageToVideoPipeline,
    }
    return classes[name]


# ── Output_type dispatchers ───────────────────────────────────────────


def _run_image(payload: dict) -> int:
    """v0.8 image path — extracted unchanged from pre-v0.9 ``main()``.

    Reaches here only after :func:`_validate_payload_shape` passed the
    payload (steps 1-3 in :func:`main`). All keys referenced below
    are either required or validated-optional per the schema.
    """
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

    pipe = DiffusionPipeline.from_pretrained(
        payload["repo"],
        torch_dtype=torch.bfloat16,
    )

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
    pipe_kwargs.update(payload.get("param_overrides") or {})

    if payload.get("input_path"):
        pipe_kwargs["image"] = load_image(payload["input_path"])

    result = pipe(**pipe_kwargs)

    try:
        out_fp, pil_format = _open_output_for_save(payload["output_path"])
    except OSError as e:
        sys.stderr.write(
            f"runner: refused to write output_path (errno {e.errno}); "
            "path may be a symlink (O_NOFOLLOW) or permission denied\n"
        )
        return 1
    with out_fp:
        result.images[0].save(out_fp, format=pil_format)
    return 0


def _run_video(payload: dict) -> int:
    """v0.9 commit 4 video path — LTX-shaped pipeline → MP4 output
    via atomic-rename pattern.

    Per §F: ``imageio.mimsave`` doesn't accept fd-mode for libx264
    muxing (moov atom requires seekable container), so the
    O_NOFOLLOW write pattern from the image path doesn't apply.
    Instead:

    1. Refuse if ``output_dir`` is a symlink (parent-traversal
       protection; symlinked output_path itself is replaced
       atomically by the rename).
    2. Write inference frames to a ``NamedTemporaryFile`` in the
       same directory as output_path (same-fs requirement for atomic
       os.rename).
    3. ``os.rename`` the temp to output_path — atomic; replaces any
       pre-existing symlink at output_path with a regular file.

    On any failure between step 2 and step 3 the temp file is
    unlinked before re-raising; output_dir stays clean.

    Trust boundary: ``output_path`` is already validated by
    :func:`_validate_payload_shape` (absolute, .mp4 extension), and
    ``pipeline_class`` is allowlisted before reaching
    :func:`_resolve_pipeline_class`. ffmpeg subprocess is launched
    by imageio-ffmpeg's BUNDLED binary inside .venv-diffusers — no
    system-PATH ffmpeg invocation.
    """
    import tempfile
    from pathlib import Path

    output_path = Path(payload["output_path"])
    output_dir = output_path.parent

    # 1. Parent-traversal symlink guard. The output_path itself can
    # be a pre-existing symlink — os.rename replaces it (which we
    # own semantically per §F atomic-rename design). But output_dir
    # being a symlink means a same-uid attacker could redirect the
    # write upstream of our checks; refuse.
    if output_dir.is_symlink():
        sys.stderr.write(
            "runner: refused to write — output_dir is symlink\n"
        )
        return 1

    try:
        import torch
        import imageio
    except ImportError as e:
        sys.stderr.write(
            f"runner: diffusers/imageio stack import failed: {e}. "
            "Re-run bootstrap.sh and answer 'y' at the diffusers "
            "prompt (or set IMGEN_INSTALL_DIFFUSERS=1 for "
            "non-interactive install). For video output, ensure "
            "imageio + imageio-ffmpeg are installed in "
            ".venv-diffusers (see imgen video --help).\n"
        )
        return 3

    pipeline_class = _resolve_pipeline_class(payload["pipeline_class"])

    pipe = pipeline_class.from_pretrained(
        payload["repo"],
        torch_dtype=torch.bfloat16,
    )

    if payload.get("force_cpu_offload", False):
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
        num_frames=payload["num_frames"],
        generator=torch.Generator(device="mps").manual_seed(
            payload["seed"],
        ),
    )
    pipe_kwargs.update(payload.get("param_overrides") or {})

    result = pipe(**pipe_kwargs)
    # diffusers LTX pipelines return result.frames[0] = list of PIL
    # Images (one batch element). Image pipelines return result.images.
    frames = result.frames[0]

    # 2. Write to NamedTemporaryFile in output_dir (same-fs for
    # atomic os.rename). delete=False so the path persists between
    # the close-of-handle and imageio.mimsave open.
    with tempfile.NamedTemporaryFile(
        dir=str(output_dir),
        suffix=".mp4",
        prefix=".imgen-video-",
        delete=False,
    ) as tmp:
        tmp_path = tmp.name

    try:
        imageio.mimsave(
            tmp_path, frames,
            fps=payload["fps"],
            codec="libx264",
            quality=8,
        )
        # 3. Atomic rename — replaces any symlink at output_path
        # with a regular file. Race window between dir symlink check
        # and write closes here.
        os.rename(tmp_path, output_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return 0


# ── Entry point ────────────────────────────────────────────────────────


def main() -> int:
    """Read payload from stdin, validate, dispatch by output_type
    to ``_run_video`` or ``_run_image``.

    Order (security pre-vet N2):
      1. Read up to ``_RUNNER_STDIN_MAX_BYTES + 1`` bytes; reject
         if over cap (DoS guard).
      2. ``json.loads`` (catch decode error).
      3. ``_validate_payload_shape`` (strict schema + repo regex +
         path checks + allowlist + per-output-type matrix).
      4. Dispatch by ``payload["output_type"]`` (defaults to
         "image" for v0.8 compat).
      5. Lazy import torch + diffusers (~3-5s cold-import cost —
         deferred so validation errors return in <100 ms) inside
         the dispatched function.
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

    # 4. Dispatch by output_type (v0.8 compat: absent ⇒ image).
    if payload.get("output_type") == "video":
        return _run_video(payload)
    return _run_image(payload)


if __name__ == "__main__":
    sys.exit(main())

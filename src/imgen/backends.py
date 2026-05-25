"""mflux backend registry.

Each Backend captures the variant-specific behavior of an mflux binary:
which executable to call, whether HF token is needed, the spelling of the
image-input flag, and which optional flags it supports.

Two sources of backends:

* **Built-ins** (``BUILTIN_BACKENDS`` / legacy alias ``BACKENDS``) — FLUX
  and Qwen, hardcoded. Modifying these needs a code change.
* **User TOMLs** (v0.4) — ``~/.imgen/backends.d/*.toml``. Drop a file in
  to add a new ``--backend NAME`` option without a code change. Filename
  stem is the backend name; collisions with built-ins get a ``_0001``
  suffix (built-ins win), mirroring ``styles.d`` semantics.

``frozen=True`` so registry entries can't be mutated at runtime
(constants should stay constant). ``slots=True`` for tighter memory +
early errors on typo'd attribute access.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ._safe import has_control_bytes as _has_control_bytes
from ._schema import validate_against_schema

__all__ = [
    "BACKENDS",
    "BUILTIN_BACKENDS",
    "USER_BACKEND_MAX_BYTES",
    "Backend",
    "UserBackendError",
    "build_mflux_cmd",
    "filter_compatible_loras",
    "get_backend",
    "list_backends",
    "load_user_backend_file",
    "load_user_backends_dir",
    "merge_user_backends",
    "model_from_backend",
    "reset_backends_cache",
    "validate_user_backend_schema",
]

# Cap user backend TOML file size. The largest realistic backend TOML
# is on the order of a few hundred bytes (binary path + flag set +
# optional secret section). 16 KB is several orders above realistic
# use — a larger file means corruption or a misuse, refuse rather
# than slurp into memory.
USER_BACKEND_MAX_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class Backend:
    binary: str                  # basename of the mflux entry-point script
    needs_token: bool            # gated HF repo → require ~/.imgen/hf_token / $HF_TOKEN
    image_flag: str              # mflux's input-image flag (--image-path vs --image-paths)
    supports_strength: bool      # accepts --image-strength
    supports_negative: bool      # accepts --negative-prompt
    extra_args: tuple[str, ...]  # fixed flags appended unconditionally (e.g. --model X)
    # v0.4: custom backends (registered via ~/.imgen/backends.d/*.toml)
    # may declare a single env var that imgen forwards from the parent
    # environment into the subprocess. None on built-ins; FLUX keeps
    # using the legacy ``needs_token=True`` path with ~/.imgen/hf_token
    # because that path also owns whoami validation + atomic save —
    # generalizing those is out of scope for v0.4 (see
    # project_v040_design.md, decision 2 + schema migration trap).
    secret_env_var: str | None = None
    # When ``secret_env_var`` is set: die at command-construction time
    # if the env var is missing from os.environ. False means "best
    # effort" — forward if set, silently skip if not, let the backend
    # binary report its own auth failure.
    secret_required: bool = True
    # v0.5: LLM prompt enhancer. ``enhance_system_prompt`` is the system
    # instruction fed to the local LLM (Qwen2.5-7B-Instruct by default)
    # when ``--enhance-prompt`` is active. None on a backend = enhancer
    # is silently skipped for that backend (fail-safe — better no-op
    # than a generic instruction that might mis-shape the prompt for
    # this backend's conventions). ``enhance_invariants`` are substrings
    # that, if present in the input prompt, must also be present in the
    # enhanced output — otherwise we fall back to the original. This is
    # defence-in-depth: the system prompt explicitly tells the LLM to
    # keep clauses like ``while preserving …`` intact, this is the
    # tripwire that catches LLM drift.
    enhance_system_prompt: str | None = None
    enhance_invariants: tuple[str, ...] = ()
    # v0.6: LoRA compatibility group identifier. Style presets declare a
    # tuple of compat-groups their LoRAs were trained against (in
    # ``LoraRef.compatible_with``); at command-construction time, only
    # LoRAs whose ``compatible_with`` includes this backend's group are
    # applied. Built-in flux is ``"flux-1"`` (FLUX.1-Kontext-dev shares
    # the FLUX.1 architecture family with FLUX.1-dev and FLUX.1-schnell
    # — most FLUX.1 LoRAs work across the family). Built-in qwen is
    # ``"qwen"``. Empty string (default for user backends that don't
    # declare it) means "no LoRA support" — any LoRA reference in a
    # style is silently skipped with a warn for this backend.
    lora_compat_group: str = ""
    # v0.7.0 (post-tag review UX-gap fix): HuggingFace gated-repo URL
    # path (e.g. ``"black-forest-labs/FLUX.1-dev"``) for the model
    # this backend loads. When mflux exits non-zero AND the user's
    # token is set but the HF API returns 401 / GatedRepoError, the
    # cmd_draw post-run hint surfaces this URL so the user can accept
    # the per-repo license (gated-repo licenses are accepted per-model
    # on HF — sharing a token across two BFL repos doesn't auto-share
    # the license-grant). None for backends that don't need a gated
    # repo (qwen) or for user TOMLs that don't declare it.
    hf_gated_repo: str | None = None
    # v0.8.1 HIGH-2 closure: v0.8 Model-shape fields preserved on Backend
    # so user TOMLs can declare them in ~/.imgen/models.d/*.toml and they
    # flow through to Engine routing via ``model_from_backend(name, b)``.
    # Defaults match the v0.7 Backend behaviour, so a TOML that omits
    # them loads unchanged — additive forward-compat per memo §F's
    # "user TOMLs' v0.8 schema is a SUPERSET of v0.7" promise.
    #
    # ``engine`` — which Engine implementation handles this backend.
    # ``"mflux"`` is the default; ``"diffusers_mps"`` routes through
    # DiffusersMpsEngine when the helper ``model_from_backend`` constructs
    # a Model from this Backend. v0.8.1 schema enforces the binary/repo
    # invariant (binary required iff engine="mflux"; repo required iff
    # engine="diffusers_mps") at validation time.
    engine: str = "mflux"
    # ``repo`` — HF Hub repo id (e.g. ``"Qwen/Qwen-Image-2512"``). Used
    # only by diffusers_mps engine; None / ignored on mflux.
    repo: str | None = None
    # ``cpu_offload_threshold_mp`` — diffusers_mps activates pipeline
    # cpu-offload at output >= this many megapixels. Sentinel-ish: a
    # very high default (999.0) means "never offload" so the field
    # doesn't accidentally fire on mflux backends.
    cpu_offload_threshold_mp: float = 999.0
    # ``ram_baseline_gb`` / ``ram_slope_gb_per_mp`` / ``encoder_ram_gb``
    # — per-Model RAM math (memo §L). Defaults are flux-class
    # conservative values so a v0.7-shape user TOML gets a reasonable
    # preflight estimate without declaring them. Built-in registry rows
    # in models.py override with calibrated per-model anchors.
    ram_baseline_gb: float = 13.5
    ram_slope_gb_per_mp: float = 4.0
    encoder_ram_gb: float = 0.0
    # ``default_steps`` / ``default_guidance`` / ``min_guidance`` /
    # ``max_guidance`` — per-Model param defaults consulted by
    # cmd_helpers._resolve_iteration_params and Engine.validate.
    default_steps: int = 20
    default_guidance: float = 3.5
    min_guidance: float = 0.0
    max_guidance: float = 10.0
    # ``supported_quants`` — quantize ladder. mflux backends usually
    # support (3, 4, 5, 6, 8); diffusers_mps usually omits quantize
    # entirely (omit_quantize=True) so the tuple may be empty.
    supported_quants: tuple[int, ...] = (3, 4, 5, 6, 8)
    omit_quantize: bool = False
    # ``param_overrides`` — tuple-of-tuples for frozen-friendly storage;
    # Engine code converts to dict at call boundary. Allowlist-keyed at
    # runtime (only diffusers_mps recognises specific keys today).
    param_overrides: tuple[tuple[str, object], ...] = ()


# ── System prompt constants — re-exported from models.py at v0.8.0 ─────
#
# These v0.5+ enhancer system prompts moved to models.py at 4b (canonical
# location alongside the Model rows that reference them). Imported here
# for back-compat with any v0.7 caller that still does
# `from imgen.backends import _FLUX_KONTEXT_ENHANCE_SYS` — internal use
# only, leading-underscore name signals "do not depend on this". The
# direct user is tests/test_enhance*.py which should migrate to
# importing from imgen.models in a follow-up cleanup.
from .models import (
    _FLUX_DEV_DRAW_ENHANCE_SYS,
    _FLUX_KONTEXT_ENHANCE_SYS,
    _IDENTITY_ANCHOR_INVARIANTS,
    _QWEN_EDIT_ENHANCE_SYS,
    BUILTIN_MODELS,
    Model,
    _V07_TO_V08_MODEL_RENAMES,
)


# ── BUILTIN_BACKENDS — backward-derived v0.7-keyed view (4b) ──────────
#
# Per [[project-v080-design]] §Q commit 4b + architect 4b pre-vet HIGH-1:
# the canonical registry source-of-truth is now ``models.BUILTIN_MODELS``
# (v0.8-keyed). ``BUILTIN_BACKENDS`` becomes a DERIVED BACKWARD view
# keyed by v0.7 names so v0.7.x test fixtures (test_backends.py asserts
# ``BACKENDS["flux"].binary == "mflux-generate-kontext"``) stay green
# without churn.
#
# Forward conversion (Model → Backend) drops v0.8-only fields (engine,
# repo, ram_*, default_*, supported_quants, omit_quantize,
# param_overrides) — Backend dataclass doesn't have them. v0.4 secret
# fields (secret_env_var, secret_required) default to None / True since
# built-in Models don't declare them.
#
# Per architect 4b pre-vet N-3: this derivation runs at module load
# (eager snapshot). If BUILTIN_MODELS were mutated post-load (it's a
# regular dict, no MappingProxyType), BUILTIN_BACKENDS wouldn't reflect
# — that's intentional, registry is conceptually constant.


def _backend_from_model(m: Model, v07_name: str) -> Backend:
    """Convert a v0.8 Model → Backend (for backward-derived
    BUILTIN_BACKENDS view + back-compat shim test fixtures).

    v0.8.1 (HIGH-2 closure): preserves ALL v0.8 Model fields on Backend
    so a downstream ``model_from_backend(name, b)`` can losslessly
    reconstruct a Model for Engine routing — used by both the built-in
    path (Model → Backend → Model round-trip; preserved fields make it
    pure pass-through) and the user-TOML path (Backend constructed by
    ``validate_user_backend_schema`` carries the declared fields too).
    """
    return Backend(
        binary=m.binary or "",
        needs_token=m.needs_token,
        image_flag=m.image_flag or "--image-path",
        supports_strength=m.supports_strength,
        supports_negative=m.supports_negative,
        extra_args=m.extra_args,
        # Built-in Models never have v0.4 secret-env fields set; user
        # TOMLs going through validate_user_backend_schema preserve
        # them directly (this helper is only for the built-in
        # backward-derivation path).
        secret_env_var=None,
        secret_required=True,
        enhance_system_prompt=m.enhance_system_prompt,
        enhance_invariants=m.enhance_invariants,
        lora_compat_group=m.lora_compat_group,
        hf_gated_repo=m.hf_gated_repo,
        # v0.8.1 HIGH-2 closure: preserve v0.8 Model fields on Backend.
        engine=m.engine,
        repo=m.repo,
        cpu_offload_threshold_mp=m.cpu_offload_threshold_mp,
        ram_baseline_gb=m.ram_baseline_gb,
        ram_slope_gb_per_mp=m.ram_slope_gb_per_mp,
        encoder_ram_gb=m.encoder_ram_gb,
        default_steps=m.default_steps,
        default_guidance=m.default_guidance,
        min_guidance=m.min_guidance,
        max_guidance=m.max_guidance,
        supported_quants=m.supported_quants,
        omit_quantize=m.omit_quantize,
        param_overrides=m.param_overrides,
    )


def model_from_backend(name: str, b: Backend) -> Model:
    """Convert a Backend → Model for Engine routing (v0.8.1 HIGH-2 closure).

    Pure function — built-ins go Model → Backend at module load (via
    ``_backend_from_model``) and this is the round-trip; user TOMLs
    go TOML → Backend via ``validate_user_backend_schema`` and this is
    the first Model construction for them. ``__post_init__`` invariants
    (engine in {mflux, diffusers_mps}; binary/repo required per engine;
    ram fields > 0) fire at every call, so a hand-edited user TOML
    that escaped one of the schema validators still gets caught at
    Engine-routing time.

    ``name`` is consumed in the error message ``__post_init__`` raises
    when invariants fail — no other side-effect.

    Used by ``_model_for_validate`` in cmd_helpers.py to extend Engine
    routing to user TOMLs after they declare v0.8 fields.
    """
    return Model(
        engine=b.engine,
        binary=b.binary if b.engine == "mflux" else None,
        repo=b.repo,
        extra_args=b.extra_args,
        image_flag=b.image_flag if b.engine == "mflux" else None,
        cpu_offload_threshold_mp=b.cpu_offload_threshold_mp,
        supports_strength=b.supports_strength,
        supports_negative=b.supports_negative,
        needs_token=b.needs_token,
        lora_compat_group=b.lora_compat_group,
        hf_gated_repo=b.hf_gated_repo,
        default_steps=b.default_steps,
        default_guidance=b.default_guidance,
        min_guidance=b.min_guidance,
        max_guidance=b.max_guidance,
        supported_quants=b.supported_quants,
        omit_quantize=b.omit_quantize,
        param_overrides=b.param_overrides,
        ram_baseline_gb=b.ram_baseline_gb,
        ram_slope_gb_per_mp=b.ram_slope_gb_per_mp,
        encoder_ram_gb=b.encoder_ram_gb,
        enhance_system_prompt=b.enhance_system_prompt,
        enhance_invariants=b.enhance_invariants,
    )


def _build_v07_compat_backends() -> dict[str, Backend]:
    """Build the v0.7-keyed BUILTIN_BACKENDS view from BUILTIN_MODELS.

    Inverts ``_V07_TO_V08_MODEL_RENAMES`` to map v0.8 canonical names
    back to v0.7 keys (``flux-kontext`` → ``flux``,
    ``qwen-image-edit-v1`` → ``qwen``). Unchanged names (``flux-dev``,
    ``flux2-klein-edit-9b``) pass through.

    Result: dict[v0.7-key, Backend] suitable for legacy callers and
    test_backends.py assertions.
    """
    v08_to_v07: dict[str, str] = {
        v08: v07 for v07, v08 in _V07_TO_V08_MODEL_RENAMES.items()
    }
    return {
        v08_to_v07.get(v08_name, v08_name): _backend_from_model(m, v08_name)
        for v08_name, m in BUILTIN_MODELS.items()
    }


BUILTIN_BACKENDS: dict[str, Backend] = _build_v07_compat_backends()

# Backwards-compatible alias. Points at the built-in dict only — DO NOT
# read from this in code that needs to see user backends from
# ~/.imgen/backends.d/. Use ``get_backend()`` / ``list_backends()``
# instead, which transparently include user TOMLs. Kept so existing
# test_backends.py (and any downstream code expecting ``BACKENDS``)
# keeps working — those callers only care about the built-in set.
BACKENDS: dict[str, Backend] = BUILTIN_BACKENDS


class UserBackendError(Exception):
    """Raised when a user TOML in ~/.imgen/backends.d/ has bad shape/values.

    Caught by ``load_user_backends_dir`` so a single malformed file
    warns-and-skips rather than killing the load of the rest. Mirrors
    ``UserStyleError`` (intentionally — same caller behaviour, same
    exception shape, see project_v040_design.md decision 4).
    """


# ── Field validators ─────────────────────────────────────────────────────


_IMAGE_FLAG_CHOICES = {"--image-path", "--image-paths"}

# POSIX-ish env var name: leading letter or underscore, then alphanums
# and underscores. The shell will export anything technically (`env
# 'WEIRD-NAME=x' foo` works), but standard-named vars are what users
# actually use and what tooling expects. Reject the rest at schema
# time so malformed names surface BEFORE subprocess launch.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_str_nonempty(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_list_of_clean_str(v: Any) -> bool:
    """List of strings with no control bytes anywhere. Mirrors the
    binary-field defence: argv elements go straight to execvp, not
    a shell, so injection isn't the concern — but a flag value
    containing `\\x1b` would leak escape sequences into mflux's
    stderr (and from there into our log files via the
    redaction-tee). Reject at schema time. (v0.4 security-reviewer
    NIT-2.)"""
    return (
        isinstance(v, list)
        and all(isinstance(x, str) and not _has_control_bytes(x) for x in v)
    )


def _is_clean_str(v: Any) -> bool:
    """String with no C0/DEL/C1 control bytes. Used for user-supplied
    fields that flow into prompts / log files / terminal output (the
    enhance system prompt is a prime example: it's a free-form
    instruction the user types into a TOML, and it ends up displayed
    in dry-run output and the per-batch log)."""
    return isinstance(v, str) and not _has_control_bytes(v)


_USER_BACKEND_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "binary": ("non-empty string", _is_str_nonempty),
    "image_flag": (
        f"one of {sorted(_IMAGE_FLAG_CHOICES)}",
        lambda v: isinstance(v, str) and v in _IMAGE_FLAG_CHOICES,
    ),
    "supports_strength": ("bool", lambda v: isinstance(v, bool)),
    "supports_negative": ("bool", lambda v: isinstance(v, bool)),
    "extra_args": (
        "list of strings (no control bytes)", _is_list_of_clean_str,
    ),
    # v0.5: optional per-backend enhancer config. A user backend may
    # declare its own system prompt + identity-anchor invariants so
    # ``--enhance-prompt`` works on it the same way it works on the
    # built-in flux/qwen backends. Both fields are optional — absent →
    # Backend.enhance_system_prompt stays None → enhancer cleanly skips
    # with ``fallback_reason="not_supported_by_backend"`` (no crash,
    # just a no-op for that backend). control-byte filter on the
    # system prompt because it ends up in subprocess argv (via JSON
    # stdin to enhance_runner) AND in dry-run terminal display.
    "enhance_system_prompt": (
        "string (no control bytes)", _is_clean_str,
    ),
    "enhance_invariants": (
        "list of strings (no control bytes)", _is_list_of_clean_str,
    ),
    # v0.6: LoRA compatibility group identifier (see Backend
    # dataclass docstring). User backends opt into LoRA support by
    # declaring this. The string is the identifier style TOMLs use in
    # their LoraRef.compatible_with field; common values: "flux-1",
    # "flux-2", "qwen", "z-image", "fibo". Bare lower-case stems for
    # readability + control-byte safety (it ends up in warn() output
    # when a LoRA is skipped for compat mismatch).
    "lora_compat_group": (
        "non-empty string (no control bytes)",
        lambda v: isinstance(v, str) and v.strip() != "" and _is_clean_str(v),
    ),
    # v0.7.12 (gap 5): HF gated-repo identifier, e.g. "briaai/FIBO" or
    # "black-forest-labs/FLUX.1-dev". When mflux subprocess fails with
    # GatedRepoError, ``cmd_draw`` / ``cmd_generate`` / ``cmd_batch``
    # post-failure hint surfaces a URL pointing at the per-model
    # license page on huggingface.co so the user can accept the gate
    # and retry — same UX as built-in flux / flux-dev / flux2-klein-
    # edit-9b rows. Pre-v0.7.12 only built-in backends could set this
    # (Backend.hf_gated_repo existed but user TOMLs got "unknown field"
    # warn). Control-byte filter because the value ends up rendered in
    # terminal output via the post-failure hint.
    "hf_gated_repo": (
        "non-empty string (no control bytes)",
        lambda v: isinstance(v, str) and v.strip() != "" and _is_clean_str(v),
    ),
    # ── v0.8.1 HIGH-2 closure: v0.8 OPTIONAL Model-shape fields ────────
    # Per [[project-v080-design]] §F line 766: "user TOMLs' v0.8 schema
    # is a SUPERSET of v0.7". All optional with defaults that match v0.7
    # behaviour, so an existing v0.7-shape TOML loads unchanged.
    "engine": (
        "one of {'mflux', 'diffusers_mps'}",
        lambda v: isinstance(v, str) and v in {"mflux", "diffusers_mps"},
    ),
    "repo": (
        "non-empty string (HF Hub repo id; no control bytes)",
        lambda v: isinstance(v, str) and v.strip() != "" and _is_clean_str(v),
    ),
    # RAM math: floats, strictly positive. Sentinel 0.0 fails
    # __post_init__ on Model construction (memo §L), so a user TOML
    # declaring 0.0 here would crash at engine-routing time — reject
    # at schema time for a clearer message.
    "ram_baseline_gb": (
        "positive float",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and float(v) > 0.0,
    ),
    "ram_slope_gb_per_mp": (
        "positive float",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and float(v) > 0.0,
    ),
    "encoder_ram_gb": (
        "non-negative float",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and float(v) >= 0.0,
    ),
    # Param defaults — bounded by the existing CLI ranges so an out-of-
    # band TOML default doesn't surface as a downstream CLI validation
    # failure (better diagnostic at schema time).
    "default_steps": (
        "int in 1..200",
        lambda v: isinstance(v, int) and not isinstance(v, bool)
        and 1 <= v <= 200,
    ),
    "default_guidance": (
        "float in 0.0..15.0",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and 0.0 <= float(v) <= 15.0,
    ),
    "min_guidance": (
        "float in 0.0..15.0",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and 0.0 <= float(v) <= 15.0,
    ),
    "max_guidance": (
        "float in 0.0..15.0",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and 0.0 <= float(v) <= 15.0,
    ),
    # supported_quants: list of ints from {3, 4, 5, 6, 8}. Empty allowed
    # for engines that don't quantize (diffusers_mps). bool sub-check
    # because Python's bool is an int subtype.
    "supported_quants": (
        "list of ints from {3, 4, 5, 6, 8} (may be empty)",
        lambda v: (
            isinstance(v, list)
            and all(
                isinstance(x, int) and not isinstance(x, bool) and x in {3, 4, 5, 6, 8}
                for x in v
            )
        ),
    ),
    "omit_quantize": ("bool", lambda v: isinstance(v, bool)),
    "cpu_offload_threshold_mp": (
        "positive float",
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        and float(v) > 0.0,
    ),
    # param_overrides: TOML array of [str, value] pairs. Mirrors the
    # ``tuple[tuple[str, object], ...]`` runtime shape on Model.
    # Allowlist-keyed at the engine runtime (today: diffusers_mps only
    # recognises specific keys like "true_cfg_scale"); schema here
    # validates structure only.
    "param_overrides": (
        "list of [key, value] pairs (key: non-empty string, no control bytes)",
        lambda v: (
            isinstance(v, list)
            and all(
                isinstance(pair, list) and len(pair) == 2
                and isinstance(pair[0], str) and pair[0].strip() != ""
                and not _has_control_bytes(pair[0])
                for pair in v
            )
        ),
    ),
}

_SECRET_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "env_var": (
        "non-empty string matching ^[A-Za-z_][A-Za-z0-9_]*$",
        lambda v: isinstance(v, str) and _ENV_VAR_NAME_RE.match(v) is not None,
    ),
    "required": ("bool", lambda v: isinstance(v, bool)),
}

# Dynamic-linker / interpreter override env vars that we refuse to
# forward into the subprocess even if a user TOML declares them. The
# whole point of subprocess_helpers._MFLUX_ENV_ALLOWLIST is to keep
# unknown env vars out of the child process; a malicious or naive
# TOML declaring secret.env_var = "LD_PRELOAD" would have bypassed
# that allowlist by going through the build_mflux_env backend_secret
# path. Reject at schema time so a forum-distributed sdxl.toml can't
# exploit a user who happens to have LD_PRELOAD set for legitimate
# reasons. (v0.4 security-reviewer IMP-1.)
_DANGEROUS_ENV_VARS: frozenset[str] = frozenset({
    # dyld (macOS) — see man dyld(1)
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_VERSIONED_FRAMEWORK_PATH",
    "DYLD_VERSIONED_LIBRARY_PATH", "DYLD_FORCE_FLAT_NAMESPACE",
    "DYLD_SHARED_REGION", "DYLD_PRINT_LIBRARIES",
    # Linux / glibc (here for paranoia; imgen runs on macOS but a user
    # could share TOMLs with a Linux colleague using mflux there too).
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_ASSUME_KERNEL",
    # Python interpreter override — could rebind imports or run code
    # before main(). Most subprocess binaries aren't Python, but mflux
    # IS Python so this is load-bearing for mflux backends.
    "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONEXECUTABLE",
    "PYTHONNOUSERSITE", "PYTHONUSERBASE",
})

# Fields with no schema entry but a documented default. Validator fills
# these in if the TOML doesn't override.
_USER_BACKEND_DEFAULTS: dict[str, Any] = {
    "supports_strength": False,
    "supports_negative": False,
    "extra_args": (),
}


def _validate_binary_field(value: str, source: Path) -> None:
    """Reject control bytes and embedded relative paths in ``binary``.

    Two acceptable shapes:
      * **Bare name** — no '/' anywhere, resolved via $PATH at exec time
        (e.g. ``"mflux-generate-sdxl"``).
      * **Absolute path** — starts with '/', used as-is, must exist on
        disk at validation time.

    Relative paths (``./bin/foo``, ``../bin/foo``, ``sub/dir/foo``) are
    rejected: their meaning depends on CWD at exec time, which is
    different from CWD at TOML-parse time, and gives no clear semantics
    for the user.
    """
    if _has_control_bytes(value):
        raise UserBackendError(
            f"{source}: binary contains control bytes (C0/DEL/C1) — "
            "reject so they don't leak into subprocess argv or logs"
        )
    if "/" in value:
        # If it has any slash, the only allowed shape is absolute.
        if not value.startswith("/"):
            raise UserBackendError(
                f"{source}: binary {value!r} is a relative path — use "
                "either a bare command name (PATH-resolvable) or an "
                "absolute path starting with '/'"
            )
        # is_file() (not exists()) so a directory at the path is also
        # rejected — `binary = "/usr/local/bin"` would otherwise pass
        # validation and crash at subprocess.Popen with an opaque
        # IsADirectoryError. (v0.4 python-reviewer IMP-1.)
        if not Path(value).is_file():
            raise UserBackendError(
                f"{source}: binary {value!r} is not a regular file "
                "(must be the executable itself, not a directory)"
            )
        # TOCTOU note: validator checks at parse time, subprocess.Popen
        # happens later. Same trust boundary as paths.ensure_state_dir:
        # a same-uid attacker already has direct file access, so racing
        # the binary path doesn't unlock anything they can't do directly.
        # (v0.4 security-reviewer NIT-1.)


# v0.8.2 NIT-B: per-engine "field doesn't apply here" sets. Used by the
# load-time warn surface. Conservative — only fields the engine has NO
# conceivable use for, not "engine probably doesn't care". Adding to
# these sets without testing is a UX regression risk (over-warning is
# almost as bad as silent ignore).
_INAPPLICABLE_FIELDS_FOR_ENGINE: dict[str, frozenset[str]] = {
    "mflux": frozenset({
        # diffusers_mps-only routing fields
        "repo",
        "cpu_offload_threshold_mp",
        # param_overrides is currently a diffusers_mps-only feature
        # (allowlist-keyed at the runner trust boundary); mflux argv
        # composition has no consumer for it.
        "param_overrides",
    }),
    "diffusers_mps": frozenset({
        # mflux argv-composition fields — diffusers_mps doesn't have
        # an argv layer at all (JSON-stdin transport via
        # _diffusers_runner).
        "binary",
        "image_flag",
        "extra_args",
    }),
}


def _warn_inapplicable_fields_per_engine(
    engine: str, data: dict, source: Path,
) -> None:
    """Emit a per-field warn for fields SET in ``data`` but
    inapplicable for the declared ``engine``. Caller: load-time
    validator, AFTER schema gate, BEFORE Backend construction.

    Pure side-effect (prints warn lines). The inapplicable fields
    still flow into Backend storage (with their TOML-declared
    values); this just surfaces the dead-field UX gap to the user.
    """
    from .colors import warn as _warn

    inapplicable = _INAPPLICABLE_FIELDS_FOR_ENGINE.get(engine, frozenset())
    for field in sorted(inapplicable & set(data.keys())):
        _warn(
            f"{source}: {field!r} is set but inapplicable for "
            f"engine={engine!r} — field will be stored but never "
            f"consulted at runtime. Safe to delete from the TOML."
        )


def validate_user_backend_schema(data: dict, source: Path) -> Backend:
    """Turn a parsed TOML dict into a Backend instance, enforcing schema.

    Unknown top-level fields warn and are dropped (forward-compat with
    future additions). Known fields with bad types/values raise
    UserBackendError carrying the source path for diagnostics.

    The ``[secret]`` section is optional. If present, ``env_var`` must
    be set; ``required`` defaults to True.

    Mirrors :func:`styles.load_user_style_file` shape. Both call sites
    go through the shared ``_schema.py::validate_against_schema`` helper
    extracted in v0.4 (closed the v0.2-era 3-callers-rule trigger).
    """
    from .colors import warn

    # Required fields — engine-conditional per v0.8.1 HIGH-2 closure.
    # ``engine`` itself is OPTIONAL (defaults to "mflux"); the
    # required-field set then differs by engine:
    #   mflux        → binary + image_flag (v0.7 contract, unchanged)
    #   diffusers_mps → repo (binary/image_flag inapplicable; image_flag
    #                  defaults to "--image-path" downstream to satisfy
    #                  the Backend dataclass, but isn't consulted)
    engine = data.get("engine", "mflux")
    if engine == "mflux":
        for required in ("binary", "image_flag"):
            if required not in data:
                raise UserBackendError(
                    f"{source}: required field {required!r} missing "
                    f"(engine={engine!r} requires it)"
                )
    elif engine == "diffusers_mps":
        if "repo" not in data:
            raise UserBackendError(
                f"{source}: required field 'repo' missing "
                f"(engine='diffusers_mps' loads a HF Hub repo, not a "
                f"local binary)"
            )
    else:
        # Unknown engine string — caught by the per-field validator
        # in the schema pass below, but raise here for a clearer
        # message that includes the source path. This branch is
        # reached only when ``data.get("engine", "mflux")`` returned
        # a non-string OR an unrecognised string; the schema validator
        # would also reject it.
        raise UserBackendError(
            f"{source}: engine={engine!r} not in "
            "{'mflux', 'diffusers_mps'}"
        )

    # Reject the plural [[secrets]] form explicitly. v0.4 supports at
    # most one secret per backend ([secret] singular); a colleague
    # might reach for the plural-array-of-tables form by reflex,
    # especially when migrating from tools that use it. Without this
    # explicit check the unknown-field branch below would warn-and-
    # drop, leaving the user confused about why their second secret
    # is silently absent at runtime. v0.5 may generalize to multi-
    # secret; THIS rejection is the lock-in that prevents silent
    # data loss until that lands. (v0.4 architect IMP-3 fallback.)
    if "secrets" in data:
        raise UserBackendError(
            f"{source}: [[secrets]] (plural) is not supported — v0.4 "
            "allows at most one secret per backend. Use [secret] "
            "(singular). Multi-secret backends are tracked for v0.5+."
        )

    # Validate every known top-level field (required + optional).
    # The "secret" table is in skip_keys because it gets its own
    # nested validation pass below — without the skip, the top-level
    # loop would emit a spurious "unknown field 'secret' — ignored"
    # warn before the [secret] handler ever runs.
    validated = validate_against_schema(
        data, _USER_BACKEND_SCHEMA, UserBackendError,
        source=str(source), skip_keys={"secret"},
    )

    # Extra binary-content checks beyond the schema predicate.
    # diffusers_mps backends don't carry a binary — skip this check
    # for them. mflux backends always have binary= per the
    # engine-conditional required-field gate above.
    if engine == "mflux":
        _validate_binary_field(validated["binary"], source)

    # v0.8.2 NIT-B closure: warn on fields that pass schema validation
    # but are INAPPLICABLE for the declared engine. A common confusion
    # source: colleague copies an mflux template, switches
    # ``engine = "diffusers_mps"`` + adds ``repo = ...``, but forgets
    # to delete ``binary = ...`` or ``image_flag = ...``. Pre-NIT-B
    # those fields silently flowed into Backend storage and got
    # ignored at runtime — no signal to the user that they were dead.
    # Now we emit a per-field warn at load time naming the engine
    # context. Warns don't reject the file; the inapplicable fields
    # are still stored on Backend (defaults match v0.7) but the user
    # knows to clean them up. Per-engine inapplicability sets are
    # intentionally conservative — only fields that the engine has
    # NO conceivable use for, not "engine probably doesn't care".
    _warn_inapplicable_fields_per_engine(engine, data, source)

    # [secret] section — optional whole-section opt-in.
    secret_env_var: str | None = None
    secret_required: bool = True
    secret_data = data.get("secret")
    if secret_data is not None:
        if not isinstance(secret_data, dict):
            raise UserBackendError(
                f"{source}: [secret] must be a TOML table, "
                f"got {type(secret_data).__name__}"
            )
        if "env_var" not in secret_data:
            raise UserBackendError(
                f"{source}: [secret] table requires env_var = \"...\""
            )
        # Same (desc, predicate) loop pattern as the top-level pass —
        # field_prefix="secret." gets error messages like
        # ``"path: secret.required: expected bool, got 'yes'"``.
        validate_against_schema(
            secret_data, _SECRET_SCHEMA, UserBackendError,
            source=str(source), field_prefix="secret.",
        )
        secret_env_var = secret_data["env_var"]
        if secret_env_var in _DANGEROUS_ENV_VARS:
            raise UserBackendError(
                f"{source}: secret.env_var {secret_env_var!r} is a "
                "dynamic-linker / interpreter override variable that "
                "could change which code runs in the subprocess. "
                "Use an application-specific API key variable instead. "
                "(See _DANGEROUS_ENV_VARS in backends.py for the full "
                "denylist.)"
            )
        secret_required = secret_data.get("required", True)

    # Fill defaults for absent optional fields.
    for key, default in _USER_BACKEND_DEFAULTS.items():
        validated.setdefault(key, default)
    # extra_args is a list[str] in TOML; Backend wants tuple[str, ...].
    extra_args = tuple(validated["extra_args"])

    # v0.5 enhancer fields. Both optional — absent → Backend defaults
    # (None / empty tuple) → enhancer cleanly skips this backend with
    # ``fallback_reason="not_supported_by_backend"``. enhance_invariants
    # is a list in TOML; Backend wants tuple[str, ...].
    enhance_system_prompt = validated.get("enhance_system_prompt")
    enhance_invariants = tuple(validated.get("enhance_invariants", ()))

    # v0.6: LoRA compat-group identifier. Optional — absent → empty
    # string → "no LoRA support for this backend" → any LoRA reference
    # in a style is silently warn-skipped at command-construction time.
    lora_compat_group = validated.get("lora_compat_group", "")

    # v0.7.12 (gap 5): HF gated-repo identifier. Optional — absent →
    # None → Backend default → post-failure hint silently skips. Set
    # to e.g. "briaai/FIBO" or "black-forest-labs/FLUX.1-dev" so the
    # error path surfaces the license-acceptance URL. Intentionally
    # NOT in `_USER_BACKEND_DEFAULTS` — None sentinel is the correct
    # absent value (symmetric with `enhance_system_prompt` above).
    hf_gated_repo = validated.get("hf_gated_repo")

    # v0.8.1 HIGH-2: v0.8 OPTIONAL Model-shape fields. Backend defaults
    # match v0.7 behaviour; user TOMLs that omit them load unchanged.
    # The schema validator above already enforced type/range; here we
    # just pull-with-default and pack into the Backend ctor.
    v08_engine = validated.get("engine", "mflux")
    v08_repo = validated.get("repo")
    # diffusers_mps backends don't ship a binary; downstream code
    # (build_mflux_cmd, doctor) reads Backend.binary unconditionally,
    # so fill an empty-string sentinel here. ``model_from_backend``
    # downstream skips binary for diffusers_mps.
    binary_value = validated.get("binary", "") if v08_engine == "diffusers_mps" \
        else validated["binary"]
    image_flag_value = validated.get("image_flag", "--image-path") \
        if v08_engine == "diffusers_mps" else validated["image_flag"]
    v08_cpu_offload_threshold_mp = validated.get(
        "cpu_offload_threshold_mp", 999.0,
    )
    v08_ram_baseline_gb = validated.get("ram_baseline_gb", 13.5)
    v08_ram_slope_gb_per_mp = validated.get("ram_slope_gb_per_mp", 4.0)
    v08_encoder_ram_gb = validated.get("encoder_ram_gb", 0.0)
    v08_default_steps = validated.get("default_steps", 20)
    v08_default_guidance = validated.get("default_guidance", 3.5)
    v08_min_guidance = validated.get("min_guidance", 0.0)
    v08_max_guidance = validated.get("max_guidance", 10.0)
    # supported_quants: TOML list → tuple. diffusers_mps default empty
    # tuple (omit_quantize is the typical flag) but mflux default
    # mirrors the canonical (3,4,5,6,8) ladder.
    raw_supported_quants = validated.get(
        "supported_quants",
        [] if v08_engine == "diffusers_mps" else [3, 4, 5, 6, 8],
    )
    v08_supported_quants = tuple(raw_supported_quants)
    v08_omit_quantize = validated.get(
        "omit_quantize",
        v08_engine == "diffusers_mps",  # default True for diffusers_mps
    )
    # param_overrides: TOML array-of-pairs → tuple-of-tuples. The schema
    # validator above already enforced shape (each element is a 2-list
    # with str key); convert to immutable form here.
    raw_param_overrides = validated.get("param_overrides", [])
    v08_param_overrides = tuple(
        (pair[0], pair[1]) for pair in raw_param_overrides
    )

    return Backend(
        binary=binary_value,
        needs_token=False,
        image_flag=image_flag_value,
        supports_strength=validated["supports_strength"],
        supports_negative=validated["supports_negative"],
        extra_args=extra_args,
        secret_env_var=secret_env_var,
        secret_required=secret_required,
        enhance_system_prompt=enhance_system_prompt,
        enhance_invariants=enhance_invariants,
        lora_compat_group=lora_compat_group,
        hf_gated_repo=hf_gated_repo,
        # v0.8.1 HIGH-2 fields.
        engine=v08_engine,
        repo=v08_repo,
        cpu_offload_threshold_mp=v08_cpu_offload_threshold_mp,
        ram_baseline_gb=v08_ram_baseline_gb,
        ram_slope_gb_per_mp=v08_ram_slope_gb_per_mp,
        encoder_ram_gb=v08_encoder_ram_gb,
        default_steps=v08_default_steps,
        default_guidance=v08_default_guidance,
        min_guidance=v08_min_guidance,
        max_guidance=v08_max_guidance,
        supported_quants=v08_supported_quants,
        omit_quantize=v08_omit_quantize,
        param_overrides=v08_param_overrides,
    )


# ── Loader (file → Backend) ─────────────────────────────────────────────


def load_user_backend_file(path: Path) -> Backend:
    """Parse + validate one ~/.imgen/backends.d/*.toml file.

    Size-capped + atomically-failed: oversize file or TOML parse error
    raise UserBackendError instead of slurping into memory or failing
    midway through validation. Mirrors load_user_style_file shape.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        raise UserBackendError(f"{path}: {e}") from e
    if size > USER_BACKEND_MAX_BYTES:
        raise UserBackendError(
            f"{path}: too large ({size} bytes; cap "
            f"{USER_BACKEND_MAX_BYTES})"
        )
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise UserBackendError(f"{path}: {e}") from e
    return validate_user_backend_schema(raw, path)


# ── Directory scanner ───────────────────────────────────────────────────


def _is_safe_backend_stem(stem: str) -> bool:
    """Same byte-range filter as styles._is_safe_stem — duplicated per
    v0.4 design (two callers now, three triggers the v0.5 extraction
    to a shared _safe.py module). Reject C0/DEL/C1 in filename stems
    because the stem becomes the backend name and rides into argv,
    --list-backends output, doctor reports, and log files."""
    return not _has_control_bytes(stem)


def load_user_backends_dir(dir_path: Path) -> dict[str, Backend]:
    """Scan a directory for `*.toml`; return {filename_stem: Backend}.

    Alphabetical order so collision-suffix numbering is deterministic.
    A single bad file (parse error, schema violation, unsafe stem)
    warns and continues — never kills the load of the rest.

    Symlinked ``backends.d`` is refused (warn + return empty) — parallel
    to the LOGS_DIR guard in ``runs.py``. The threat is cross-account
    NFS / multi-user-Mac scenarios where another uid could place a
    symlink at ``~/.imgen/backends.d/`` pointing at a directory they
    control, then drop attacker-chosen TOMLs there for imgen to load
    and subprocess-exec. Single-user trust still says STATE_DIR's
    parent is trusted, but the cost of checking is zero and matches
    the precedent. (v0.4 security-reviewer IMP-3.)
    """
    from .colors import warn

    if not dir_path.exists() or not dir_path.is_dir():
        return {}
    if dir_path.is_symlink():
        warn(f"{dir_path} is a symlink; refusing to load user backends. "
             "Remove the symlink and replace with a real directory.")
        return {}
    result: dict[str, Backend] = {}
    for path in sorted(dir_path.iterdir()):
        if path.suffix != ".toml" or not path.is_file():
            continue
        if not _is_safe_backend_stem(path.stem):
            warn(f"Skipping {path.name!r}: control bytes in filename "
                 "(unsafe to use as a backend name)")
            continue
        try:
            result[path.stem] = load_user_backend_file(path)
        except UserBackendError as e:
            warn(f"Skipping {path.name}: {e}")
            continue
    return result


# ── Merge (collision policy: built-ins win, suffix _0001) ───────────────


_SUFFIX_RE = re.compile(r"_\d{4}$")


def _strip_auto_suffix(name: str) -> str:
    """Drop a trailing `_NNNN` so re-suffixing produces clean `flux_0002`
    rather than `flux_0001_0001`. Duplicate of styles._strip_auto_suffix
    (per v0.4 design — extract in v0.5 when a 3rd surface appears)."""
    return _SUFFIX_RE.sub("", name)


def _find_free_suffix(base: str, taken: dict) -> str:
    """Smallest N >= 1 such that ``f"{base}_{N:04d}"`` is unused."""
    n = 1
    while f"{base}_{n:04d}" in taken:
        n += 1
    return f"{base}_{n:04d}"


def merge_user_backends(
    builtins: dict[str, Backend],
    user: dict[str, Backend],
) -> dict[str, Backend]:
    """Combine built-in backends with user backends. Built-ins win.

    A user backend whose name clashes with a built-in (or an earlier
    user file alphabetically) gets renamed `<name>_NNNN` (4-digit zero-
    padded counter). The original entry stays accessible under its
    original name. Same semantics as styles.merge_user_styles.

    Does NOT mutate either input.
    """
    from .colors import warn

    merged: dict[str, Backend] = dict(builtins)
    for name, backend in user.items():
        if name not in merged:
            merged[name] = backend
            continue
        base = _strip_auto_suffix(name)
        new_name = _find_free_suffix(base, merged)
        warn(
            f"backends.d: '{name}' already taken (built-in or earlier "
            f"user file), registered as '{new_name}'"
        )
        merged[new_name] = backend
    return merged


# ── Public accessors (cached merge of built-ins + user backends) ────────


_cached_merged: dict[str, Backend] | None = None


def _load_merged_backends() -> dict[str, Backend]:
    """Lazy-merge built-ins + ~/.imgen/backends.d/ + ~/.imgen/models.d/.

    v0.8.0 commit 3: ``models.d/`` is the new canonical user-TOML path
    (per [[project-v080-design]] §H). Reads BOTH directories during the
    v0.8.x deprecation window:

    * ``models.d/`` overlays ``backends.d/`` for same-stem TOMLs:
      ``{**from_backends_d, **from_models_d}`` lets a colleague drop
      the new copy in ``models.d/`` and see it win immediately without
      first deleting the old one.
    * Same-stem collisions BETWEEN user TOMLs and built-ins still go
      through ``merge_user_backends`` (built-ins win, user gets the
      ``_NNNN`` suffix).
    * Symlinked directories on either side are refused with a warn —
      protection mirrored from v0.4 IMP-3 for ``backends.d/``.

    v0.8.0 commit 4a: per-file DEPRECATED warn for ``backends.d/``
    entries. The migration window stays open through v0.8.x; v0.9.0
    drops the ``backends.d/`` read entirely. Warn message points at
    the concrete ``mv`` command so a colleague can fix it without
    consulting docs.

    Cache means the warn fires ONCE per process — the ``_cached_merged``
    short-circuit avoids re-firing on every ``get_backend()`` or
    ``list_backends()`` call. This is the intended UX (one migration
    nudge per CLI invocation, not one per call site). Tests reset the
    cache between cases via ``reset_backends_cache()`` so each test
    sees the warn fresh.
    """
    global _cached_merged
    if _cached_merged is None:
        from .colors import warn as _warn
        # Local import to avoid module-load circularity with paths.py.
        from .paths import BACKENDS_D, MODELS_D
        from_backends_d = load_user_backends_dir(BACKENDS_D)
        from_models_d = load_user_backends_dir(MODELS_D)
        # v0.8.0 commit 4a: DEPRECATED warn per file in legacy dir.
        # Lives at the orchestration layer (not inside
        # ``load_user_backends_dir``) so the loader stays a pure
        # "read this directory, return dict" helper. Test:
        # tests/test_user_models.py::test_user_toml_warns_on_backends_d_load
        for name in sorted(from_backends_d):
            _warn(
                f"~/.imgen/backends.d/{name}.toml: DEPRECATED — "
                f"v0.8.0 renamed this directory to ~/.imgen/models.d/. "
                f"Run `mv ~/.imgen/backends.d/{name}.toml "
                f"~/.imgen/models.d/{name}.toml`. "
                f"backends.d/ read removed in v0.9.0."
            )
        # `**` overlay — models.d entries replace same-stem backends.d
        # entries before the built-in collision pass. Test:
        # tests/test_user_models.py::test_user_toml_models_d_wins_on_collision
        user = {**from_backends_d, **from_models_d}
        _cached_merged = merge_user_backends(BUILTIN_BACKENDS, user)
    return _cached_merged


def list_backends() -> list[str]:
    """Sorted list of available backend names (built-in + user)."""
    return sorted(_load_merged_backends().keys())


def get_backend(name: str) -> Backend:
    """Return Backend by name. v0.8.0 commit 4b: back-compat shim
    accepting BOTH v0.7 names (``flux``, ``qwen``) and v0.8 canonical
    names (``flux-kontext``, ``qwen-image-edit-v1``). The v0.7 →
    v0.8-key translation lives here so the alias surface is
    single-source (architect 4b pre-vet M-1): ``models.get_model()``
    is strict v0.8-only; this shim handles both.

    Internally the merged registry is v0.7-keyed (BUILTIN_BACKENDS
    derived backward + user TOML stems by filename). A v0.8 input
    is mapped back to its v0.7 key via the inverse rename map before
    lookup.

    Raises KeyError if unknown after both translation attempts.
    """
    merged = _load_merged_backends()
    # First try direct lookup — covers v0.7 names + user TOML stems.
    if name in merged:
        return merged[name]
    # Then try v0.8 → v0.7 translation: if user passed `flux-kontext`,
    # registry has it under `flux`.
    v08_to_v07: dict[str, str] = {
        v08: v07 for v07, v08 in _V07_TO_V08_MODEL_RENAMES.items()
    }
    canonical = v08_to_v07.get(name)
    if canonical is not None and canonical in merged:
        return merged[canonical]
    available = ", ".join(sorted(merged.keys()))
    raise KeyError(
        f"Unknown backend '{name}'. Available: {available}"
    )


def reset_backends_cache() -> None:
    """Wipe ``_cached_merged`` so the next ``_load_merged_backends``
    re-reads ~/.imgen/backends.d/. Tests use this between fixtures;
    not part of the public Backend surface."""
    global _cached_merged
    _cached_merged = None


def filter_compatible_loras(
    loras: tuple,  # tuple[styles.LoraRef, ...] — avoid circular import
    backend: Backend,
) -> tuple[tuple, tuple]:
    """Split ``loras`` into (compatible, incompatible) tuples based on
    ``backend.lora_compat_group``. Each LoraRef's ``compatible_with``
    field is a tuple of group identifiers; a LoRA is compatible iff
    that tuple contains the backend's group.

    Empty ``backend.lora_compat_group`` (the default for any backend
    that hasn't opted into LoRA support) means "no LoRA support" → all
    LoRAs land in the incompatible bucket. The caller (typically
    ``build_mflux_cmd``) emits a warn for each incompatible entry
    rather than silently dropping them — silent drops would make
    a user with three LoRAs in their style file wonder why only one
    fired (e.g. a Qwen run with a mix of FLUX and Qwen LoRAs).

    Pure: no I/O, no warnings. ``build_mflux_cmd`` does the warn
    emission so this stays unit-testable without colors / stderr.
    """
    group = backend.lora_compat_group
    if not group:
        return ((), tuple(loras))
    compatible = []
    incompatible = []
    for lora in loras:
        if group in lora.compatible_with:
            compatible.append(lora)
        else:
            incompatible.append(lora)
    return (tuple(compatible), tuple(incompatible))


def build_mflux_cmd(
    *,
    binary: Path,
    model: Backend,
    input_path: Path | None,
    output_path: Path,
    prompt: str,
    negative: str,
    quantize: int,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
    width: int,
    height: int,
    mlx_cache_gb: int,
    battery_stop: int,
    loras: tuple = (),  # tuple[styles.LoraRef, ...]
) -> list[str]:
    """Build the mflux argv for `model` from already-resolved parameters.

    Pure: no I/O, no env reads, no subprocess. Keyword-only because 16
    positional args would be a footgun.

    v0.8.0 commit 4b: kwarg renamed ``backend=`` → ``model=`` in
    lockstep with the registry source-of-truth flip. The function still
    accepts a ``Backend`` instance (v0.7-shape v0.8-derived view) at
    that slot; future Engine-layer migration will tighten this to
    ``Model`` after callers move to the new dispatch path.

    Order preserved from v0.1.x: common args first, then strength (if
    supported), then `extra_args` (e.g. `--model dev`), then negative
    prompt (if supported and non-empty). v0.6 appends ``--lora-paths``
    + ``--lora-scales`` AFTER ``extra_args`` so mflux's ``--model``
    selection happens before LoRA application — same order CLI users
    typically write by hand.

    ``loras`` is a tuple of :class:`styles.LoraRef`. Only entries whose
    ``compatible_with`` includes ``model.lora_compat_group`` are
    applied; incompatibles emit a warn (visible in ``--dry-run`` output
    + interactive runs) explaining the mismatch. Empty tuple (default)
    → no LoRA argv emitted, identical to v0.5 behaviour.

    ``input_path`` (v0.7.0): ``Path | None`` — None for t2i (``imgen
    draw``) where there's no source photo. When None, the
    ``backend.image_flag <path>`` argv pair is omitted entirely.
    Backends supporting t2i (``flux-dev``) still declare
    ``image_flag="--image-path"`` for dataclass-shape consistency;
    the runtime gate is here, not on the dataclass field.

    v0.3.2: dropped ``--metadata``. mflux's ``--metadata`` writes a
    ``<output>.metadata.json`` sidecar next to every generated image,
    which clutters the user's gallery folder. The PNG itself still
    gets metadata embedded via mflux's ``_embed_metadata`` +
    ``MetadataBuilder.embed_metadata`` calls (these fire whenever a
    ``metadata`` dict exists, independent of the ``--metadata`` flag);
    the sidecar JSON was duplicate data. We also already store every
    run's params in ``~/.imgen/history.jsonl`` for replay, making the
    sidecars triply redundant.
    """
    # v0.8.0 commit 7 (§M): respect ``model.omit_quantize=True`` for
    # prequantized model repos (e.g. mlx-community/Qwen-Image-2512-4bit
    # — weights are already int4-packed; mflux's --quantize 4 against
    # them no-ops, but the contract is undocumented). Skipping the
    # flag emission makes the contract explicit AT THE MODEL LEVEL,
    # not at the per-binary cmd_* level. Built-ins at commit 7 ship
    # with omit_quantize=False (default); user TOMLs gain the v0.8
    # schema field at commit 6+ once the loader extension lands.
    cmd = [str(binary)]
    if not getattr(model, "omit_quantize", False):
        cmd += ["--quantize", str(quantize)]
    if input_path is not None:
        cmd += [model.image_flag, str(input_path)]
    cmd += [
        "--prompt", prompt,
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(seed),
        "--width", str(width),
        "--height", str(height),
        "--mlx-cache-limit-gb", str(mlx_cache_gb),
        "--battery-percentage-stop-limit", str(battery_stop),
        "--output", str(output_path),
    ]
    if model.supports_strength:
        cmd += ["--image-strength", str(strength)]
    cmd += list(model.extra_args)
    if model.supports_negative and negative:
        cmd += ["--negative-prompt", negative]

    # v0.6: LoRA argv emission. Compatible entries land as parallel
    # --lora-paths + --lora-scales lists (mflux accepts space-separated
    # multi-value args for both). Pure: this function does NOT warn —
    # warn emission is hoisted to ``build_iterations`` so N×M batch runs
    # don't spam 150 identical warns on the same incompatible CLI LoRA.
    # (v0.6.x backlog python IMP-3.)
    if loras:
        compatible, _incompatible = filter_compatible_loras(loras, model)
        if compatible:
            cmd += ["--lora-paths", *(lora.ref for lora in compatible)]
            cmd += ["--lora-scales", *(str(lora.weight) for lora in compatible)]
    return cmd

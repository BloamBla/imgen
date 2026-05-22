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

from ._schema import validate_against_schema

__all__ = [
    "BACKENDS",
    "BUILTIN_BACKENDS",
    "USER_BACKEND_MAX_BYTES",
    "Backend",
    "UserBackendError",
    "build_mflux_cmd",
    "get_backend",
    "list_backends",
    "load_user_backend_file",
    "load_user_backends_dir",
    "merge_user_backends",
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


# System prompts for built-in backends. Module-level constants so tests
# can reference the exact text and import-time look at Backend tuples
# stays terse. Tuned per backend conventions (see
# project_v050_v060_design.md, "System prompts per backend").
#
# The CRITICAL section in each system prompt is the identity-anchor
# preservation directive. v0.3.4 BFL Kontext tuning baked three specific
# preservation phrases into our styles.py (one per style family) — the
# enhancer must NOT substitute them for synonyms or alternative wording.
# Phase C-1 smoke (2026-05-22) caught Qwen2.5-7B-4bit silently rewriting
# "preserving the facial identity, hairstyle, body proportions, and pose"
# to "preserving the overall composition and the relative position of
# all subjects" — substring "preserving" survived but the semantic
# identity-anchor was discarded. The fix is two-layer: (1) tighter
# system prompt forbidding anchor substitution; (2) multi-substring
# invariants below that fall back per-style when the specific anchor
# is missing from the LLM output.
_FLUX_KONTEXT_ENHANCE_SYS = (
    "You expand image-editing prompts for FLUX.1 Kontext, an image-"
    "conditioning model that restyles input photos while preserving "
    "identity, pose, and composition. Take the user prompt and expand "
    "it to 40-60 tokens. "
    "CRITICAL: you MUST preserve the entire 'while preserving …' "
    "clause from the user prompt VERBATIM. Keep every word inside that "
    "clause exactly as written — particularly identity anchors such as "
    "'facial identity', 'exact facial features', or 'recognizable "
    "expression'. Do NOT replace these anchors with synonyms or "
    "alternative preservation language (e.g. NEVER substitute 'overall "
    "composition' or 'relative position of subjects' for the identity "
    "anchor). "
    "Add specific stylistic descriptors (lighting, color palette, art "
    "technique, materials) at the START or END, not inside the "
    "preserving clause. Do NOT invent objects, scenes, or characters "
    "not in the user prompt — expand existing details only. NEVER "
    "describe the input photo's content — Kontext sees it directly. "
    "Output ONLY the expanded prompt with no preamble, no quotes, "
    "no explanation."
)

_QWEN_EDIT_ENHANCE_SYS = (
    "You expand instruction-style edit prompts for Qwen-Image-Edit. "
    "Use imperative verbs ('transform', 'restyle', 'apply'). Keep the "
    "output under 40 tokens — Qwen-Edit prefers shorter directives "
    "than FLUX. "
    "CRITICAL: preserve the entire 'while preserving …' clause from "
    "the user prompt VERBATIM, including identity anchors like "
    "'facial identity', 'exact facial features', or 'recognizable "
    "expression'. Do NOT swap these for synonyms. "
    "Do NOT invent objects, scenes, or characters not in the user "
    "prompt — expand existing details only. NEVER describe the input "
    "photo's content. Output ONLY the expanded prompt with no preamble, "
    "no quotes, no explanation."
)

# Multi-substring invariant: each entry is a per-style-family identity
# anchor. ``check_invariants`` enforces an invariant ONLY when it
# appears in the original prompt, so the three anchors don't compete —
# whichever one the style chose, that one gets enforced.
#
# Coverage matrix (v0.5 ship, see styles.py):
#   "facial identity"          → pixar, anime, ghibli
#   "exact facial features"    → vangogh, pencil
#   "recognizable expression"  → simpsons
#
# User-defined styles in ``~/.imgen/styles.d/*.toml`` that don't use
# any of these anchors get no enhanced protection — they fall through
# the invariant gate (no anchor in original = no enforcement). That's
# a known v0.5 limitation; tightening user-side anchors is a v0.6+
# extension once we've validated the built-in path.
_IDENTITY_ANCHOR_INVARIANTS: tuple[str, ...] = (
    "facial identity",
    "exact facial features",
    "recognizable expression",
)


BUILTIN_BACKENDS: dict[str, Backend] = {
    "flux": Backend(
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
        enhance_system_prompt=_FLUX_KONTEXT_ENHANCE_SYS,
        enhance_invariants=_IDENTITY_ANCHOR_INVARIANTS,
    ),
    "qwen": Backend(
        binary="mflux-generate-qwen-edit",
        needs_token=False,
        image_flag="--image-paths",
        supports_strength=False,
        supports_negative=False,
        extra_args=("--model", "qwen"),
        enhance_system_prompt=_QWEN_EDIT_ENHANCE_SYS,
        enhance_invariants=_IDENTITY_ANCHOR_INVARIANTS,
    ),
}

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


def _has_control_bytes(s: str) -> bool:
    """C0 / DEL / C1 byte detector. Mirrors styles._is_safe_stem inverted
    so backends.py can reject the same byte ranges in user-supplied
    fields (binary path, env var name) without cross-module
    underscore-imports. Tiny enough to duplicate per v0.4 design
    decision (extract to a shared _safe.py module in v0.5 when a 3rd
    surface appears)."""
    return any(
        c < ' ' or c == '\x7f' or '\x80' <= c <= '\x9f'
        for c in s
    )


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


def validate_user_backend_schema(data: dict, source: Path) -> Backend:
    """Turn a parsed TOML dict into a Backend instance, enforcing schema.

    Unknown top-level fields warn and are dropped (forward-compat with
    future additions). Known fields with bad types/values raise
    UserBackendError carrying the source path for diagnostics.

    The ``[secret]`` section is optional. If present, ``env_var`` must
    be set; ``required`` defaults to True.

    Mirrors styles.validate_user_style_schema / load_user_style_file
    shape. Future ``_schema.py::validate_against_schema`` extraction
    (v0.2 backlog item, three-callers trigger now) is deferred to v0.5.
    """
    from .colors import warn

    # Required fields.
    for required in ("binary", "image_flag"):
        if required not in data:
            raise UserBackendError(
                f"{source}: required field {required!r} missing"
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
    _validate_binary_field(validated["binary"], source)

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

    return Backend(
        binary=validated["binary"],
        needs_token=False,
        image_flag=validated["image_flag"],
        supports_strength=validated["supports_strength"],
        supports_negative=validated["supports_negative"],
        extra_args=extra_args,
        secret_env_var=secret_env_var,
        secret_required=secret_required,
        enhance_system_prompt=enhance_system_prompt,
        enhance_invariants=enhance_invariants,
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
    """Lazy-merge built-ins + ~/.imgen/backends.d/. Cached per process."""
    global _cached_merged
    if _cached_merged is None:
        # Local import to avoid module-load circularity with paths.py.
        from .paths import BACKENDS_D
        _cached_merged = merge_user_backends(
            BUILTIN_BACKENDS, load_user_backends_dir(BACKENDS_D)
        )
    return _cached_merged


def list_backends() -> list[str]:
    """Sorted list of available backend names (built-in + user)."""
    return sorted(_load_merged_backends().keys())


def get_backend(name: str) -> Backend:
    """Return Backend by name. Raises KeyError if unknown."""
    merged = _load_merged_backends()
    if name not in merged:
        available = ", ".join(sorted(merged.keys()))
        raise KeyError(
            f"Unknown backend '{name}'. Available: {available}"
        )
    return merged[name]


def reset_backends_cache() -> None:
    """Wipe ``_cached_merged`` so the next ``_load_merged_backends``
    re-reads ~/.imgen/backends.d/. Tests use this between fixtures;
    not part of the public Backend surface."""
    global _cached_merged
    _cached_merged = None


def build_mflux_cmd(
    *,
    binary: Path,
    backend: Backend,
    input_path: Path,
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
) -> list[str]:
    """Build the mflux argv for `backend` from already-resolved parameters.

    Pure: no I/O, no env reads, no subprocess. Keyword-only because 15
    positional args would be a footgun.

    Order preserved from v0.1.x: common args first, then strength (if
    supported), then `extra_args` (e.g. `--model dev`), then negative
    prompt (if supported and non-empty). Locked in by test_generate_cmd.

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
    cmd = [
        str(binary),
        "--quantize", str(quantize),
        backend.image_flag, str(input_path),
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
    if backend.supports_strength:
        cmd += ["--image-strength", str(strength)]
    cmd += list(backend.extra_args)
    if backend.supports_negative and negative:
        cmd += ["--negative-prompt", negative]
    return cmd

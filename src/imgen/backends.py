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


BUILTIN_BACKENDS: dict[str, Backend] = {
    "flux": Backend(
        binary="mflux-generate-kontext",
        needs_token=True,
        image_flag="--image-path",
        supports_strength=True,
        supports_negative=True,
        extra_args=("--model", "dev"),
    ),
    "qwen": Backend(
        binary="mflux-generate-qwen-edit",
        needs_token=False,
        image_flag="--image-paths",
        supports_strength=False,
        supports_negative=False,
        extra_args=("--model", "qwen"),
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


def _is_list_of_str(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


_USER_BACKEND_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "binary": ("non-empty string", _is_str_nonempty),
    "image_flag": (
        f"one of {sorted(_IMAGE_FLAG_CHOICES)}",
        lambda v: isinstance(v, str) and v in _IMAGE_FLAG_CHOICES,
    ),
    "supports_strength": ("bool", lambda v: isinstance(v, bool)),
    "supports_negative": ("bool", lambda v: isinstance(v, bool)),
    "extra_args": ("list of strings", _is_list_of_str),
}

_SECRET_SCHEMA: dict[str, tuple[str, Callable[[Any], bool]]] = {
    "env_var": (
        "non-empty string matching ^[A-Za-z_][A-Za-z0-9_]*$",
        lambda v: isinstance(v, str) and _ENV_VAR_NAME_RE.match(v) is not None,
    ),
    "required": ("bool", lambda v: isinstance(v, bool)),
}

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
        if not Path(value).exists():
            raise UserBackendError(
                f"{source}: binary {value!r} doesn't exist on disk"
            )


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

    # Validate every known top-level field (required + optional).
    validated: dict[str, Any] = {}
    for key, value in data.items():
        if key == "secret":
            # Handled separately below.
            continue
        if key not in _USER_BACKEND_SCHEMA:
            warn(f"{source}: unknown field '{key}' — ignored")
            continue
        expected_desc, predicate = _USER_BACKEND_SCHEMA[key]
        if not predicate(value):
            raise UserBackendError(
                f"{source}: {key}: expected {expected_desc}, got {value!r}"
            )
        validated[key] = value

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
        for skey, svalue in secret_data.items():
            if skey not in _SECRET_SCHEMA:
                warn(f"{source}: [secret] unknown field '{skey}' — ignored")
                continue
            expected_desc, predicate = _SECRET_SCHEMA[skey]
            if not predicate(svalue):
                raise UserBackendError(
                    f"{source}: secret.{skey}: expected {expected_desc}, "
                    f"got {svalue!r}"
                )
        secret_env_var = secret_data["env_var"]
        secret_required = secret_data.get("required", True)

    # Fill defaults for absent optional fields.
    for key, default in _USER_BACKEND_DEFAULTS.items():
        validated.setdefault(key, default)
    # extra_args is a list[str] in TOML; Backend wants tuple[str, ...].
    extra_args = tuple(validated["extra_args"])

    return Backend(
        binary=validated["binary"],
        needs_token=False,
        image_flag=validated["image_flag"],
        supports_strength=validated["supports_strength"],
        supports_negative=validated["supports_negative"],
        extra_args=extra_args,
        secret_env_var=secret_env_var,
        secret_required=secret_required,
    )


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

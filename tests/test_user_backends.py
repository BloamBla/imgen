"""User backend TOMLs in ~/.imgen/backends.d/ — schema validation +
loader + merge semantics.

Mirrors test_user_styles.py shape. The schema validator is the heart of
the v0.4 registry: it's the trust boundary between "user TOML" and
"subprocess argv we'll exec", so adversarial / typo / type-confusion
inputs all get covered here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from imgen.backends import (
    UserBackendError,
    validate_user_backend_schema,
)


# ── Required field validation ───────────────────────────────────────────


def test_validator_rejects_missing_binary():
    data = {"image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="binary"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_missing_image_flag():
    data = {"binary": "mflux-generate-x"}
    with pytest.raises(UserBackendError, match="image_flag"):
        validate_user_backend_schema(data, Path("test.toml"))


# ── Field type / value validation ───────────────────────────────────────


def test_validator_rejects_non_string_binary():
    data = {"binary": 42, "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="binary.*non-empty string"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_empty_binary():
    data = {"binary": "   ", "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="binary.*non-empty string"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_unknown_image_flag():
    """Only --image-path and --image-paths are recognized — these map to
    mflux's two binaries. A typo or made-up flag is caught at parse
    time so the user finds out before subprocess launch."""
    data = {"binary": "x", "image_flag": "--input"}
    with pytest.raises(UserBackendError, match="image_flag"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_non_bool_supports_strength():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "supports_strength": "yes",
    }
    with pytest.raises(UserBackendError, match="supports_strength.*bool"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_extra_args_with_non_string_element():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "extra_args": ["--model", 42],
    }
    with pytest.raises(UserBackendError, match="extra_args.*list of strings"):
        validate_user_backend_schema(data, Path("test.toml"))


# ── Binary content validation (paths + control bytes) ───────────────────


def test_validator_rejects_relative_binary_path():
    """Relative paths are ambiguous (CWD at exec ≠ CWD at parse).
    User must pick: bare name (PATH lookup) or absolute path."""
    data = {"binary": "./bin/foo", "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="relative path"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_parent_relative_binary_path():
    data = {"binary": "../bin/foo", "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="relative path"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_nested_relative_binary_path():
    data = {"binary": "sub/dir/foo", "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="relative path"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_nonexistent_absolute_binary():
    data = {
        "binary": "/totally/not/a/real/path/binary",
        "image_flag": "--image-path",
    }
    with pytest.raises(UserBackendError, match="doesn't exist"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_accepts_existing_absolute_binary(tmp_path):
    binary = tmp_path / "fake-mflux"
    binary.touch()
    data = {"binary": str(binary), "image_flag": "--image-path"}
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.binary == str(binary)


def test_validator_rejects_control_bytes_in_binary():
    """ESC / DEL / C1 controls in binary name would leak into argv +
    logs. Same defence as _is_safe_stem on style filenames."""
    data = {"binary": "evil\x1b[2J", "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="control bytes"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_accepts_bare_name():
    """The common case: name resolvable via $PATH at exec time."""
    data = {"binary": "mflux-generate-sdxl", "image_flag": "--image-path"}
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.binary == "mflux-generate-sdxl"


# ── Defaults for absent optional fields ─────────────────────────────────


def test_validator_fills_defaults_for_optional_fields():
    """Minimum-shape TOML (just binary + image_flag) gets the documented
    defaults for the rest of the Backend dataclass."""
    data = {"binary": "x", "image_flag": "--image-path"}
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.supports_strength is False
    assert be.supports_negative is False
    assert be.extra_args == ()
    assert be.needs_token is False
    assert be.secret_env_var is None
    assert be.secret_required is True


def test_validator_converts_extra_args_list_to_tuple():
    """Backend.extra_args is tuple[str, ...] (frozen-dataclass-friendly).
    TOML deserializes arrays into Python list, so the validator converts."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "extra_args": ["--model", "sdxl", "--cfg-rescale", "0.6"],
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.extra_args == ("--model", "sdxl", "--cfg-rescale", "0.6")
    assert isinstance(be.extra_args, tuple)


# ── Unknown fields warn but don't fail ──────────────────────────────────


def test_validator_warns_on_unknown_top_level_field(capsys):
    """Forward-compat: a new field we don't know about should be ignored
    with a warn, not kill the load. Same pattern as styles.d."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "future_field": "whatever",
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.binary == "x"
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "future_field" in combined
    assert "ignored" in combined


# ── [secret] section ────────────────────────────────────────────────────


def test_validator_accepts_secret_section():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"env_var": "MY_API_KEY"},
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.secret_env_var == "MY_API_KEY"
    assert be.secret_required is True  # default


def test_validator_accepts_secret_with_required_false():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"env_var": "OPTIONAL_KEY", "required": False},
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.secret_env_var == "OPTIONAL_KEY"
    assert be.secret_required is False


def test_validator_rejects_secret_without_env_var():
    """An [secret] section with no env_var is malformed — user probably
    intended one but forgot. Raise so they fix it."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"required": True},
    }
    with pytest.raises(UserBackendError, match="env_var"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_secret_not_a_table():
    """[secret] must be a table; `secret = "foo"` is wrong shape."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": "FOO_TOKEN",
    }
    with pytest.raises(UserBackendError, match="TOML table"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_invalid_env_var_name():
    """Env var name must match ^[A-Za-z_][A-Za-z0-9_]*$ — embedded
    spaces, hyphens, or starting digits aren't standard POSIX."""
    bad_names = ["FOO-BAR", "1FOO", "FOO BAR", "FOO$BAR", ""]
    for name in bad_names:
        data = {
            "binary": "x", "image_flag": "--image-path",
            "secret": {"env_var": name},
        }
        with pytest.raises(UserBackendError, match="env_var"):
            validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_non_bool_required():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"env_var": "FOO", "required": "yes"},
    }
    with pytest.raises(UserBackendError, match="required"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_warns_on_unknown_secret_field(capsys):
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"env_var": "FOO", "future_field": "x"},
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.secret_env_var == "FOO"
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "future_field" in combined


# ── Custom backends never inherit needs_token ───────────────────────────


def test_user_backends_always_have_needs_token_false():
    """Custom backends use the new secret schema, not the legacy
    needs_token + ~/.imgen/hf_token path. Validator must hard-code this
    so a user can't accidentally claim needs_token=true and confuse
    the HF token plumbing (which is FLUX-specific)."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        # Even if the user puts `needs_token = true` in their TOML, the
        # validator's _USER_BACKEND_SCHEMA doesn't have that field —
        # it warns + drops it, and the resulting Backend has the
        # default False.
        "needs_token": True,
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.needs_token is False

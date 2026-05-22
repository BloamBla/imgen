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
    with pytest.raises(UserBackendError, match="not a regular file"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_directory_as_binary(tmp_path):
    """v0.4 python-reviewer IMP-1: `binary = "/usr/local/bin"` (a dir)
    used to pass `.exists()` and crash at subprocess.Popen with
    IsADirectoryError. is_file() rejects it at schema time."""
    a_dir = tmp_path / "iam-a-directory"
    a_dir.mkdir()
    data = {"binary": str(a_dir), "image_flag": "--image-path"}
    with pytest.raises(UserBackendError, match="not a regular file"):
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


# ── Denylist for dynamic-linker / interpreter override env vars ─────────


@pytest.mark.parametrize("dangerous", [
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FORCE_FLAT_NAMESPACE",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONHOME",
])
def test_validator_rejects_dangerous_env_var_names(dangerous):
    """v0.4 security-reviewer IMP-1: a forum-distributed sdxl.toml
    declaring secret.env_var = "LD_PRELOAD" used to be accepted by
    the schema (matches the POSIX-name regex) and would forward
    whatever LD_PRELOAD the user has set in their shell into the
    subprocess env — bypassing the _MFLUX_ENV_ALLOWLIST. Reject at
    schema time so the exploit shape is closed regardless of
    whether the user's shell has the variable set."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "secret": {"env_var": dangerous},
    }
    with pytest.raises(UserBackendError, match="dynamic-linker"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_accepts_normal_api_key_env_var_names():
    """Sanity: the denylist doesn't over-match. Common API-key env
    var shapes still validate cleanly."""
    for env_var in [
        "REPLICATE_API_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "HF_TOKEN", "MY_CUSTOM_KEY", "BACKEND_SECRET_2",
    ]:
        data = {
            "binary": "x", "image_flag": "--image-path",
            "secret": {"env_var": env_var},
        }
        be = validate_user_backend_schema(data, Path("test.toml"))
        assert be.secret_env_var == env_var


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


# ── load_user_backend_file (TOML I/O + size cap + parse errors) ─────────


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_load_user_backend_file_happy_path(tmp_path):
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "sdxl.toml"
    _write_toml(toml_path, '''
binary = "mflux-generate-sdxl"
image_flag = "--image-path"
supports_strength = true
extra_args = ["--model", "sdxl"]
''')
    be = load_user_backend_file(toml_path)
    assert be.binary == "mflux-generate-sdxl"
    assert be.supports_strength is True
    assert be.extra_args == ("--model", "sdxl")


def test_load_user_backend_file_rejects_oversized(tmp_path):
    """Real backend TOMLs are tiny (hundreds of bytes). A multi-MB
    file is corruption or misuse — refuse rather than slurp."""
    from imgen.backends import USER_BACKEND_MAX_BYTES, load_user_backend_file
    toml_path = tmp_path / "huge.toml"
    # 1 byte over cap.
    toml_path.write_text("x = " + ("y" * (USER_BACKEND_MAX_BYTES + 1)))
    with pytest.raises(UserBackendError, match="too large"):
        load_user_backend_file(toml_path)


def test_load_user_backend_file_rejects_malformed_toml(tmp_path):
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "broken.toml"
    toml_path.write_text("binary = \"x\nimage_flag = \"--image-path\"")
    with pytest.raises(UserBackendError):
        load_user_backend_file(toml_path)


def test_load_user_backend_file_rejects_missing_file(tmp_path):
    from imgen.backends import load_user_backend_file
    with pytest.raises(UserBackendError):
        load_user_backend_file(tmp_path / "ghost.toml")


# ── load_user_backends_dir (directory iteration + warn-skip on errors) ──


def test_load_dir_returns_empty_when_nonexistent(tmp_path):
    from imgen.backends import load_user_backends_dir
    assert load_user_backends_dir(tmp_path / "ghost") == {}


def test_load_dir_returns_empty_when_path_is_file(tmp_path):
    from imgen.backends import load_user_backends_dir
    fake_file = tmp_path / "not_a_dir"
    fake_file.touch()
    assert load_user_backends_dir(fake_file) == {}


def test_load_dir_refuses_when_backends_dir_itself_is_symlink(
    tmp_path, capsys
):
    """v0.4 security-reviewer IMP-3: a symlinked backends.d/ (cross-
    account NFS scenario) would let imgen load TOMLs from an attacker-
    controlled directory. Mirror the LOGS_DIR symlink defence."""
    from imgen.backends import load_user_backends_dir
    target = tmp_path / "attacker_dir"
    target.mkdir()
    # Drop a TOML in the target — if the guard fails, imgen would
    # load it.
    (target / "evil.toml").write_text(
        'binary = "/bin/sh"\nimage_flag = "--image-path"\n'
    )
    backends_link = tmp_path / "backends.d"
    backends_link.symlink_to(target)

    result = load_user_backends_dir(backends_link)
    # Refusal: no entries returned, attacker's TOML ignored.
    assert result == {}
    combined = capsys.readouterr()
    assert "symlink" in (combined.out + combined.err)


def test_load_dir_loads_multiple_tomls(tmp_path):
    from imgen.backends import load_user_backends_dir
    _write_toml(tmp_path / "a.toml",
                'binary = "a-bin"\nimage_flag = "--image-path"\n')
    _write_toml(tmp_path / "b.toml",
                'binary = "b-bin"\nimage_flag = "--image-paths"\n')
    result = load_user_backends_dir(tmp_path)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"].binary == "a-bin"
    assert result["b"].binary == "b-bin"


def test_load_dir_skips_non_toml_files(tmp_path):
    from imgen.backends import load_user_backends_dir
    _write_toml(tmp_path / "real.toml",
                'binary = "x"\nimage_flag = "--image-path"\n')
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "README").write_text("also ignore")
    result = load_user_backends_dir(tmp_path)
    assert list(result.keys()) == ["real"]


def test_load_dir_warns_and_skips_malformed_file(tmp_path, capsys):
    from imgen.backends import load_user_backends_dir
    _write_toml(tmp_path / "good.toml",
                'binary = "x"\nimage_flag = "--image-path"\n')
    (tmp_path / "bad.toml").write_text("totally not toml \"")
    result = load_user_backends_dir(tmp_path)
    # Bad file skipped, good file survived.
    assert list(result.keys()) == ["good"]
    combined = capsys.readouterr()
    combined_str = combined.out + combined.err
    assert "Skipping bad.toml" in combined_str


def test_load_dir_processes_files_in_sorted_order(tmp_path):
    """Determinism for collision suffixing depends on alphabetical
    iteration — lock it. Adversarial filenames that exercise the sort
    boundary."""
    from imgen.backends import load_user_backends_dir
    _write_toml(tmp_path / "z.toml",
                'binary = "z-bin"\nimage_flag = "--image-path"\n')
    _write_toml(tmp_path / "a.toml",
                'binary = "a-bin"\nimage_flag = "--image-path"\n')
    result = load_user_backends_dir(tmp_path)
    # dict preserves insertion order in modern Python; first key is alphabetic-first
    keys = list(result.keys())
    assert keys == ["a", "z"]


# ── merge_user_backends (collision policy: built-ins win, suffix) ───────


def _bare_backend(binary: str = "x"):
    from imgen.backends import Backend
    return Backend(
        binary=binary, needs_token=False, image_flag="--image-path",
        supports_strength=False, supports_negative=False, extra_args=(),
    )


def test_merge_adds_new_user_backends():
    from imgen.backends import BUILTIN_BACKENDS, merge_user_backends
    user = {"sdxl": _bare_backend("sdxl-bin")}
    merged = merge_user_backends(BUILTIN_BACKENDS, user)
    assert set(merged.keys()) == {"flux", "qwen", "sdxl"}
    assert merged["sdxl"].binary == "sdxl-bin"


def test_merge_collision_with_builtin_gets_suffix(capsys):
    """User TOML named after a built-in (e.g. flux.toml) gets
    rebranded — built-in wins. Mirrors styles.d semantics."""
    from imgen.backends import BUILTIN_BACKENDS, merge_user_backends
    user = {"flux": _bare_backend("user-tampered")}
    merged = merge_user_backends(BUILTIN_BACKENDS, user)
    # Built-in flux unchanged.
    assert merged["flux"].binary == "mflux-generate-kontext"
    # User entry renamed.
    assert "flux_0001" in merged
    assert merged["flux_0001"].binary == "user-tampered"
    # User informed.
    combined = capsys.readouterr()
    combined_str = combined.out + combined.err
    assert "flux" in combined_str
    assert "flux_0001" in combined_str


def test_merge_does_not_mutate_inputs():
    from imgen.backends import BUILTIN_BACKENDS, merge_user_backends
    builtins_snapshot = dict(BUILTIN_BACKENDS)
    user = {"new": _bare_backend()}
    user_snapshot = dict(user)
    merge_user_backends(BUILTIN_BACKENDS, user)
    assert BUILTIN_BACKENDS == builtins_snapshot
    assert user == user_snapshot


def test_merge_strip_then_resuffix_for_user_name_with_trailing_NNNN(capsys):
    """User backend named `flux_0001` colliding with a previous-pass
    auto-rename: re-suffix becomes `flux_0002`, NOT `flux_0001_0001`.
    Same semantics as styles."""
    from imgen.backends import BUILTIN_BACKENDS, merge_user_backends
    user = {
        "flux": _bare_backend("user1"),
        "flux_0001": _bare_backend("user2"),
    }
    merged = merge_user_backends(BUILTIN_BACKENDS, user)
    # flux built-in survives, user1 → flux_0001, user2 → flux_0002.
    assert merged["flux"].binary == "mflux-generate-kontext"
    assert merged["flux_0001"].binary == "user1"
    assert merged["flux_0002"].binary == "user2"


# ── list_backends / get_backend / reset_backends_cache ──────────────────


@pytest.fixture
def isolated_backends_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR/backends.d/ to a tmp_path location so a real
    ~/.imgen/backends.d/ doesn't leak into the test. Also resets the
    in-process cache so each test sees a fresh load."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    backends_mod.reset_backends_cache()
    yield backends_dir
    backends_mod.reset_backends_cache()


def test_list_backends_includes_user_entries(isolated_backends_dir):
    from imgen.backends import list_backends
    _write_toml(
        isolated_backends_dir / "myback.toml",
        'binary = "my-bin"\nimage_flag = "--image-path"\n',
    )
    names = list_backends()
    assert "flux" in names
    assert "qwen" in names
    assert "myback" in names


def test_get_backend_returns_user_backend(isolated_backends_dir):
    from imgen.backends import get_backend
    _write_toml(
        isolated_backends_dir / "myback.toml",
        'binary = "my-bin"\nimage_flag = "--image-paths"\n',
    )
    be = get_backend("myback")
    assert be.binary == "my-bin"
    assert be.image_flag == "--image-paths"


def test_get_backend_unknown_raises_keyerror(isolated_backends_dir):
    from imgen.backends import get_backend
    with pytest.raises(KeyError, match="nonexistent"):
        get_backend("nonexistent")


def test_reset_backends_cache_picks_up_new_files(isolated_backends_dir):
    """After the first load + cache, a new file in backends.d/ is not
    seen until reset. Lock-in test for the cache invalidation contract
    — without it, dev iteration on a TOML file requires a full
    process restart."""
    from imgen.backends import list_backends, reset_backends_cache
    # First load — only built-ins.
    assert "newly_dropped" not in list_backends()
    # Drop a file.
    _write_toml(
        isolated_backends_dir / "newly_dropped.toml",
        'binary = "x"\nimage_flag = "--image-path"\n',
    )
    # Without reset, still not seen.
    assert "newly_dropped" not in list_backends()
    # After reset, picked up.
    reset_backends_cache()
    assert "newly_dropped" in list_backends()

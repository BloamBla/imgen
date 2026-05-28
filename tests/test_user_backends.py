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


def test_validator_rejects_extra_args_with_control_bytes():
    """v0.4 security-reviewer NIT-2: argv strings reach mflux stderr
    (and our log files via the redaction-tee). Embedded ESC bytes
    could clear screens or set terminal titles when the user later
    `cat ~/.imgen/logs/<id>.log`. Filter at schema time, symmetric
    with the `binary` field defence."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "extra_args": ["--model", "evil\x1b[2J"],
    }
    with pytest.raises(UserBackendError, match="extra_args.*control"):
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
    # v0.5: enhance_* fields default to None / empty tuple → user
    # backend cleanly opts out of the LLM enhancer.
    assert be.enhance_system_prompt is None
    assert be.enhance_invariants == ()


# ── v0.5: optional enhance_* fields on user backends ────────────────────


def test_validator_accepts_enhance_system_prompt():
    """User backends may declare their own enhancer system prompt to
    enable ``--enhance-prompt`` for that backend. Mirrors the built-in
    flux/qwen pattern (different prompt conventions, different system
    instructions). Closes architect CRITICAL #2 from v0.5 pre-tag
    review."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "enhance_system_prompt": "You expand prompts for MyModel.",
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.enhance_system_prompt == "You expand prompts for MyModel."
    assert be.enhance_invariants == ()  # absent → default empty tuple


def test_validator_accepts_enhance_invariants():
    """``enhance_invariants`` is a list of identity-anchor substrings
    that the LLM output must preserve. Comes in as TOML list, stored
    as tuple to match the frozen Backend dataclass shape."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "enhance_system_prompt": "...",
        "enhance_invariants": ["preserving", "identity"],
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.enhance_invariants == ("preserving", "identity")
    assert isinstance(be.enhance_invariants, tuple)


def test_validator_rejects_enhance_system_prompt_with_control_bytes():
    """Symmetric with extra_args defence. The enhance system prompt
    ends up in subprocess argv (via JSON stdin to enhance_runner) AND
    in dry-run terminal display — a ``\\x1b`` byte could leak terminal
    escapes."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "enhance_system_prompt": "evil\x1b[2J prompt",
    }
    with pytest.raises(UserBackendError, match="enhance_system_prompt.*control"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_enhance_invariants_with_control_bytes():
    """Each invariant string flows into history.jsonl + log files. Apply
    the same control-byte filter as extra_args."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "enhance_system_prompt": "...",
        "enhance_invariants": ["good", "bad\x1b[m"],
    }
    with pytest.raises(UserBackendError, match="enhance_invariants.*control"):
        validate_user_backend_schema(data, Path("test.toml"))


# ── v0.6: lora_compat_group field on user backends ──────────────────


def test_validator_accepts_lora_compat_group():
    """User backends opt into LoRA support by declaring their compat
    group identifier (matches LoraRef.compatible_with in style TOMLs).
    Common values: "flux-1", "flux-2", "qwen". Bare lower-case
    stems."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "lora_compat_group": "flux-2",
    }
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.lora_compat_group == "flux-2"


def test_validator_lora_compat_group_defaults_to_empty():
    """Absent → "" → "this backend has no LoRA support" → any LoRA
    declared in a style is silently warn-skipped for this backend.
    Defensive default matches built-in backends' implicit no-LoRA
    state for any backend that doesn't opt in."""
    data = {"binary": "x", "image_flag": "--image-path"}
    be = validate_user_backend_schema(data, Path("test.toml"))
    assert be.lora_compat_group == ""


def test_validator_rejects_empty_lora_compat_group():
    """Empty string is the "no LoRA" sentinel — users who explicitly
    set it to "" mean the same as omitting, but it's confusing to
    accept both. Reject empty so the user sees a clean error."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        "lora_compat_group": "   ",
    }
    with pytest.raises(UserBackendError, match="lora_compat_group"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_lora_compat_group_with_control_bytes():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "lora_compat_group": "flux\x1b-1",
    }
    with pytest.raises(UserBackendError, match="lora_compat_group"):
        validate_user_backend_schema(data, Path("test.toml"))


def test_validator_rejects_non_string_lora_compat_group():
    data = {
        "binary": "x", "image_flag": "--image-path",
        "lora_compat_group": 42,
    }
    with pytest.raises(UserBackendError, match="lora_compat_group"):
        validate_user_backend_schema(data, Path("test.toml"))


# ── Built-in backends carry lora_compat_group (lock-in) ─────────────


def test_builtin_flux_carries_flux1_lora_compat_group():
    """Built-in flux backend declares ``"flux-1"`` so FLUX.1-family
    LoRAs (the bulk of what's published on HF for FLUX.1-dev) apply
    cleanly to FLUX.1-Kontext-dev. Lock-in test against accidental
    drop of the field."""
    from imgen.backends import BUILTIN_BACKENDS
    assert BUILTIN_BACKENDS["flux"].lora_compat_group == "flux-1"


def test_builtin_qwen_carries_qwen_lora_compat_group():
    """Qwen LoRAs are a separate ecosystem from FLUX. The "qwen"
    group identifier ensures FLUX LoRAs don't accidentally apply to
    Qwen backends (different transformer architecture)."""
    from imgen.backends import BUILTIN_BACKENDS
    assert BUILTIN_BACKENDS["qwen"].lora_compat_group == "qwen"


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


def test_validator_rejects_plural_secrets_form():
    """v0.4 architect IMP-3 fallback: a TOML using `[[secrets]]`
    (plural array-of-tables) hits the unknown-field branch and would
    silently warn-and-drop. Reject explicitly with a message
    pointing to the singular [secret] form, so a colleague who
    types the plural by reflex gets a clear error instead of
    "your second secret is silently missing at runtime"."""
    data = {
        "binary": "x", "image_flag": "--image-path",
        # tomllib parses [[secrets]] as a list of dicts.
        "secrets": [{"env_var": "FOO"}, {"env_var": "BAR"}],
    }
    with pytest.raises(UserBackendError, match="plural"):
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


# ── v0.7.12 (gap 5): hf_gated_repo in user TOML schema ───────────────────

def test_load_user_backend_file_accepts_hf_gated_repo(tmp_path):
    """v0.7.12 (gap 5): user TOML can declare ``hf_gated_repo`` so the
    backend's gate-URL hint (surfaced by doctor + cmd_draw post-failure
    handler) works for user-registered gated repos like
    ``briaai/FIBO``, matching how FLUX.1-dev's built-in row uses it."""
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "fibo.toml"
    _write_toml(toml_path, '''
binary = "mflux-generate-fibo"
image_flag = "--image-paths"
hf_gated_repo = "briaai/FIBO"
extra_args = ["-m", "briaai/FIBO"]
''')
    be = load_user_backend_file(toml_path)
    assert be.hf_gated_repo == "briaai/FIBO"


def test_load_user_backend_file_hf_gated_repo_defaults_to_none(tmp_path):
    """Field is optional — TOMLs that don't declare it get None on the
    Backend (matches built-in qwen behaviour: open repo, no gate hint
    needed)."""
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "open.toml"
    _write_toml(toml_path, '''
binary = "mflux-generate-foo"
image_flag = "--image-path"
extra_args = []
''')
    be = load_user_backend_file(toml_path)
    assert be.hf_gated_repo is None


def test_load_user_backend_file_rejects_hf_gated_repo_with_control_bytes(tmp_path):
    """C0/DEL/C1 byte in hf_gated_repo would leak into the post-failure
    hint URL display + doctor output → terminal escape injection risk
    matching all the other string-field hardenings (binary, extra_args,
    enhance_system_prompt). Schema rejects at load time."""
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "evil.toml"
    _write_toml(toml_path, '''
binary = "mflux-generate-foo"
image_flag = "--image-path"
hf_gated_repo = "evil/\\u001b[2Jrepo"
extra_args = []
''')
    with pytest.raises(UserBackendError, match="hf_gated_repo"):
        load_user_backend_file(toml_path)


def test_load_user_backend_file_rejects_hf_gated_repo_empty(tmp_path):
    """Empty string is also invalid — either declare a real repo or
    omit the field entirely."""
    from imgen.backends import load_user_backend_file
    toml_path = tmp_path / "empty_repo.toml"
    _write_toml(toml_path, '''
binary = "mflux-generate-foo"
image_flag = "--image-path"
hf_gated_repo = ""
extra_args = []
''')
    with pytest.raises(UserBackendError, match="hf_gated_repo"):
        load_user_backend_file(toml_path)


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
    # v0.7.0 added flux-dev; v0.7.5 added flux2-klein-edit-9b;
    # v0.9 commit 7 added ltx-video; v0.10 commit 2 added flux2-klein-4b.
    # User TOMLs land on top of the built-in set.
    assert set(merged.keys()) == {
        "flux", "qwen", "flux-dev", "flux2-klein-edit-9b", "ltx-video",
        "flux2-klein-4b", "flux2-klein-4b-edit",  # v0.11.1 (V-2)
        "sdxl",
    }
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
    in-process cache so each test sees a fresh load.

    Patches BOTH ``STATE_DIR`` and ``BACKENDS_D`` — the latter is a
    module-level constant captured at paths.py import time, so a
    ``STATE_DIR`` monkeypatch alone leaves BACKENDS_D pointing at the
    real ``~/.imgen/backends.d/``. (v0.4 architect IMP-4 trap.)
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    state = tmp_path / ".imgen"
    state.mkdir()
    backends_dir = state / "backends.d"
    backends_dir.mkdir()
    monkeypatch.setattr(paths_mod, "STATE_DIR", state)
    monkeypatch.setattr(paths_mod, "BACKENDS_D", backends_dir)
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

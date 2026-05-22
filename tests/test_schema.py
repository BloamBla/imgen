"""Shared schema validator (extracted v0.4 — architect IMP-2).

The helper is now the single source of truth for config.py /
styles.py / backends.py (top-level + [secret]) validation loops.
Tests here lock the helper's behaviour directly so future tweaks
have a thin, focused surface to update rather than re-discovering
the contract via the four consumer modules.
"""
from __future__ import annotations

import pytest

from imgen._schema import validate_against_schema


class _Exc(Exception):
    """Test-only sentinel exception."""


_TRIVIAL_SCHEMA = {
    "foo": ("string", lambda v: isinstance(v, str)),
    "bar": ("int", lambda v: isinstance(v, int) and not isinstance(v, bool)),
}


# ── Happy path ──────────────────────────────────────────────────────────


def test_validates_known_keys_with_good_values():
    data = {"foo": "hello", "bar": 42}
    out = validate_against_schema(data, _TRIVIAL_SCHEMA, _Exc, source="t")
    assert out == {"foo": "hello", "bar": 42}


def test_returns_new_dict_does_not_mutate_input():
    """Pure: caller's dict is never modified, and a new instance is
    returned so the caller can freely modify the result."""
    data = {"foo": "x"}
    snapshot = dict(data)
    out = validate_against_schema(data, _TRIVIAL_SCHEMA, _Exc, source="t")
    assert data == snapshot
    assert out is not data


# ── Unknown keys (warn + drop) ──────────────────────────────────────────


def test_drops_unknown_keys_with_warn(capsys):
    data = {"foo": "x", "future_field": "ignored"}
    out = validate_against_schema(data, _TRIVIAL_SCHEMA, _Exc, source="t")
    assert "future_field" not in out
    assert out == {"foo": "x"}
    combined = capsys.readouterr()
    assert "future_field" in (combined.out + combined.err)
    assert "unknown" in (combined.out + combined.err)


def test_unknown_key_message_includes_source():
    """Source label is embedded in warns so a user with multiple
    config files reading the error can identify which one tripped."""
    capsys_buf = []
    data = {"unknown_thing": True}
    out = validate_against_schema(
        data, _TRIVIAL_SCHEMA, _Exc, source="my_source.toml",
    )
    assert out == {}


# ── Bad values (raise exc_type) ─────────────────────────────────────────


def test_raises_exc_type_on_bad_value():
    """The chosen exception class is raised, not the default Exception.
    This is what makes the helper safe to share across modules with
    distinct error types (ConfigError / UserStyleError /
    UserBackendError)."""
    data = {"foo": 42}  # foo should be string

    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        validate_against_schema(data, _TRIVIAL_SCHEMA, CustomError, source="t")


def test_error_message_includes_source_field_desc_value():
    data = {"foo": 42}
    with pytest.raises(_Exc) as exc_info:
        validate_against_schema(data, _TRIVIAL_SCHEMA, _Exc, source="t.toml")
    msg = str(exc_info.value)
    assert "t.toml" in msg
    assert "foo" in msg
    assert "string" in msg  # description from schema
    assert "42" in msg      # value repr


# ── skip_keys (used by backends.py top-level pass) ──────────────────────


def test_skip_keys_silently_omitted_no_warn(capsys):
    """Top-level backend validation skips the 'secret' key because it's
    handled by a separate nested pass. Skipped keys must NOT trigger
    an "unknown field" warn — that would be a spurious error for the
    user."""
    data = {"foo": "x", "skip_me": "anything", "drop_me": "warn"}
    out = validate_against_schema(
        data, _TRIVIAL_SCHEMA, _Exc, source="t", skip_keys={"skip_me"},
    )
    assert out == {"foo": "x"}
    combined = capsys.readouterr()
    out_str = combined.out + combined.err
    assert "skip_me" not in out_str  # not warned
    assert "drop_me" in out_str       # warned as usual


# ── field_prefix (used by backends.py [secret] nested pass) ─────────────


def test_field_prefix_in_error_message():
    """[secret] subsection uses field_prefix='secret.' so error
    messages read 'source: secret.field: ...' rather than just
    'source: field: ...'."""
    data = {"foo": 42}
    with pytest.raises(_Exc) as exc_info:
        validate_against_schema(
            data, _TRIVIAL_SCHEMA, _Exc,
            source="t.toml", field_prefix="secret.",
        )
    assert "secret.foo" in str(exc_info.value)


def test_field_prefix_renders_bracketed_in_unknown_warn(capsys):
    """Unknown-field warns under a prefix get '[secret] unknown
    field ...' shape so the user sees which subsection complained."""
    data = {"alien_field": "x"}
    validate_against_schema(
        data, _TRIVIAL_SCHEMA, _Exc,
        source="t.toml", field_prefix="secret.",
    )
    combined = capsys.readouterr()
    out_str = combined.out + combined.err
    assert "[secret]" in out_str
    assert "alien_field" in out_str

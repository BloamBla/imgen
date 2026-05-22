"""Shared schema validation for user-supplied TOML data.

Four call sites used to carry near-identical copies of the same
``(description, predicate)`` validation loop:

  * :func:`imgen.config.validate_section` for ``~/.imgen/config.toml``
    ``[defaults]`` / ``[ui]`` sections (since v0.2.0).
  * :func:`imgen.styles.load_user_style_file` for
    ``~/.imgen/styles.d/*.toml`` (since v0.2.0).
  * :func:`imgen.backends.validate_user_backend_schema` top-level fields
    for ``~/.imgen/backends.d/*.toml`` (v0.4).
  * Same function's nested ``[secret]`` section validation (v0.4).

Architect IMP-2 from the v0.4 review caught this duplication
("4 inline copies, the same file as the third"). The architect's
original v0.2 backlog item suggested extracting a
``validate_against_schema`` helper; v0.4 hit the 3rd-caller-rule
threshold and pulls it forward.

The signature is small and deliberately not pluggable beyond what
the four call sites need: ``skip_keys`` lets the top-level backends
validator skip the ``"secret"`` nested table that's validated in a
second pass, and ``field_prefix`` lets the nested-table pass label
its errors as ``"source: secret.foo:"`` rather than just
``"source: foo:"``.
"""
from __future__ import annotations

from typing import Any, Callable, Container, Mapping

__all__ = ["validate_against_schema"]


def validate_against_schema(
    data: Mapping[str, Any],
    schema: Mapping[str, tuple[str, Callable[[Any], bool]]],
    exc_type: type[Exception],
    *,
    source: str,
    field_prefix: str = "",
    skip_keys: Container[str] = frozenset(),
) -> dict[str, Any]:
    """Filter ``data`` to keys present in ``schema``, validating values.

    Args:
        data:         Parsed TOML / config dict (must be a mapping).
        schema:       ``{field_name: (description, predicate)}``. The
                      description is rendered into error messages; the
                      predicate decides if a value is acceptable.
        exc_type:     Exception class to raise on bad values
                      (``ConfigError`` / ``UserStyleError`` /
                      ``UserBackendError``).
        source:       Source identifier embedded in messages — usually
                      a path or ``"<path> [section]"`` for nested
                      contexts.
        field_prefix: Prepended to the field name in error messages.
                      Used by nested-table validation to produce
                      ``"source: parent.child: expected ..."`` shape.
        skip_keys:    Keys present in ``data`` that should be silently
                      skipped (neither validated nor warned about).
                      Top-level backend validation uses this for
                      ``"secret"`` — the section is validated in a
                      separate pass.

    Returns:
        A new dict containing only the validated keys + values.
        Unknown keys are dropped after emitting a warn.

    Raises:
        ``exc_type``: A schema-known key whose value fails its
        predicate. Message format:
        ``f"{source}: {field_prefix}{key}: expected {desc}, got {value!r}"``.
    """
    # Local import keeps this module dependency-free at the top level
    # — colors.warn pulls in terminal-detection state we don't need
    # to materialize for callers that import _schema in isolation.
    from .colors import warn

    validated: dict[str, Any] = {}
    for key, value in data.items():
        if key in skip_keys:
            continue
        if key not in schema:
            label = f"[{field_prefix.rstrip('.')}] " if field_prefix else ""
            warn(f"{source}: {label}unknown field '{key}' — ignored")
            continue
        desc, predicate = schema[key]
        if not predicate(value):
            raise exc_type(
                f"{source}: {field_prefix}{key}: expected {desc}, "
                f"got {value!r}"
            )
        validated[key] = value
    return validated

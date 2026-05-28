"""v0.10.0 commit 10 — trained-LoRA ``.meta.json`` sidecar reader.

Leaf module (stdlib + ``_safe`` only) so both ``parser`` (resolution-
time enrichment) and any future ``imgen lora list`` / ``imgen doctor``
caller can import it without a layering inversion.

``read_lora_meta`` is the consumer side of the sidecar written by
``imgen train`` (commit 6 ``build_meta_json`` / ``_write_meta_json``).
It is intentionally BEST-EFFORT: a missing / corrupt / oversized /
tampered sidecar degrades to ``(None, None)`` (or drops just the bad
field) so a malformed file never breaks an ``imgen draw`` run. The
trigger word and compat group are conveniences, not correctness-
critical inputs — when in doubt, skip the enrichment.

Per [[project-v100-design]] §I + §H.3 + §R.1 ROUND-1 CLOSURES:
* 16 KB read cap (security C-3 — DoS gate, mirror of
  ``USER_BACKEND_MAX_BYTES``).
* Trigger length 1..64 re-validated on READ (python H-8 — the file is
  hand-editable; don't trust the write-time validation).
* Control-byte filter on the trigger (defence-in-depth; the
  ``_trigger_token_arg`` validator already filtered at train time).
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from ._safe import has_control_bytes

__all__ = ["read_lora_meta"]

# Security C-3: 16 KB cap. The sidecar imgen writes is ~600 bytes; a
# file an order of magnitude larger is a tampering / DoS signal.
_META_MAX_BYTES: int = 16 * 1024

# python H-8: trigger length bounds re-checked on READ (same envelope
# as the _trigger_token_arg validator at train time).
_TRIGGER_MIN_LEN: int = 1
_TRIGGER_MAX_LEN: int = 64


def read_lora_meta(
    safetensors_path: Path,
) -> tuple[str | None, str | None]:
    """Read the ``<stem>.meta.json`` sidecar next to a trained LoRA.

    Returns ``(trigger, compat_group)``:

    * ``trigger`` — the validated trigger phrase, or ``None`` if the
      sidecar is missing / corrupt / oversized, or the trigger field
      is absent / not a string / empty / >64 chars / control-byte-
      bearing.
    * ``compat_group`` — the ``lora_compat_group`` string, or ``None``
      if absent / not a string.

    The two fields degrade INDEPENDENTLY past the file-level gate: a
    valid file with a bad trigger but a good compat group returns
    ``(None, "flux2-klein-4b")``. File-level failures (missing,
    corrupt, oversized, non-UTF-8) return ``(None, None)``.
    """
    meta_path = safetensors_path.with_suffix(".meta.json")
    try:
        raw = meta_path.read_bytes()
    except (OSError, ValueError):
        return (None, None)

    # Security C-3: cap BEFORE decode/parse so a giant file can't force
    # a multi-MB JSON parse.
    if len(raw) > _META_MAX_BYTES:
        return (None, None)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return (None, None)

    try:
        meta = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return (None, None)

    if not isinstance(meta, dict):
        return (None, None)

    return (_extract_trigger(meta), _extract_compat_group(meta))


def _extract_trigger(meta: dict) -> str | None:
    """python H-8 + control-byte + Unicode Cf/Mn re-validation on READ.

    Mirrors the write-side ``_trigger_token_arg`` validator: the
    ``.meta.json`` is hand-editable, so re-apply the SAME rejections
    (length, C0/DEL/C1 control bytes, and bidi-override / zero-width /
    combining-mark categories) before the trigger is prepended to an
    ``imgen draw`` prompt (security M-2 — identity-spoofing defence).
    """
    trigger = meta.get("trigger")
    if not isinstance(trigger, str):
        return None
    trigger = trigger.strip()
    if not (_TRIGGER_MIN_LEN <= len(trigger) <= _TRIGGER_MAX_LEN):
        return None
    if has_control_bytes(trigger):
        return None
    if any(unicodedata.category(ch) in ("Cf", "Mn") for ch in trigger):
        return None
    return trigger


def _extract_compat_group(meta: dict) -> str | None:
    compat = meta.get("lora_compat_group")
    if not isinstance(compat, str) or not compat:
        return None
    return compat

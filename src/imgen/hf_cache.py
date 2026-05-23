"""HuggingFace cache layout helpers — single source of truth for the
``models--<author>--<name>`` directory convention used by
``huggingface_hub``.

Three call sites used to duplicate this mapping (``commands/doctor.py``
for the enhance-model cache probe + the model listing in `doctor`,
``commands/clean.py`` for ``clean --all`` rendering, ``parser.py``
for the ``--list-loras`` cache probe). v0.6.2 (architect I-3 from the
v0.6 pre-tag review) extracted them here so a future change to the HF
convention (or a HF cache-layout migration) lands in exactly one place.

Both functions are pure: no I/O, no env access. ``hf_cache_dir_for``
returns a path object the caller can ``is_dir()`` or stat; whether
that directory exists is the caller's question.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "hf_cache_dir_for",
    "repo_from_cache_dir",
]


def hf_cache_dir_for(repo: str, hf_cache: Path) -> Path:
    """Return the ``models--<author>--<name>`` directory under ``hf_cache``
    for an HF repo id.

    Empty ``repo`` returns ``hf_cache`` itself (an edge case the existing
    callers preserved — keeps ``Path / hf_cache_dir_for("", hf_cache)``
    valid without raising); absolute-path ``repo`` (the user pointed
    ``--enhance-model`` or a LoRA at a local checkpoint) returns the
    path verbatim so the caller's ``is_dir()`` probe just checks the
    on-disk location.

    The convention mirrors ``huggingface_hub.cached_assets``: an
    ``author/name`` repo lands under
    ``<HF_HOME>/hub/models--<author>--<name>/`` with ``snapshots/`` and
    ``blobs/`` subdirectories.
    """
    if not repo or repo.startswith("/"):
        return Path(repo) if repo else hf_cache
    return hf_cache / ("models--" + repo.replace("/", "--"))


def repo_from_cache_dir(name: str) -> str:
    """Reverse the convention: ``models--openfree--flux-chatgpt-ghibli-lora``
    → ``openfree/flux-chatgpt-ghibli-lora``. Used by ``imgen doctor``
    and ``imgen clean --all`` to render cached-model names back into
    the form the user typed.

    Caller is responsible for passing only a name with the
    ``models--`` prefix; an arbitrary string is returned with both
    leading prefix and ``--`` substrings rewritten anyway (preserving
    the v0.1.x behaviour) — but the function's contract is only
    well-defined for the cached-model directory layout.
    """
    return name.replace("models--", "").replace("--", "/")

"""v0.8.0 Engine abstraction layer.

Per [[project-v080-design]] §C. ``Engine`` is the dispatch Protocol;
``GenParams`` is the pure-data parameter envelope. Subclasses:

* ``MfluxEngine`` (commit 2) — wraps mflux-generate-* subprocess.
* ``DiffusersMpsEngine`` (commit 6) — out-of-process diffusers via
  stdin-JSON to a static runner in ``_diffusers_runner``.
"""
from __future__ import annotations

from .base import Engine, GenParams
from .diffusers_mps_engine import DiffusersMpsEngine
from .mflux_engine import MfluxEngine

__all__ = ["DiffusersMpsEngine", "Engine", "GenParams", "MfluxEngine"]

"""v0.8.0 Engine abstraction layer.

Per [[project-v080-design]] §C. ``Engine`` is the dispatch Protocol;
``GenParams`` is the pure-data parameter envelope. Subclasses:

* ``MfluxEngine`` (commit 2) — wraps mflux-generate-* subprocess.
* ``DiffusersMpsEngine`` (commit 6) — out-of-process diffusers via
  stdin-JSON to a static runner in ``_diffusers_runner``.

v0.9.5 architect M-2 closure: ``ENGINES`` dict + ``get_engine``
consolidate the ``if engine == "mflux" / elif "diffusers_mps"``
dispatch previously duplicated at 3 sites. Single source of truth
for the engine-name → class mapping. A drift-lock test in
``tests/test_engines.py::TestEngineRegistry`` pins ENGINES keys
against ``Model.__post_init__``'s accept set so future contributors
can't register one without the other.
"""
from __future__ import annotations

from .base import Engine, GenParams
from .diffusers_mps_engine import DiffusersMpsEngine
from .mflux_engine import MfluxEngine

__all__ = [
    "DiffusersMpsEngine",
    "Engine",
    "ENGINES",
    "GenParams",
    "MfluxEngine",
    "get_engine",
]


# v0.9.5 M-2: registry. Adding a 3rd engine = one-line dict entry;
# the 3 dispatch sites (_engine_for_model in engine_dispatch,
# ram_required_gb in checks, doctor RAM forecast) pick it up
# automatically via get_engine().
#
# ``Model.__post_init__`` keeps its literal {'mflux',
# 'diffusers_mps'} guard intentionally — its per-engine invariants
# (mflux needs binary=, diffusers_mps needs repo=) are intrinsic,
# not pure dispatch. The drift-lock test pins ENGINES keys against
# Model's accept set so adding a 3rd engine surfaces the gap.
#
# ``iteration_dryrun_display`` (engine_dispatch:_iteration_dryrun_*)
# also keeps its branched code path — its diffusers branch routes
# through _format_diffusers_dryrun / _format_diffusers_video_dryrun
# helpers (different shape per output_type), NOT through
# Engine.format_dryrun (the Protocol doesn't define that method).
# Lifting dryrun rendering into the Engine Protocol is a v0.10.x
# design call when the 3rd engine actually lands.
ENGINES: dict[str, type[Engine]] = {
    "mflux": MfluxEngine,
    "diffusers_mps": DiffusersMpsEngine,
}


def get_engine(name: str) -> Engine:
    """Return a fresh Engine instance for ``name``.

    Raises ``ValueError`` with the registered names listed for
    discoverability. Callers needing a ``SystemExit`` (e.g.
    ``_engine_for_model``) catch and die() at the application
    boundary — the registry is pure data, error class is the
    caller's call.

    Each invocation constructs a new instance (Engine impls are
    cheap dataclasses); callers may cache if they want. The doctor
    RAM forecast pre-builds one instance per registered engine to
    avoid N-loop allocations across the per-Model grid.
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise ValueError(
            f"unknown engine={name!r}; expected one of {sorted(ENGINES)}"
        )
    return engine_cls()

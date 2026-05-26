"""v0.8.4 M-NEW-D test helper: derive mflux argv from an Iteration.

Pre-v0.8.4 ``Iteration.cmd`` carried the build-time argv snapshot;
v0.8.4 drops the field and derives argv at dispatch-time via
``MfluxEngine.build_cmd(model, params)``. Tests that previously read
``it.cmd[i]`` for assertions now route through this helper.

Lives in its own module (not conftest) so it's a clean ``from
_iteration_argv import iteration_argv`` import — conftest.py is for
fixtures, mixing free helpers into it muddies the namespace.
"""
from __future__ import annotations

from pathlib import Path


def iteration_argv(it, binary=None) -> list[str]:
    """Return ``MfluxEngine.build_cmd(it.model, it.params, binary=...)``.

    Same path that ``MfluxEngine.run`` uses internally and that
    ``engine_dispatch.iteration_dryrun_display`` shows on dry-run — so
    test assertions on argv content remain authoritative for "what
    mflux would actually see".

    ``binary`` defaults to ``/fake/bin/<model.binary>`` so existing
    tests keep their cmd[0] shape from the pre-v0.8.4 era when build
    helpers were called with the same fake path. Pass an explicit
    value when a test cares about argv[0] identity.
    """
    from imgen.engines.mflux_engine import MfluxEngine
    if binary is None:
        binary = Path(f"/fake/bin/{it.model.binary}")
    return MfluxEngine().build_cmd(it.model, it.params, binary=binary)

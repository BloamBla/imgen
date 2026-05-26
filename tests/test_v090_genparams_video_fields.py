"""v0.9 commit 2 — GenParams.num_frames + GenParams.fps widening.

Per [[project-v090-design]] §D + §B keystone 3. Both fields appended
AT THE END of the GenParams field list (after ``battery_stop``).
Defaulted (``num_frames=1`` / ``fps=24``) so v0.8 image callers don't
need to pass them — including positional callers, which stay
byte-additive.

Architect's v0.8.0 reconciliation noted: "don't speculate, widen at
concrete-runtime time." LTX IS the concrete runtime. This widening is
the speculation-free version.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest


def _minimal_genparams(**overrides):
    """Smallest valid GenParams — every test starts from here."""
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="a samurai",
        negative="",
        width=512,
        height=512,
        steps=20,
        guidance=3.5,
        seed=42,
        quantize=8,
        strength=0.5,
        input_path=None,
        output_path=Path("/tmp/out.png"),
        loras=(),
    )
    defaults.update(overrides)
    return GenParams(**defaults)


class TestGenParamsVideoFieldShape:
    """v0.9 commit 2: GenParams widens with num_frames + fps."""

    def test_genparams_has_num_frames_field(self):
        from imgen.engines.base import GenParams
        names = {f.name for f in fields(GenParams)}
        assert "num_frames" in names

    def test_genparams_has_fps_field(self):
        from imgen.engines.base import GenParams
        names = {f.name for f in fields(GenParams)}
        assert "fps" in names

    def test_genparams_default_num_frames_is_1(self):
        """Image default — single-frame output for v0.8 callers."""
        p = _minimal_genparams()
        assert p.num_frames == 1

    def test_genparams_default_fps_is_24(self):
        """Video standard default; ignored when num_frames == 1."""
        p = _minimal_genparams()
        assert p.fps == 24


class TestGenParamsFieldOrderLock:
    """§B keystone 3 + §D: video fields appended AT END of field list.
    Ordering matters for two reasons: (a) positional construction
    stays byte-additive for v0.8 callers; (b) non-default-before-
    default rule preserved (all v0.8 fields except defaults stay
    in the same relative order)."""

    def test_v0_9_genparams_field_order_locked(self):
        """The full v0.9 field order — drift here forces a deliberate
        update. Both video fields are LAST."""
        from imgen.engines.base import GenParams
        order = [f.name for f in fields(GenParams)]
        expected = [
            "prompt", "negative", "width", "height",
            "steps", "guidance", "seed", "quantize", "strength",
            "input_path", "output_path", "loras",
            "mlx_cache_gb", "battery_stop",
            # v0.9 commit 2 — appended at END
            "num_frames", "fps",
        ]
        assert order == expected, (
            f"GenParams field-order drift: got {order!r} vs expected {expected!r}"
        )

    def test_num_frames_appended_after_battery_stop(self):
        """Explicit ordering invariant — num_frames immediately follows
        battery_stop, fps follows num_frames."""
        from imgen.engines.base import GenParams
        order = [f.name for f in fields(GenParams)]
        idx_battery_stop = order.index("battery_stop")
        idx_num_frames = order.index("num_frames")
        idx_fps = order.index("fps")
        assert idx_num_frames == idx_battery_stop + 1
        assert idx_fps == idx_num_frames + 1


class TestGenParamsV08PositionalConstructionStillWorks:
    """§D verified-at-commit-2 lock-in. v0.8-shaped positional
    construction (no num_frames / fps in argv) MUST still work —
    the new fields default to image values, so positional callers
    that stop at the v0.8 surface are byte-additive."""

    def test_genparams_v08_positional_construction_still_works(self):
        """Build GenParams with v0.8 positional argv (14 args, no
        video fields). Result must have num_frames=1 / fps=24 by
        default."""
        from imgen.engines.base import GenParams
        p = GenParams(
            "a samurai",         # prompt
            "",                  # negative
            512,                 # width
            512,                 # height
            20,                  # steps
            3.5,                 # guidance
            42,                  # seed
            8,                   # quantize
            0.5,                 # strength
            None,                # input_path
            Path("/tmp/x.png"),  # output_path
            (),                  # loras
            12,                  # mlx_cache_gb (v0.8.0 §C default)
            20,                  # battery_stop (v0.8.0 §C default)
        )
        # v0.8 fields populated correctly
        assert p.prompt == "a samurai"
        assert p.battery_stop == 20
        # v0.9 fields defaulted (image)
        assert p.num_frames == 1
        assert p.fps == 24

    def test_genparams_v07_minimum_positional_construction_still_works(self):
        """Even shorter positional construction (12 args, no
        mlx_cache_gb / battery_stop either) — both v0.8 AND v0.9
        defaults apply."""
        from imgen.engines.base import GenParams
        p = GenParams(
            "a samurai", "", 512, 512, 20, 3.5, 42, 8, 0.5, None,
            Path("/tmp/x.png"), (),
        )
        # v0.8 fields defaulted
        assert p.mlx_cache_gb == 12
        assert p.battery_stop == 20
        # v0.9 fields defaulted
        assert p.num_frames == 1
        assert p.fps == 24


class TestGenParamsExplicitVideoParams:
    """v0.9 video callers populate num_frames + fps explicitly via
    keyword. Lock-in that the new fields actually carry through."""

    def test_explicit_num_frames_carries_through(self):
        """LTX canonical: 25 frames @ 24 fps ≈ 1 sec."""
        p = _minimal_genparams(num_frames=25, fps=24)
        assert p.num_frames == 25
        assert p.fps == 24

    def test_explicit_fps_25_carries_through(self):
        p = _minimal_genparams(num_frames=25, fps=25)
        assert p.fps == 25

    def test_genparams_with_video_fields_still_frozen(self):
        """Frozen invariant unchanged by widening — verified explicitly
        because new fields could in principle violate frozen=True if
        the wrong dataclass decorator was used."""
        from dataclasses import FrozenInstanceError
        p = _minimal_genparams(num_frames=25, fps=24)
        with pytest.raises(FrozenInstanceError):
            p.num_frames = 33  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            p.fps = 30  # type: ignore[misc]

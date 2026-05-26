"""v0.9 commit 9 — doctor RAM forecast video extension (§K + §L).

Per [[project-v090-design]] §L. The existing doctor RAM forecast
iterates BUILTIN_MODELS × supported_quants. LTX has
supported_quants=() (bf16-only) so the existing loop silently
skipped it. Commit 9 extends the renderer with a dedicated video
section that surfaces the §L envelope rows:

* 768×512 × 25 frames  (canonical: ~1 sec @ 24 fps)
* 1024×576 × 33 frames (heavy mode: ~1.4 sec @ 24 fps)
* 1280×720 × 121 frames (out-of-envelope: ~5 sec @ 24 fps)

Each row shows the model's actual RAM estimate via
``Engine.ram_estimate_gb`` (consistent with image forecast — single
source of truth). Verdict ✅/⚠️/❌ vs available RAM.
"""
from __future__ import annotations

import pytest


def _ram_forecast_output(available_ram: float) -> str:
    """Capture _render_ram_forecast_rows stdout for the given budget."""
    import io
    import contextlib
    from imgen.commands.doctor import _render_ram_forecast_rows
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _render_ram_forecast_rows(available_ram)
    return buf.getvalue()


class TestVideoForecastRowsAppearForLTX:
    """LTX-Video built-in row was added in commit 7 (pulled forward
    from commit 9). The doctor renderer must now surface its §L
    envelope rows."""

    def test_ltx_video_appears_in_forecast(self):
        out = _ram_forecast_output(32.0)
        assert "ltx-video" in out, (
            f"ltx-video missing from doctor forecast; got: {out!r}"
        )

    def test_ltx_canonical_768x512_25frames_row(self):
        """§L canonical envelope row — ~17 GB on M2 Pro 32 GB."""
        out = _ram_forecast_output(32.0)
        # Row should mention either 768x512 / 768×512 or 25 frames
        assert "25" in out and ("768" in out or "0.39" in out.lower())

    def test_ltx_heavy_mode_1024x576_33frames_row(self):
        """§L heavy mode — possible with closed apps, ~19 GB."""
        out = _ram_forecast_output(32.0)
        assert "33" in out and ("1024" in out or "576" in out)

    def test_ltx_out_of_envelope_1280x720_121frames_row(self):
        """§L out-of-envelope: ~29 GB, infeasible on M2 Pro 32 GB
        without M3 Ultra refurb per user-ml-hardware-plan."""
        out = _ram_forecast_output(32.0)
        assert "121" in out and ("1280" in out or "720" in out)


class TestVideoForecastVerdict:
    """Verdict column reflects available RAM vs §L envelope estimate."""

    def test_32gb_available_canonical_passes(self):
        """At 32 GB available, the ~17 GB canonical LTX row should
        get ✅ (or equivalent fits-marker)."""
        out = _ram_forecast_output(32.0)
        lines_25 = [
            line for line in out.splitlines()
            if "25" in line and "ltx-video" in line
        ]
        assert lines_25, (
            f"no ltx-video 25-frame row in forecast; out: {out!r}"
        )
        # Canonical row should be marked as fitting (✅ in the helper)
        assert any("✅" in line for line in lines_25)

    def test_low_available_canonical_fails(self):
        """At 12 GB available (heavy app load), the ~17 GB row gets ❌."""
        out = _ram_forecast_output(12.0)
        lines_25 = [
            line for line in out.splitlines()
            if "25" in line and "ltx-video" in line
        ]
        assert lines_25
        assert any("❌" in line for line in lines_25)


class TestImageForecastRowsUnchanged:
    """Image Models (FLUX family + Qwen) keep their existing
    per-quant rows — no regression from the video extension."""

    def test_flux_kontext_q8_row_still_appears(self):
        out = _ram_forecast_output(32.0)
        # Per quant grid: flux-kontext q3, q4, q5, q6, q8 all show
        assert "flux-kontext" in out
        assert "q8" in out

    def test_flux2_klein_edit_9b_row_still_appears(self):
        out = _ram_forecast_output(32.0)
        assert "flux2-klein-edit-9b" in out


class TestVideoForecastUsesEngineRamEstimate:
    """§K + §L: doctor video rows derive numbers from
    Engine.ram_estimate_gb — single source-of-truth with the
    preflight gate."""

    def test_canonical_row_matches_design_envelope_17gb(self):
        """§L canonical: LTX 768×512 × 25 frames ≈ 17.07 GB
        (10 baseline + 1.57 slope + 3 encoder + 2.5 frame-term)."""
        out = _ram_forecast_output(32.0)
        lines_25 = [
            line for line in out.splitlines()
            if "25" in line and "ltx-video" in line
        ]
        assert lines_25, "no ltx-video 25-frame row"
        line = lines_25[0]
        # The row should contain a GB number that is "17.X"
        assert "17." in line, (
            f"canonical 25-frame row should show ~17 GB; got: {line!r}"
        )

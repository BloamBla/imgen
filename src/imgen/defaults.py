"""Default parameter values + resource thresholds + mflux pin."""
from __future__ import annotations

__all__ = [
    "DEFAULTS",
    "HISTORY_SCHEMA_VERSION",
    "MFLUX_PIN",
    "MIN_BATTERY_PCT",
    "MIN_DISK_GB",
    "PREVIEW_OVERRIDES",
    "RAM_REQUIRED_GB",
]

DEFAULTS = {
    "style": "pixar",
    "backend": "flux",   # flux | qwen — default for `imgen generate` (i2i)
    "backend_draw": "flux-dev",  # v0.7.0: default for `imgen draw` (t2i)
    "quantize": 8,       # 3 4 5 6 8
    "steps": 20,
    "guidance": 3.5,
    "strength": 0.55,
    "mlx_cache_gb": 12,
    "battery_stop": 20,  # %
}

# --preview overrides (only applied if user didn't explicitly set the flag)
PREVIEW_OVERRIDES = {
    "quantize": 4,
    "steps": 8,
}

# Peak RAM (GB) required during inference: UNet weights + text encoders
# + activations + MLX cache headroom. Conservative estimates.
# TODO v0.8: replace fixed-table with dimension-aware
# (backend, quant, megapixels) → GB function. Current shape gates
# refine by worst-case 2K², which over-blocks 24 GB Macs that could
# fit Q4/1.5K refine. See defaults.py block comment + checks.py
# for the swap-in point.
RAM_REQUIRED_GB = {
    ("flux", 3): 8,
    ("flux", 4): 9,
    ("flux", 5): 12,
    ("flux", 6): 14,
    ("flux", 8): 18,
    ("qwen", 3): 10,
    ("qwen", 4): 12,
    ("qwen", 5): 16,
    ("qwen", 6): 18,
    ("qwen", 8): 25,
    # v0.7.0: FLUX.1-dev (t2i) shares the FLUX.1 transformer family
    # weight footprint with FLUX.1-Kontext-dev. Same RAM envelope.
    # Without these rows preflight falls back to the conservative 16GB
    # default — accurate enough for go/no-go but under-reports the
    # actual headroom.
    ("flux-dev", 3): 8,
    ("flux-dev", 4): 9,
    ("flux-dev", 5): 12,
    ("flux-dev", 6): 14,
    ("flux-dev", 8): 18,
    # v0.7.5: FLUX.2-klein-edit-9B (i2i refine backend). 9B params
    # (smaller than FLUX.1's 12B family), but native ≤4 MP support
    # means LARGER activation budget than flux-dev's 1.5K ceiling.
    #
    # v0.7.7 CALIBRATION (real measurement, M2 Pro 32GB, Q4 @ 2048²):
    # resident peak 23.10 GB + compressed 3.88 GB + swap 2.86 GB =
    # ~30 GB total memory pressure. Initial v0.7.5 estimates were
    # speculative (based on backends.py "12-14 GB" comment math that
    # under-counted activations for native 4 MP); bumped to reflect
    # ground truth. Rows now reflect WORST-CASE (2K²) so refine at
    # 1.5K is over-conservative but 16 GB Macs get correctly gate-
    # blocked at Q4+ (cannot fit even Q4/1.5K reliably under memory
    # compression pressure; would OOM after ~20-30 min of refine).
    # Future v0.8: dimension-aware preflight that lets 1.5K Macs
    # pass while gating 2K. Only Q4/2048² is real-measured; Q3 /
    # Q5 / Q6 / Q8 rows are extrapolated linearly from Q4 weights
    # (~4.5 GB) + same activation / TE / cache overhead envelope.
    ("flux2-klein-edit-9b", 3): 22,
    ("flux2-klein-edit-9b", 4): 24,
    ("flux2-klein-edit-9b", 5): 26,
    ("flux2-klein-edit-9b", 6): 28,
    ("flux2-klein-edit-9b", 8): 30,
}

MIN_DISK_GB = 5             # minimum free disk to attempt
MIN_BATTERY_PCT = 30        # below this on battery → warn (not block)

# Pin to known-working mflux version. Bump after manual verification.
MFLUX_PIN = "mflux==0.17.5"

# Bump when an entry field changes meaning. Old entries without "v" key
# are treated as v=0 and still replay (best-effort .get throughout). An
# entry with "v" > this constant is refused with a "run imgen upgrade" hint.
#
# v=2 (v0.5): added optional fields for LLM prompt enhancer recording.
# Forward-compat: v=1 entries do NOT carry these fields — readers must
# use ``entry.get(key, fallback)``. replay_entry treats absence of any
# enhance_* field as "enhancement was off" which is the correct
# historical interpretation for v0.4.x entries.
#
#   prompt_original (str): the pre-LLM prompt; equals ``prompt`` when
#                          enhancement was off or fell back to original
#   enhanced (bool):       True iff the LLM ran AND its output survived
#                          invariant checks (so ``prompt`` differs from
#                          ``prompt_original``)
#   enhance_model (str|null): which LLM was used, e.g.
#                          ``"mlx-community/Qwen2.5-7B-Instruct-4bit"``.
#                          Null when the LLM's output was discarded
#                          (invariant_violated / runner_error / empty
#                          output) — the value records WHAT MFLUX SAW
#                          (record-truth), not what the user INTENDED to
#                          enhance with (record-intent). Replay with
#                          ``--re-enhance`` reads ``enhance_model`` to
#                          decide which LLM to re-run; a null entry
#                          means "no enhancement actually happened in
#                          this run, so there's nothing to re-do".
#                          (v0.5 architect NIT #7 trade-off doc.)
#   enhance_fallback_reason (str|null): None when ``enhanced=True``;
#                          one of "user_opt_out" / "input_too_long"
#                          / "empty_input" / "empty_llm_output"
#                          / "invariant_violated" / "runner_error"
#                          / "not_supported_by_backend"
#   enhance_fallback_detail (str|null, v0.6.4+): verbose diagnostic
#                          string when the coarse fallback_reason
#                          token loses detail. Populated for
#                          "invariant_violated" (the check_invariants
#                          reason names which clause(s) the LLM dropped)
#                          and, since v0.6.5, for "runner_error" (the
#                          str(RunnerError) message — model-load trace,
#                          timeout, etc.). None for paths where the
#                          coarse token IS the full story. Read-
#                          compatible additive field; v=2/v=3 readers
#                          using ``entry.get`` see it as missing on
#                          entries written before v0.6.4.
#
#                          Note (v0.6.5): for v=3 history rows written
#                          by v0.6.0–v0.6.4, the runner-error message
#                          lived in ``raw_llm_output`` instead and
#                          this field is absent/null. A future
#                          ``imgen history --verbose`` reader should
#                          prefer this field first and fall back to
#                          ``raw_llm_output`` only when
#                          ``enhance_fallback_reason == "runner_error"``
#                          and this is null — see EnhanceResult
#                          docstring in enhance.py for the full rule.
#
# v=3 (v0.6): added optional ``loras`` field recording the LoRA stack
# mflux actually saw for the iteration. Architect-CRITICAL #1 from the
# v0.6 pre-tag review: without persistence, ``imgen replay`` silently
# diverged on LoRA selection (style's current built-in LoRAs got
# silently re-injected instead of the originally-applied stack, and
# original --lora / --no-lora opt-outs were lost).
#
#   loras (list[dict]):    list of LoraRef-shaped dicts, one per LoRA
#                          mflux saw on the iteration. Each dict has
#                          ``ref`` (str), ``weight`` (float),
#                          ``compatible_with`` (list[str]), and
#                          ``trigger`` (str|null). Empty list = no
#                          LoRAs ran (either text-only style or
#                          --no-lora opt-out). Absent (v<3 entries) =
#                          unknown, replay falls back to current
#                          style's LoRA mapping. Read-compatible
#                          additive migration; no rewrite pass.
#
# v=3 (v0.7.0): two additive fields lock-in t2i (`imgen draw`) shape:
#
#   command (str):         which subcommand produced the entry —
#                          "generate" | "batch" | "draw". Drives
#                          replay routing in commands/history.py
#                          (cmd_replay dispatches by this field).
#                          Absent on v0.6.x and earlier entries; the
#                          reader uses ``entry.get("command", "generate")``
#                          so old rows replay through cmd_generate.
#                          Read-compatible additive; NO v=4 bump
#                          (additive fields don't bump version per
#                          the v0.6.5 IMP-1 precedent).
#
#   input (str|null):      WIDENED from str-only to nullable.
#                          ``command="draw"`` entries have ``input=null``
#                          (t2i has no source photo). i2i entries
#                          (generate/batch) still carry the photo path
#                          unchanged. Readers using ``.get`` already
#                          tolerate absence; the None case is the
#                          v0.7.0 net change.
HISTORY_SCHEMA_VERSION = 3

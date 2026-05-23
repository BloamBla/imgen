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
    "backend": "flux",   # flux | qwen
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
HISTORY_SCHEMA_VERSION = 3

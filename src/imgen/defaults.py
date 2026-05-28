"""Default parameter values + resource thresholds + mflux pin."""
from __future__ import annotations

__all__ = [
    "DEFAULTS",
    "FULL_PRECISION_QUANTIZE",
    "HISTORY_SCHEMA_VERSION",
    "MFLUX_PIN",
    "MIN_BATTERY_PCT",
    "MIN_DISK_GB",
    "PREVIEW_OVERRIDES",
]

# v0.11.0: the user-facing "don't quantize" sentinel for mflux models.
# `--quantize 16` = full bf16 weights → ``build_mflux_cmd`` OMITS the
# ``--quantize`` argv entirely (mflux then loads native bf16). 16 is NOT
# an mflux quant level ({3,4,5,6,8}); it reads as "16-bit / full" and
# slots correctly into the RAM formula (weights = baseline * 16/8 = 2x Q8).
# Use it when a small model (e.g. flux2-klein-4b) has RAM headroom and you
# want max quality. ``MfluxEngine.validate`` accepts it on any mflux model
# regardless of ``supported_quants`` (full precision is always runnable).
FULL_PRECISION_QUANTIZE = 16
# v0.8.0 commit 8 (§L): RAM_REQUIRED_GB + ACTIVATION_GB_PER_MP_ABOVE_BASELINE
# DELETED. Per-Model RAM math moved into ``Model.ram_baseline_gb`` /
# ``Model.ram_slope_gb_per_mp`` / ``Model.encoder_ram_gb`` declared on
# each row of ``models.BUILTIN_MODELS``. Computation lives in
# ``Engine.ram_estimate_gb`` (single source-of-truth, exercised by
# both the preflight gate in ``checks.ram_required_gb`` and the
# doctor RAM table renderer).

DEFAULTS = {
    "style": "pixar",
    # v0.8.0 commit 5: key renamed `backend` → `model` + value
    # translated to v0.8 canonical form. Config schema accepts both
    # `[defaults] backend = ...` (DEPRECATED warn-and-bridge) and
    # `[defaults] model = ...` (preferred) through the v0.8.x
    # deprecation window; v0.9.0 drops the legacy key.
    "model": "flux-kontext",
    # v0.7.0 t2i default for `imgen draw`. Key unchanged at commit 5
    # — §J §Q scope only covers `backend` → `model`; `backend_draw`
    # stays its own config key. Architect HIGH-1 (4a pre-vet) tied
    # this back-compat to a real risk: silently auto-translating
    # `[defaults] backend_draw = "flux"` to `flux-kontext` would
    # replace one config wrong (FLUX.1-Kontext is i2i, not t2i) with
    # a different wrong.
    "backend_draw": "flux-dev",
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
#                          Read-compatible additive; NO v=4 bump at v0.7
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
#
# v=4 (v0.8.0 commit 9, §K + §Q): KEY RENAME — ``backend`` → ``model``.
# Not additive: the meaning of the "what was used to generate this row"
# slot moved from the v0.7 ``backend`` identifier ("flux", "flux-kontext",
# "qwen", ...) to the v0.8 ``model`` identifier ("flux-kontext",
# "flux-dev", ...). The two spaces overlap heavily but the rename is
# semantic — v0.8 ``flux-kontext`` was v0.7 ``flux`` — so a version
# bump is required to disambiguate readers.
#
# Dual-shape READ dispatch lives in ``history.entry_model_name(entry)``:
# the helper resolves either key (v=4 ``model`` wins; v=3 ``backend``
# fallback) and runs the value through the v0.7→v0.8 rename map so
# old rows render their v0.8 canonical name in list/replay/ETA paths.
# v0.9 may drop the v=3 ``backend`` key fallback after a deprecation
# window; until then, the helper is the single source of truth for
# resolving the model identifier from any history-entry shape.
HISTORY_SCHEMA_VERSION = 4

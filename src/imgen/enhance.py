"""LLM prompt enhancer — public facade re-exporting decide + runtime layers.

v0.7.16 split: this module is now a thin re-export layer over
:mod:`imgen.enhance_decide` (pure-function decision logic) and
:mod:`imgen.enhance_runtime` (subprocess runner + iteration-level
orchestrator). Pre-v0.7.16 the same content lived inline in a
784-LoC `enhance.py` that pushed the CLAUDE.md soft cap (800 LoC).
Split along the pure-vs-impure seam — matching the v0.5 architect's
"pure decision layer" framing in the original module docstring.

The split is back-compat by construction: every name previously
importable from ``imgen.enhance`` (``EnhanceResult``,
``decide_final_prompt``, ``run_with_mlx_lm``, etc.) re-exports from
the new submodules through the explicit ``__all__`` below. Callers
(cmd_helpers, tests) keep using ``from imgen.enhance import X``
unchanged.

Project rules (preserved across the split):

* Frozen+slots :class:`EnhanceResult` dataclass.
* Fail-safe: invariant violation / empty LLM output / oversized input
  all fall back to the ORIGINAL prompt with a diagnostic reason
  string. We never crash a generation because enhancement failed.
* Decision layer is GPU-free — 100% testable without mlx_lm /
  model weights installed (see :mod:`enhance_decide`).
"""
from __future__ import annotations

# Pure-function decision layer.
from .enhance_decide import (
    DEFAULT_ENHANCE_MODEL,
    DEFAULT_ENHANCE_TIMEOUT_S,
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    EnhanceResult,
    apply_length_cap,
    build_messages,
    check_invariants,
    decide_final_prompt,
    extract_enhanced_text,
    is_enhanceable,
    should_enhance,
)
# Subprocess runner + iteration-level orchestrator.
from .enhance_runtime import (
    RunnerError,
    build_runner_payload,
    enhance_iteration_prompts,
    parse_runner_response,
    replace_prompt_in_cmd,
    run_with_mlx_lm,
)

__all__ = [
    # Constants
    "DEFAULT_ENHANCE_MODEL",
    "DEFAULT_ENHANCE_TIMEOUT_S",
    "DEFAULT_MAX_INPUT_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    # Data + exceptions
    "EnhanceResult",
    "RunnerError",
    # Pure-decision API
    "apply_length_cap",
    "build_messages",
    "check_invariants",
    "decide_final_prompt",
    "extract_enhanced_text",
    "is_enhanceable",
    "should_enhance",
    # Runtime API
    "build_runner_payload",
    "enhance_iteration_prompts",
    "parse_runner_response",
    "replace_prompt_in_cmd",
    "run_with_mlx_lm",
]

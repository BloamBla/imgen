"""Subprocess entry point for the v0.5 LLM prompt enhancer.

Spawned by ``imgen.enhance.run_with_mlx_lm`` as
``python -m imgen.enhance_runner``. Reads a JSON payload from stdin,
loads an MLX-quantized LLM (Qwen2.5-7B-Instruct-4bit by default) ONCE,
generates enhanced prompts for every item in the batch, writes results
as JSON to stdout. Exits 0 on success, 1 on any failure (with an
``{"error": "..."}`` JSON object on stdout for the wrapper to surface).

Wire protocol (kept in sync with :mod:`imgen.enhance` —
:func:`build_runner_payload` + :func:`parse_runner_response`):

Request (stdin)::

    {
      "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
      "temperature": 0.0,
      "max_tokens": 200,
      "items": [
        {"system": "...kontext system prompt...",
         "user": "Restyle this person as anime ..."}
      ]
    }

Response (stdout, success)::

    {"results": [{"output": "Restyle this person as cel-shaded ..."}]}

Response (stdout, failure) + exit 1::

    {"error": "failed to load Qwen/...: <reason>"}

Memory model: this subprocess is short-lived. mlx_lm.load occupies
~4 GB unified memory for Qwen2.5-7B-4bit. When the subprocess exits
the kernel reclaims everything — no explicit ``del model`` needed.
Caller (imgen main process) doesn't see the LLM weights at all.

This file is impure by design (imports mlx_lm, prints to stderr on
error, exits with codes). The pure decision logic lives in
:mod:`imgen.enhance`. Tests for THIS module use a fake-script
fixture (see ``tests/test_enhance_runner.py``) — never invoke the
real mlx_lm in the test suite to avoid the 4 GB download.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _emit_error(message: str) -> None:
    """Write a single ``{"error": ...}`` object to stdout. Caller's
    parse_runner_response surfaces this as a clean RunnerError without
    needing to inspect stderr or exit code separately."""
    json.dump({"error": message}, sys.stdout)
    sys.stdout.flush()


def _build_chat_prompt(tokenizer: Any, system: str, user: str) -> str:
    """Render the {system, user} pair via the tokenizer's chat template.

    Qwen2.5-Instruct uses a specific chat template
    (``<|im_start|>system\\n...\\n<|im_end|>\\n<|im_start|>user\\n...``)
    that's baked into the tokenizer config. ``apply_chat_template``
    formats correctly across model families (Llama, Mistral, Qwen)
    without us hardcoding any template — works for ``--enhance-model``
    overrides too.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main() -> int:
    """Entry point. Returns process exit code."""
    # ── Parse stdin ─────────────────────────────────────────────────
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        _emit_error(f"invalid JSON on stdin: {e}")
        return 1

    try:
        model_name = payload["model"]
        items = payload["items"]
        temperature = float(payload.get("temperature", 0.0))
        max_tokens = int(payload.get("max_tokens", 200))
    except (KeyError, TypeError, ValueError) as e:
        _emit_error(f"malformed payload: {e}")
        return 1

    if not isinstance(items, list):
        _emit_error("'items' must be a list")
        return 1

    # ── Load model + sampler (lazy import so test discovery doesn't
    # need mlx_lm available) ────────────────────────────────────────
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        _emit_error(f"mlx_lm import failed (is it installed?): {e}")
        return 1

    try:
        model, tokenizer = load(model_name)
    except Exception as e:  # noqa: BLE001 — any load failure is fatal
        _emit_error(f"failed to load {model_name!r}: {type(e).__name__}: {e}")
        return 1

    # temp=0.0 → greedy decoding = deterministic = replay-friendly.
    # Same temperature applied to all items in the batch (per-item
    # override is a v0.6+ extension if needed).
    sampler = make_sampler(temp=temperature)

    # ── Generate per item ──────────────────────────────────────────
    results: list[dict[str, str]] = []
    for i, item in enumerate(items):
        try:
            system = str(item["system"])
            user = str(item["user"])
        except (KeyError, TypeError) as e:
            _emit_error(f"item {i} malformed: {e}")
            return 1

        try:
            prompt_str = _build_chat_prompt(tokenizer, system, user)
            # v0.5 python N-5: ``verbose=False`` is load-bearing — the
            # runner uses stdout for the JSON response payload, so any
            # progress prints from mlx_lm.generate would corrupt the
            # caller's JSON parse. Re-validate on every mflux pin bump
            # (mlx_lm is transitively pinned via mflux); a future mlx_lm
            # release that adds a non-verbose-suppressed log channel
            # (warnings, debug, etc.) would silently mix into our JSON
            # output and break parse_runner_response.
            output = generate(
                model, tokenizer,
                prompt=prompt_str,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001 — caller falls back per-item
            _emit_error(
                f"generation failed on item {i}: {type(e).__name__}: {e}"
            )
            return 1

        results.append({"output": output})

    # ── Emit results ───────────────────────────────────────────────
    json.dump({"results": results}, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess only
    sys.exit(main())

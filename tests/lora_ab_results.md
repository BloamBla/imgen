# LoRA A/B test results

Tracking ground truth for which LoRAs actually work on FLUX.1-Kontext-dev. Driven by the v0.6.0 → v0.6.1 burn: HF "Flux LoRA" labels do NOT imply Kontext compatibility. The base-FLUX.1-dev → Kontext attention layer shape mismatch only surfaces at the FIRST denoise step, AFTER all 912 LoRA keys "match" at load time. The only ground truth is real inference on a real photo.

Methodology below; data tables sit at the bottom of the file. Append rows as we test, never delete — losers are as valuable as winners for future researchers.

## Methodology

### Two-phase test per candidate

**Phase 1 — crash screen (automatable, machine-decidable):**

1. `huggingface-cli download <repo>` → confirms repo exists + downloads weights into HF cache.
2. Run `imgen <smoke.jpg> --style <base-style> --lora "<repo>:0.8" --backend flux --preview --dry-run` then without `--dry-run`. `--preview` uses Q4 + ~8 steps → ~3 min on M2 Pro 32GB → much cheaper than full Q8 for screening.
3. Capture exit code + stderr. CRASH (non-zero exit, matmul shape exception) → candidate is FLUX.1-dev base, unusable on Kontext. NON-CRASH → moves to Phase 2.

The smoke photo lives at `tests/fixtures/smoke_portrait.jpg` (NOT committed — user-supplied; methodology assumes a portrait shot, any aspect, real face, no synthetic content). Use the same fixture for every candidate so A/B comparisons share a reference frame.

`--preview` rationale: a LoRA that crashes Kontext crashes on the FIRST denoise step regardless of total step count. 8 steps catches every crash 50 steps would catch, at 1/6 the wall-clock. Once a LoRA survives `--preview`, the assumption is full quality runs will also survive — if a counter-example emerges, document it as a row in the failure table.

**Phase 2 — quality A/B (manual, user-decidable):**

For every Phase-1 survivor:

1. Generate full-quality variant with the LoRA (`--lora REF:WEIGHT --backend flux`, no `--preview`).
2. Generate text-only baseline of the same prompt + photo (`--no-lora --backend flux`).
3. Compare side-by-side. Judgement criteria, in order:
   - **Style fidelity** — does the LoRA produce its claimed visual school visibly more than text-only?
   - **Identity preservation** — face/hair/body anchors stay intact (Kontext's main job).
   - **Artifact rate** — extra limbs, garbled text, double faces, etc.
4. Verdict per LoRA: `WIN` (ship to BUILTIN_STYLES), `TIE` (LoRA adds no measurable value over text-only, drop), `LOSE` (LoRA degrades the output, drop + document why for future researchers).

The user does Phase 2 manually — automation can't judge "this looks more anime than that". Phase 1 results land here as data; Phase 2 verdicts land here as commits with embedded photo references in `~/Desktop/imgen/<ts>/`.

### Failure modes catalogued so far

| Symptom | Diagnosis | Action |
|---|---|---|
| `ValueError: [matmul] Last dimension of first input with shape (1,N,16) must match second to last dimension of second input with shape (64,3072)` at first denoise step | LoRA is FLUX.1-dev base; rank-16 weight deltas don't fit Kontext's modified attention shape. Filename often contains "FluxDev" or "Flux-Dev". | Mark as crash in Phase 1 table; do NOT add to BUILTIN_STYLES; user can still try ad-hoc via `--lora`. |
| `huggingface_hub.utils._errors.RepositoryNotFoundError` | Repo deleted or renamed upstream. | Skip; document upstream state in notes. |
| OOM during load | Q8 base + Q8 LoRA + system overhead overshoots 32GB on M2 Pro. Try Q4. | Document quant ceiling; not necessarily a Kontext-compat failure. |
| Generation succeeds but no visible LoRA influence on output | LoRA rank may be too low to overcome base model bias on Kontext, OR weight too low. Try weight = 1.0 then 1.2. | If still no effect at 1.2, mark as TIE in Phase 2. |

## Candidate list — v0.6.3 research round 2

Sourcing strategy: enumerate LoRAs with **explicit Kontext training** in name or HF tags, plus the Shakker-Labs Kontext line (largest known Kontext-trained collection). Random "Flux LoRA" picks from HF go in only as control samples to confirm the v0.6.0→v0.6.1 pattern holds.

### Anime / cartoon family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style` | 0.8 | _pending_ | _pending_ | _pending_ | Explicit Kontext-dev training; design memo noted this as a likely-survivor. |
| (more candidates land here during v0.6.3 research) | | | | | |

### Pixar / 3D-rendered family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| (candidates land here) | | | | | |

### Ghibli / watercolor family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `openfree/flux-chatgpt-ghibli-lora` | 0.8 | WORKS | WIN (shipped v0.6.0) | **WIN — shipped** | The only Phase-1 survivor from v0.6 design memo. Currently the sole built-in LoRA on `ghibli` style. License: `flux-1-dev-non-commercial-license`. |

### Van Gogh / impressionist / oil-paint family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `UmeAiRT/FLUX.1-dev-LoRA-Impressionism` | 0.6 | _pending_ | _pending_ | _pending_ | Filename contains FLUX.1-dev → high prior on crash. Test as a control. |
| (more candidates land here) | | | | | |

### Simpsons / flat-cartoon family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| (no known Kontext-trained Simpsons LoRA on HF as of 2026-05-23 — likely IP risk explains the gap; document any findings) | | | | | |

### Pencil sketch / monochrome family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| (FLUX base handles pencil sketches well from text-only prompt; LoRA may not add measurable value — test as TIE-candidates) | | | | | |

### Novel families (no current BUILTIN_STYLES analogue)

| Repo | Weight | Phase 1 | Phase 2 | Verdict | Notes |
|---|---|---|---|---|---|
| (cyberpunk / oil-paint / watercolor / vintage-photo candidates land here if they pass — would become NEW built-in styles in v0.6.3+) | | | | | |

### Known failures (control samples)

| Repo | Phase 1 result | Reason |
|---|---|---|
| `strangerzonehf/Flux-Animeo-v1-LoRA` | CRASH | FLUX.1-dev base. Shape (1,4992,16) × (64,3072) mismatch at first denoise step. v0.6.0 shipped, v0.6.1 reverted. |
| `prithivMLmods/Canopus-Pixar-3D-Flux-LoRA` | CRASH | Same; filename literally `Canopus-Pixar-3D-FluxDev-LoRA.safetensors`. |

## Update protocol

1. Each new candidate gets a row added to the appropriate family table BEFORE testing (Phase 1 & 2 = `_pending_`).
2. After Phase 1 run: edit the row to record `WORKS` / `CRASH` + a short stderr quote in Notes.
3. After Phase 2 A/B (only for Phase-1 survivors): user records `WIN` / `TIE` / `LOSE` + verdict justification.
4. If `WIN`: separate commit to `src/imgen/styles.py` adds the LoRA to `BUILTIN_STYLES[<style>]["loras"]`. Reference the row in the commit message.
5. Never delete rows — losers are documentation. Future researchers (including future-you) need to know which candidates were already disqualified.

## Why this file exists

Per the v0.6 design memo: A/B-gated promotion was a pre-ship requirement that v0.6.0 skipped under "ship first, A/B after" pressure. The skip caused the v0.6.0→v0.6.1 burn. This file is the discipline gate that prevents that pattern from repeating: nothing enters `BUILTIN_STYLES["loras"]` without a row here showing `WIN`.

See: `~/.claude/projects/-Users-stanislav-khromazhenkov-imgen/memory/feedback_kontext_lora_compat.md` for the original lesson.

# LoRA A/B test results

Tracking ground truth for which LoRAs actually work on FLUX.1-Kontext-dev. Driven by the v0.6.0 → v0.6.1 burn: HF "Flux LoRA" labels do NOT imply Kontext compatibility. The base-FLUX.1-dev → Kontext attention layer shape mismatch only surfaces at the FIRST denoise step, AFTER all 912 LoRA keys "match" at load time. The only ground truth is real inference on a real photo.

Methodology below; data tables sit at the bottom of the file. Append rows as we test, never delete — losers are as valuable as winners for future researchers.

## 🎯 v0.6.3 Phase 1 outcome (2026-05-23)

7 candidates smoke-tested at Q4 + 8-step `--preview` on M2 Pro 32GB. Reference photo: a single portrait at `~/Desktop/imgen/refs/<uuid>.jpeg` (author-supplied, not committed; pick your own portrait if reproducing). All Phase-1-passed outputs land at `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/` for visual A/B (Phase 2).

| # | LoRA | Built-in slot | Phase 1 | Wall-clock | PNG |
|---|---|---|---|---|---|
| 1 | `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style` | anime / simpsons | ✅ WORKS | 4m 8s | `01-flatcartoon-anime.png` |
| 2 | `Kontext-Style/3D_Chibi_lora` | pixar | ✅ WORKS | 4m 3s | `02-3dchibi-pixar.png` |
| 3 | `Kontext-Style/Oil_Painting_lora` | vangogh | ✅ WORKS | 4m 2s | `03-oilpainting-vangogh.png` |
| 4 | `prithivMLmods/Monochrome-Pencil` | (was) pencil | ❌ **CRASH** | — | — |
| 5 | `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style` | pencil (replacement) | ✅ WORKS | 3m 29s | `05-sketch-pencil.png` |
| 6 | `Kontext-Style/Irasutoya_lora` | anime (alt) | ✅ WORKS | 3m 40s | `06-irasutoya-anime.png` |
| 7 | `Kontext-Style/Poly_lora` | pixar (alt) | ✅ WORKS | 3m 41s | `07-poly-pixar.png` |

**Crash #4** reproduces the v0.6.0 Animeo/Canopus-Pixar signature exactly — `(1,4992,16) × (64,3072)` mismatch on first denoise step. Confirms the lesson: HF "Kontext" tagging is author self-declaration; verify by inference.

**Ship gate**: at least 1 verified candidate exists for every non-ghibli builtin (anime / pixar / simpsons / vangogh / pencil). ghibli stays on the existing `openfree/flux-chatgpt-ghibli-lora`. Phase 2 visual A/B was the user's call after looking at the PNG outputs.

**Phase 2 verdicts (resolved 2026-05-23, applied in `BUILTIN_STYLES`)**:

| Slot | Verdict | Notes |
|---|---|---|
| `anime` | **WIN** Shakker-Labs/Flat-Cartoon-Style | Primary pick; Irasutoya kept accessible under new `anime_alt` style |
| `anime_alt` (new) | **WIN** Kontext-Style/Irasutoya_lora | User wanted both LoRAs accessible — same prompt, alt LoRA |
| `pixar` | **WIN** Kontext-Style/Poly_lora | Primary pick; 3D_Chibi kept under new `pixar_alt` style |
| `pixar_alt` (new) | **WIN** Kontext-Style/3D_Chibi_lora | Same logic as anime_alt |
| `simpsons` | **LOSE** (no LoRA shipped) | Flat-Cartoon doesn't fit Simpsons-specific aesthetic; stays text-only |
| `vangogh` | **WIN** Kontext-Style/Oil_Painting_lora | — |
| `pencil` | **WIN** Shakker-Labs/Sketch-Style | Replaces v0.6.x text-only fallback |
| `ghibli` | unchanged | Already ships `openfree/flux-chatgpt-ghibli-lora` since v0.6.0 |

**License posture concern**: Kontext-Style org LoRAs (3D_Chibi, Poly, Oil_Painting, Irasutoya) don't publish license on model cards. Shakker-Labs LoRAs (Flat-Cartoon, Sketch) carry `flux-1-dev-non-commercial-license` — same NC tier as the FLUX-Kontext-dev base. Recommend documenting the Kontext-Style licenses as "unspecified — review before commercial use" in README LoRA section when adding them to BUILTIN_STYLES (matching the FLUX-NC blanket caveat from v0.6.2).

---

## Methodology

### Two-phase test per candidate

**Phase 1 — crash screen (automatable, machine-decidable):**

1. `huggingface-cli download <repo>` → confirms repo exists + downloads weights into HF cache.
2. Run `imgen <smoke.jpg> --style <base-style> --lora "<repo>:0.8" --backend flux --preview --yes --no-open` then optionally a `--dry-run` first. `--preview` uses Q4 + ~8 steps → ~3 min on M2 Pro 32GB → much cheaper than full Q8 for screening.
3. Capture exit code + stderr. CRASH (non-zero exit, matmul shape exception) → candidate is FLUX.1-dev base, unusable on Kontext. NON-CRASH → moves to Phase 2.

A reference photo lives at `~/Desktop/imgen/refs/<uuid>_1_105_c.jpeg` (NOT committed — user-supplied; methodology assumes a portrait shot, any aspect, real face, no synthetic content). Use the same fixture for every candidate so A/B comparisons share a reference frame.

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
| OOM during load | Q8 base + Q8 LoRA + system overhead overshoots 32GB on M2 Pro. Try Q4 / `--preview`. | Document quant ceiling; not necessarily a Kontext-compat failure. |
| Generation succeeds but no visible LoRA influence on output | LoRA rank may be too low to overcome base model bias on Kontext, OR weight too low. Try weight = 1.0 then 1.2. | If still no effect at 1.2, mark as TIE in Phase 2. |

## Candidate list — v0.6.3 research round (HF survey 2026-05-23)

Sourcing strategy: enumerate LoRAs with **explicit Kontext training** in name or HF tags. Two organisations dominate the Kontext-trained ecosystem:

- **Shakker-Labs** — 6 named `FLUX.1-Kontext-dev-LoRA-<Style>` repos (consistent naming convention)
- **Kontext-Style** — community org with 10+ named `<Style>_lora` repos, all Kontext-trained

Plus standalone Kontext-trained repos by other authors (prithivMLmods, etc.). Random "Flux LoRA" picks from HF go in only as control samples to confirm the v0.6.0→v0.6.1 pattern holds.

The list below is the Phase-1 batch. After crash-screen results land, Phase-1 survivors get added to the per-family tables below.

### Phase 1 candidates (to smoke-test)

| Repo | Family | Trigger (per upstream) | Source | DL | Likes |
|---|---|---|---|---|---|
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style` | anime / simpsons | — (style is in name) | Shakker-Labs | 334 | 12 |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style` | pencil | — | Shakker-Labs | 32 | 8 |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Illustration-Style` | anime alt | — | Shakker-Labs | 30 | 7 |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Pixel-Style` | NEW: pixel | — | Shakker-Labs | 58 | 10 |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Felt-Style` | NEW: felt | — | Shakker-Labs | 28 | 6 |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Bioluminescence-Style` | NEW: bioluminescence | — | Shakker-Labs | 42 | 2 |
| `Kontext-Style/Paper_Cutting_lora` | NEW: papercut | — | Kontext-Style | 126 | 7 |
| `Kontext-Style/LEGO_lora` | NEW: lego | — | Kontext-Style | 42 | 6 |
| `Kontext-Style/Oil_Painting_lora` | vangogh alt | — | Kontext-Style | 16 | 3 |
| `Kontext-Style/Chinese_Ink_lora` | NEW: chinese-ink | — | Kontext-Style | 30 | 5 |
| `Kontext-Style/Origami_lora` | NEW: origami | — | Kontext-Style | 18 | 4 |
| `Kontext-Style/Pop_Art_lora` | NEW: popart | — | Kontext-Style | 8 | 4 |
| `Kontext-Style/Poly_lora` | pixar alt (low-poly 3D) | — | Kontext-Style | 17 | 2 |
| `Kontext-Style/3D_Chibi_lora` | pixar alt (3D chibi) | — | Kontext-Style | 23 | 6 |
| `Kontext-Style/Irasutoya_lora` | anime alt (Japanese illustration) | — | Kontext-Style | 10 | 4 |
| `Kontext-Style/Fabric_lora` | NEW: fabric | — | Kontext-Style | 9 | 2 |
| `prithivMLmods/Monochrome-Pencil` | pencil | — (must check card) | prithivMLmods | 328 | 8 |

Triggers above are placeholders — each repo's actual trigger word lives on its HF README; Phase 1 should record the canonical trigger when downloading. Many Kontext-Style LoRAs activate on "in the style of <X>"-shape prompts rather than a single token.

Known anime/pixar gap: HF Kontext-tagged anime LoRAs are sparse (only `SakikoLab/Anime-Image-Purifier-Kontext-LoRA` — and that's a cleaner/restorer, not a stylizer). HF Kontext-tagged pixar LoRAs: zero. The realistic substitutions for our `anime`/`pixar` builtins are the Flat-Cartoon-Style + 3D_Chibi/Poly options above.

### Anime / cartoon family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style` | 0.8 | **WORKS** (2026-05-23) | _pending — user A/B_ | _pending_ | Phase 1 PASSED on Kontext: 4m 8s, 8/8 denoise steps, no crash. License: `flux-1-dev-non-commercial-license`. Upstream trigger: `Convert to a flat cartoon style while keeping the subject unchanged`. mflux load reports 304/684 keys matched, 380 unmatched (mflux key-name strictness; runtime denoise succeeded regardless). Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/01-flatcartoon-anime.png` for user A/B review. |
| `Kontext-Style/Irasutoya_lora` | 0.8 | **WORKS** (2026-05-23) | _pending — user A/B_ | _pending_ | Phase 1 PASSED on Kontext: 3m 40s, 8/8 denoise steps, no crash. 684/684 keys matched. Trigger: `Irasutoya style`. License unspecified. Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/06-irasutoya-anime.png`. Alternative to Flat-Cartoon for anime — Irasutoya is a Japanese flat-illustration collection, potentially more "anime"-flavoured than Flat-Cartoon's generic cartoon look. User picks the winner. |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Illustration-Style` | 0.8 | _not yet tested_ | — | — | Wider-net illustration — backup if both anime candidates lose Phase 2. |

### Pixar / 3D-rendered family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Kontext-Style/3D_Chibi_lora` | 0.8 | **WORKS** (2026-05-23) | _pending — user A/B_ | _pending_ | Phase 1 PASSED on Kontext: 4m 3s, 8/8 denoise steps, no crash. 684/684 keys matched (perfect). Trigger: `3D Chibi` (upstream recommends `"Turn this image into the 3D_Chibi style."`). License: NOT SPECIFIED on HF model card — must verify before commercial use OR add NC caveat in README. Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/02-3dchibi-pixar.png` for user A/B. |
| `Kontext-Style/Poly_lora` | 0.8 | **WORKS** (2026-05-23) | _pending — user A/B_ | _pending_ | Phase 1 PASSED on Kontext: 3m 41s, 8/8 denoise steps, no crash. Trigger: `Poly style`. License unspecified. Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/07-poly-pixar.png`. Alternative to 3D_Chibi for pixar — Poly is low-poly geometric (closer to Mario 64 than Toy Story), so visually distinct enough that user might prefer it over 3D_Chibi for a different aesthetic. User picks the winner. |

### Ghibli / watercolor family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `openfree/flux-chatgpt-ghibli-lora` | 0.8 | WORKS | WIN (shipped v0.6.0) | **WIN — shipped** | The only Phase-1 survivor from v0.6 design memo. Currently the sole built-in LoRA on `ghibli` style. License: `flux-1-dev-non-commercial-license`. |

### Van Gogh / impressionist / oil-paint family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Kontext-Style/Oil_Painting_lora` | 0.8 | **WORKS** (2026-05-23) | _pending_ | _pending_ | Phase 1 PASSED on Kontext: 4m 2s, 8/8 denoise steps, no crash. Trigger: `Oil Painting` (upstream recommends `"Turn this image into the Oil_Painting style."`). License unspecified. PNG 1.4 MB — rich texture output suggests strong stylistic effect. Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/03-oilpainting-vangogh.png` for user A/B. |
| `Kontext-Style/Chinese_Ink_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Asian-painting alt; not vangogh per se, but a distinct painterly school worth ranking as future built-in. Trigger: unspecified (likely "Chinese Ink"). License unspecified. |

### Simpsons / flat-cartoon family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style` | 0.8 | _pending — see Anime family_ | — | _pending_ | Same upstream LoRA as anime candidate; flat-cartoon style serves both. Phase-2 evaluation will determine if it fits "simpsons" aesthetic specifically. |

### Pencil sketch / monochrome family

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `prithivMLmods/Monochrome-Pencil` | 0.8 | **CRASH** (2026-05-23) | — | **LOSE** | Same `(1,4992,16) × (64,3072)` shape mismatch as v0.6.0 Animeo/Canopus-Pixar. Despite being tagged "FLUX.1-Kontext-dev" on HF with 328 downloads, this is a FLUX.1-dev base LoRA mislabeled as Kontext. v0.6.1 lesson repeats. Trigger (irrelevant now): `replicate the image as a pencil illustration, black and white, with sketch-like detailing`. License: `flux-1-dev-non-commercial-license`. DO NOT add to BUILTIN_STYLES. |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style` | 0.8 | **WORKS** (2026-05-23) | _pending — user A/B_ | _pending_ | Phase 1 PASSED on Kontext: 3m 29s, 8/8 denoise steps, no crash. PNG 1.4MB — rich detail suggests strong stylistic effect. Trigger: `convert real photos into sketches` (or short `sketch`). License: `flux-1-dev-non-commercial-license`. Output: `~/Desktop/imgen/v0.6.3-smoke-2026-05-23/05-sketch-pencil.png`. Replaces Monochrome-Pencil as the pencil candidate. Sketch is technically distinct from "graphite pencil" (sketch tends to be looser line art; pencil tends to be tighter cross-hatching), but visually adjacent enough that the user may prefer it over the text-only baseline. |

### Novel families (no current BUILTIN_STYLES analogue — candidates for new built-in styles)

| Repo | Weight | Phase 1 (crash?) | Phase 2 (quality?) | Verdict | Notes |
|---|---|---|---|---|---|
| `Kontext-Style/Paper_Cutting_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: paper-cut silhouette aesthetic. If WIN, adds a new `papercut` built-in style. |
| `Kontext-Style/LEGO_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: LEGO-brick rendering. Could be `lego` built-in. |
| `Kontext-Style/Pop_Art_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: pop art / Warhol aesthetic. Could be `popart` built-in. |
| `Kontext-Style/Origami_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: origami / folded-paper. Niche but distinct. |
| `Kontext-Style/Fabric_lora` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: fabric / textile texture overlay. |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Pixel-Style` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: pixel art / 8-bit aesthetic. Could be `pixel` built-in. |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Felt-Style` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: felt / plush texture. Niche. |
| `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Bioluminescence-Style` | 0.8 | _pending_ | _pending_ | _pending_ | Novel: bioluminescent glow / Avatar aesthetic. Distinct from anything in current builtins. |

### Known failures (control samples — DO NOT retry)

| Repo | Phase 1 result | Reason |
|---|---|---|
| `strangerzonehf/Flux-Animeo-v1-LoRA` | CRASH | FLUX.1-dev base. Shape (1,4992,16) × (64,3072) mismatch at first denoise step. v0.6.0 shipped, v0.6.1 reverted. |
| `prithivMLmods/Canopus-Pixar-3D-Flux-LoRA` | CRASH | Same; filename literally `Canopus-Pixar-3D-FluxDev-LoRA.safetensors`. |
| `prithivMLmods/Monochrome-Pencil` | CRASH | Same shape signature. 328 dl on HF and tagged as `FLUX.1-Kontext-dev` adapter, but inference crashes identically to the v0.6.0 controls. Lesson reinforced: HF "Kontext" tags do NOT guarantee Kontext attention shape. |

## Update protocol

1. Each new candidate gets a row added to the appropriate family table BEFORE testing (Phase 1 & 2 = `_pending_`).
2. After Phase 1 run: edit the row to record `WORKS` / `CRASH` + a short stderr quote in Notes.
3. After Phase 2 A/B (only for Phase-1 survivors): user records `WIN` / `TIE` / `LOSE` + verdict justification.
4. If `WIN`: separate commit to `src/imgen/styles.py` adds the LoRA to `BUILTIN_STYLES[<style>]["loras"]`. Reference the row in the commit message.
5. Never delete rows — losers are documentation. Future researchers (including future-you) need to know which candidates were already disqualified.

## 🎯 v0.7.4 flux-dev (t2i) A/B round (2026-05-23)

The [[feedback-kontext-lora-compat]] lesson EXTENDS to FLUX.1-dev t2i: not all "FLUX.1-dev base_model"-labelled LoRAs on HuggingFace actually load at inference time. **n=3 confirmation at runtime: 3 of 5 LoRAs crash with the same `(1,4608,16) × (64,3072)` shape signature** as the v0.6.0 / v0.6.3 Kontext crashes. Verify-by-inference discipline is mandatory across the entire FLUX family, not just Kontext.

Reference image: user-supplied baseline. Prompt: `"a samurai on a misty mountain at dawn"`. Seed: `1088118853` across all rows (pinned for apples-to-apples A/B on the LoRA dimension). Backend: `flux-dev` (FLUX.1-dev t2i). Quant Q8, 20 steps, 1024×1024, guidance 3.5.

Outputs in `~/Desktop/imgen/lora-ab-2026-05-23/`.

| # | LoRA | Trigger | Phase 1 | Notes |
|---|---|---|---|---|
| 00 | (baseline, no LoRA) | — | ✅ | Anchor for the A/B visual comparison |
| 01 | `Shakker-Labs/FLUX.1-dev-LoRA-add-details` | none | ✅ WORKS | Detail boost without trigger; same Shakker `flux-1-dev-non-commercial-license` |
| 02 | `XLabs-AI/flux-RealismLora` | none | ✅ WORKS | Pushes toward photo-realism; 15.6k DL/mo |
| 03 | `strangerzonehf/Flux-Super-Realism-LoRA` | `Super Realism` | ❌ **CRASH** | Same `(1,4608,16) × (64,3072)` shape signature. MIT license but irrelevant — doesn't load. |
| 04 | `prithivMLmods/Flux.1-Dev-LoRA-HDR-Realism` | `HDR` | ❌ **CRASH** | Same signature. Experimental repo (195 DL/mo) — author probably didn't test against latest mflux. |
| 05 | `Shakker-Labs/FLUX.1-dev-LoRA-blended-realistic-illustration` | none | ❌ **CRASH** | Same signature. Important data point: even same-author (Shakker) LoRAs have different rank/projection — `add-details` works, `blended-illustration` crashes. The compat fault isn't the publisher, it's the per-LoRA training shape. |

**Phase 2 verdict** (resolved 2026-05-23):

| # | LoRA | Verdict | Why |
|---|---|---|---|
| 00 | (baseline, no LoRA) | LOSE | Anchor — flatter, less crisp detail than the LoRA-stacked variants |
| 01 | `Shakker-Labs/FLUX.1-dev-LoRA-add-details` | **WIN** ✅ | User pick. Adds detail/crispness without shifting the aesthetic toward photo-realism — keeps the prompt's intended mood. Recommended in README's `imgen draw` Quick Start v0.7.7+ |
| 02 | `XLabs-AI/flux-RealismLora` | RUNNER-UP | Strong but pushes toward photo-realism harder than the prompt asked for; better suited when the user explicitly wants a photo aesthetic. Kept as a "try this alternative" mention |

**1024² blurry follow-up**: Phase 2 winner `add-details` adds visible crispness vs baseline at 1024², but the underlying 1024² ceiling itself still bottlenecks "looks sharp on a 4K screen". The natural fix is the v0.7.5+ `imgen refine` chain: `imgen draw "..." --lora Shakker-Labs/FLUX.1-dev-LoRA-add-details && imgen refine <output>`. v0.7.7 surfaces this chain via a post-success UX hint in `cmd_draw` so the next user doesn't hit the same friction.

**Lesson cross-link**: [[feedback-kontext-lora-compat]] originally said "HF Kontext-tag ≠ Kontext compat". Now generalised: **HF FLUX.1-dev-base label ≠ flux-dev t2i compat either.** The shape-mismatch class of bug spans the entire FLUX family; rank/projection compat is per-LoRA, not per-base-model.

---

## Why this file exists

Per the v0.6 design memo: A/B-gated promotion was a pre-ship requirement that v0.6.0 skipped under "ship first, A/B after" pressure. The skip caused the v0.6.0→v0.6.1 burn. This file is the discipline gate that prevents that pattern from repeating: nothing enters `BUILTIN_STYLES["loras"]` without a row here showing `WIN`.

See README's LoRA section and the v0.6.1 commit (`c60dfac`) for the original lesson — that commit captured the first occurrence of the Kontext-vs-FLUX.1-dev shape mismatch and reverted two built-in LoRA picks.

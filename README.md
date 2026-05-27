# imgen

Local image + video generation CLI for Apple Silicon Macs. Modes today: **`imgen draw`** generates images from text prompts (FLUX.1-dev), **`imgen refine`** upsamples an existing image at 1.5×/2× via FLUX.2-klein-edit (Hires-Fix pattern), **`imgen generate` / `imgen batch`** restyle photos (FLUX.1-Kontext, Qwen-Image-Edit), and **`imgen video`** (v0.9.0+) generates short clips from text prompts (LTX-Video via diffusers) — or, with **v0.9.3+**, animates a still image via `--image PATH` (image-to-video on the same LTX-Video checkpoint). Wraps [mflux](https://github.com/filipstrand/mflux) for the image side; the video side uses HuggingFace `diffusers` in a separate isolated venv. On-device via MLX / MPS, no cloud, no API keys outside HuggingFace's gated-repo token.

```bash
# Text-to-image (v0.7.0+)
imgen draw "a samurai on a misty mountain at dawn"            # FLUX.1-dev default
imgen draw "samurai" --enhance-prompt                         # LLM expands the brief into a rich prompt
imgen draw "samurai" --lora some/style-lora,other/detail:0.6  # stack LoRAs
imgen draw "samurai" --width 1280 --height 720                # custom aspect

# Recommended LoRA for crisper detail (v0.7.7 A/B verdict, see tests/lora_ab_results.md):
imgen draw "samurai" --lora Shakker-Labs/FLUX.1-dev-LoRA-add-details

# Hires-Fix refine (v0.7.5+) — upsample an existing image at 1.5x/2x
imgen draw "samurai" --num-iterations 5 --preview             # explore: 5 variants at 1024²
imgen refine ~/Desktop/imgen/<run>/samurai-3.png              # winner → polished 1536² (~49 min on M2 Pro 32GB)
imgen refine winner.png --scale 2                             # 1024² → 2048² (FLUX.2-klein native cap)
imgen refine winner.png --width 1920 --height 1080            # explicit dims (16-multiple rounding)

# Text-to-video (v0.9.0+, LTX-Video via diffusers — single-shot per call)
imgen video "a samurai walking through bamboo forest"         # canonical: 768×512 × 25 frames @ 24fps ≈ 1 sec
imgen video "cyberpunk city at night" --duration 2            # 2 sec, ceils to next valid 8k+1 frames (49 @ 24fps)
imgen video "abstract" --num-frames 33 --fps 24               # explicit frame count (alternative to --duration)
imgen video "test" --dry-run                                  # show dispatch shape without running

# Image-to-video (v0.9.3+, animate a still per a motion prompt)
imgen video "wind blows, slow camera push-in" --image samurai.png    # i2v: animate the still
imgen draw "samurai in fog" -o samurai.png                           # ... or chain: generate still + animate
  && imgen video "wind blows" --image samurai.png --duration 2       #     (pipeline composition)

# Photo restyle (v0.1+)
imgen photo.jpg --style anime                            # preset mode
imgen photo.jpg --style simpsons --preview               # ~3 min fast test
imgen photo.jpg --custom-prompt "Mona Lisa painting style"  # bare mode (no preset baggage)
imgen photo.jpg --style anime --enhance-prompt           # smarter prompts → better results
imgen photo.jpg --style anime --no-lora                  # A/B vs the built-in LoRA (every non-simpsons style ships one in v0.6.3)
imgen batch ~/Desktop/holiday --style anime,ghibli       # every photo in folder × every style
```

> **v0.8.0 BREAKING — migration from v0.7.x**
>
> `--backend` → `--model`. Shell scripts using `--backend` now hard-error with an actionable hint. Quick migration:
>
> ```text
> Old:  imgen photo.jpg --backend flux --quantize 8
> New:  imgen photo.jpg --model flux-kontext --quantize 8
> ```
>
> Some model names also changed (`flux` → `flux-kontext`, `qwen` → `qwen-image-edit-v1`). Run `imgen --list-models` for the current canonical names.
>
> User TOMLs in `~/.imgen/backends.d/` keep working in v0.8.x **and v0.9.x** but emit a DEPRECATED warn on every run. Run `imgen migrate-toml` to move them to `~/.imgen/models.d/`. The `backends.d/` fallback is targeted for hard-removal in v0.10.0 — earlier README claimed v0.9.0 drop, but the bridge survives the v0.9 arc for continuity. The helper also detects redundant files (where a user TOML's stem matches a new v0.8 built-in Model) and offers to delete them.
>
> `[defaults] style` in `~/.imgen/config.toml` — REMOVED. Pass `--style NAME` per invocation instead. (Soft-deprecated since v0.7.13.)
>
> `~/.imgen/history.jsonl` entries from v0.7 keep replaying — the schema bumped v=3 → v=4 with a `backend` → `model` key rename, but the reader handles both shapes transparently.

> **v0.7.13 behaviour change** (heads-up for upgraders): bare `imgen photo.jpg` (no `--style`, no `--custom-prompt`) now errors out with an actionable hint instead of silently falling back to the default style (pixar). Either pass `--style NAME` (preset mode) or `--custom-prompt TEXT` / `--prompt-file PATH` (bare mode — raw prompt, no preset baggage). The silent fallback caused a class of UX bugs (negative_prompt leaks into argv, etc.) that we've now closed. Pre-v0.7.13 `history.jsonl` entries replay correctly as bare-mode runs.

```bash
```

Every run creates a timestamped folder under `~/Desktop/imgen/` — e.g. `~/Desktop/imgen/2026-05-21-14-30-12/photo-pixar.png` (photo restyle) or `~/Desktop/imgen/2026-05-21-14-30-12/a-samurai-on-a-misty-mountain.png` (text-to-image, slug from the first 6 prompt-words). The result opens in Preview automatically. Change the parent with `--output-dir PATH` or pin an exact path with `--output FILE`.

**Multi-style:** pass `--style anime,ghibli,pixar` to generate N images from one input in a single run — all dropped into the same timestamped folder, named by `<input>-<style>.png`. Confirms with a `[y/N]` summary before starting (skip with `-y/--yes`).

**Batch a folder:** `imgen batch <dir>` applies M styles to every supported image directly under `<dir>` (non-recursive — `ls <dir>` ≈ what gets batched). Same flat output layout — all N×M results in one timestamped folder, named `<input>-<style>.png`. iPhone HEIC inputs are auto-converted via `sips` before mflux sees them. Confirm gate shows N inputs × M styles + ETA; skip with `-y/--yes`.

### Person or scene?

If your photo has a person in it, `imgen` always preserves their face, hair, body proportions, and pose — that's unconditional, regardless of style or scope. The `--scope` flag only controls what happens to the **background**:

- **`--scope scene`** (default) — restyle the background to match the chosen style. Anime backgrounds become hand-painted anime scenery, Pixar backgrounds become 3D-animated environments with cinematic lighting, Van Gogh backgrounds get impasto brushstrokes, etc. Every built-in style ships with a tuned background directive.
- **`--scope person`** — keep the background photorealistic and untouched. Useful when you want a stylized person against the real-world setting they were actually in.

Photos without people work fine too — the identity-preserving language doesn't apply to absent faces, and FLUX restyles the whole scene.

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4) — MLX does not support Intel
- **Python 3.12** (install: `brew install python@3.12`)
- **32 GB unified memory recommended** for full feature set; 16 GB works for `imgen generate` / `imgen draw` with `--quantize 4`. **`imgen refine` requires 32 GB** (real measurement Q4/1536² peaks at ~23 GB resident + compression — 16 GB Macs would OOM mid-inference). **`imgen video` (LTX-Video) needs ~17 GB available** for the canonical 768×512 × 25-frame envelope — close Chrome/IDE before running on 32 GB Macs.
- **~60 GB free disk** for image (FLUX + Qwen models combined ~80 GB cached); **add ~26 GB more for video** (LTX-Video weights).
- **HuggingFace account** (for FLUX Kontext — gated model, needs license acceptance). LTX-Video is **open** (no token, no license click-through).

## Install

Two ways — pick one.

### Option A: `bootstrap.sh` (recommended for first-time users)

```bash
git clone https://github.com/BloamBla/imgen ~/imgen
cd ~/imgen
./bootstrap.sh
```

The bootstrap script:
1. Verifies macOS + Apple Silicon + Python 3.12
2. Creates venv at `.venv/`
3. Installs the `imgen` package + dependencies (pinned mflux 0.17.5)
4. Adds shell alias (`zsh` / `bash` / `fish` auto-detected)
5. Prompts for HuggingFace token (optional — only needed for FLUX)
6. Asks whether to also opt into the **diffusers** venv at `.venv-diffusers/` (~8 GB: torch + diffusers + transformers). Needed for `imgen video` (LTX-Video) and any user-TOML model with `engine = "diffusers_mps"`. Skip if you only use the image path — opt in later via `IMGEN_INSTALL_DIFFUSERS=1 ./bootstrap.sh`.

### Option B: `pipx install` (for those who already use pipx)

```bash
pipx install git+https://github.com/BloamBla/imgen
imgen setup     # interactive: HF token + state dirs
```

`pipx` manages the venv for you and puts `imgen` directly in your PATH — no shell alias needed.

### After install (either option)

```bash
imgen doctor                                 # verify everything's wired
imgen photo.jpg --preview                    # first run downloads FLUX (~24 GB, ~30 min)
```

### Shared Macs and NFS homes

`imgen` is built for a single user on a personal Mac. State (`~/.imgen/`), token (`~/.imgen/hf_token`, chmod 600), and history (`~/.imgen/history.jsonl`, chmod 600) all live under `$HOME` — fine for one account, but:

- **Multiple macOS accounts on the same machine**: each account must run `bootstrap.sh` (or `pipx install`) separately. Don't share `~/imgen/.venv/` across users — venv binaries embed absolute paths, and `~/.imgen/` perms (0o700) intentionally lock other accounts out.
- **NFS-mounted `$HOME`**: `~/.cache/huggingface/` over NFS makes mmap of the ~24 GB FLUX weights glacial (every page fault hits the network). If your Mac mounts `$HOME` from a fileserver, set `HF_HOME=/Volumes/local-ssd/hf-cache` (or any local-disk path) before first run so weights cache locally.

## Styles

| Preset    | What you get |
|-----------|--------------|
| `pixar`   | Polished 3D animated character |
| `anime`   | Japanese cel-shaded anime |
| `simpsons`| Yellow skin, bold outlines, Matt Groening style |
| `ghibli`  | Soft watercolor Studio Ghibli |
| `vangogh` | Oil painting with impasto brushstrokes |
| `pencil`  | Detailed graphite sketch |

See full prompts with `imgen --list-styles`.

### User-defined styles

Drop `*.toml` files into `~/.imgen/styles.d/` (auto-created by `imgen setup`). Filename becomes the style name:

```toml
# ~/.imgen/styles.d/noir.toml — full style
prompt = "film noir, black and white, dramatic shadows, 1940s detective"
negative = "color, daylight, modern"
guidance = 4.5
strength = 0.65
```

```toml
# ~/.imgen/styles.d/punchy.toml — param-only preset (no prompt)
guidance = 5.5
strength = 0.7
```

Use a full style with `-s noir`. Use a param-only style by combining with `--custom-prompt`: `imgen photo.jpg -s punchy --custom-prompt "..."` — the style supplies the tuning, the CLI supplies the prompt.

**Adding to a preset with `--custom-prompt`.** With a full style, `--custom-prompt` AUGMENTS the style prompt — your text is appended as a trailing detail. `imgen photo.jpg -s anime,ghibli,pixar --custom-prompt "wearing a red kimono"` runs three generations, each keeping its style's tuned prompt with the kimono detail tacked on. Without an explicit `--style`, `--custom-prompt` is the whole prompt and the default style only contributes its tuning params.

**Background style for user styles.** Add an optional `scene_suffix = "..."` to your TOML to control what `--scope scene` does with the background. Without it, you get a generic "match the same artistic style" directive — fine for most cases but less precise than the built-in styles' tuned suffixes.

```toml
# ~/.imgen/styles.d/cyberpunk.toml — with a scene-mode background directive
prompt = "Restyle this person as cyberpunk, while preserving the facial identity, hairstyle, body proportions, and pose, with neon-lit profile and metallic accents"
scene_suffix = ", and transform the background into a neon-soaked dystopian cityscape with rain reflections and holographic billboards"
guidance = 4.0
strength = 0.6
```

If a user style's filename clashes with a built-in (e.g. `styles.d/anime.toml`), it gets registered as `anime_0001` (`_0002`, `_0003`, …) with a warning, and the built-in stays accessible as `anime`. Built-ins always win on name; the suffix mechanism makes overrides explicit.

## All commands

```bash
# Generation
imgen <photo> --style anime                    # preset mode (v0.7.13+: --style is explicit opt-in)
imgen <photo> --custom-prompt "..."            # bare mode (no preset baggage; --style omitted)
imgen <photo> --style anime,ghibli,pixar       # multi-style — M images into one timestamped folder (asks [y/N])
imgen <photo> --style anime,ghibli --yes       # multi-style, skip the confirm gate
imgen <photo> --output-dir ~/Pictures/runs     # change parent of the timestamped run folder
imgen <photo> -o explicit.png                  # bypass run-folder layout (mutex with --output-dir)
imgen <photo> --custom-prompt "..."            # with --style: augments preset; without --style: sole prompt
imgen <photo> -s anime,ghibli --custom-prompt "wearing a red kimono"  # shared addition across all styles
imgen <photo> --custom-prompt -                # ← read prompt from stdin (hidden from ps)
imgen <photo> --prompt-file ~/prompts/x.txt    # ← read prompt from file (hidden from ps)
imgen <photo> -s anime --preview               # fast mode (~3-10 min)
imgen <photo> -s anime                         # default --scope=scene — restyle whole image
imgen <photo> -s anime --scope person          # keep background photorealistic, restyle person only
imgen <photo> -s anime --enhance-prompt        # smarter prompts via local AI (see "Smart prompts" below)
imgen <photo> -s anime --no-lora               # A/B against the style's built-in LoRA (see "LoRAs" below)
imgen <photo> -s pencil --lora REF[:WEIGHT]    # attach an extra LoRA; REF = HF repo or local .safetensors
imgen <photo> --model qwen-image-edit-v1       # use Qwen Edit (no HF token needed)
imgen <photo> --force                          # skip resource preflight checks

# Refine — Hires-Fix upsample (v0.7.5+, FLUX.2-klein-edit-9b default backend)
imgen refine <input>                           # default --scale 1.5 (1024² → 1536²)
imgen refine <input> --scale 2                 # 2x → 2048² (FLUX.2-klein native ~4 MP cap)
imgen refine <input> --width 1920 --height 1080  # explicit dims (mutex with --scale)
imgen refine <input> --prompt "polished, ..."  # override the baked-in refine prompt
imgen refine <input> --strength 0.5            # higher = more refine, lower = more input-faithful (default 0.3)
imgen refine <input> --model flux-kontext      # fall back to FLUX.1-Kontext (capped at ~1.5K cleanly)
imgen refine <input> -o ~/Desktop/refined.png  # explicit output path

# Video — text-to-video (v0.9.0+, LTX-Video default model, diffusers_mps engine)
imgen video "samurai walking"                  # canonical: 768×512 × 25 frames @ 24fps ≈ 1 sec
imgen video "..." --duration 2                 # seconds; ceils to next 8k+1 alignment
imgen video "..." --num-frames 33              # explicit frame count (mutex with --duration)
imgen video "..." --fps 30                     # {24, 25, 30} allowlist
imgen video "..." --width 1024 --height 576    # heavy mode (~19 GB; close other apps)
imgen video "..." --seed 42                    # reproducible
imgen video "..." -o ~/clip.mp4                # .mp4 extension required
imgen video "..." --dry-run                    # show dispatch payload without running

# Video — image-to-video (v0.9.3+, conditioning still + motion prompt)
imgen video "wind blows softly" --image still.png             # animate the still per the prompt
imgen video "..." --image still.png --guidance 7              # stronger motion-prompt adherence (default 5 for i2v)
imgen video "..." --image still.png --negative-prompt "blur"  # override the motion-anchor negative

# Batch a folder — same flags as generate except no -o/--output (always run-folder layout)
imgen batch <dir>                              # every photo × default style → one timestamped folder
imgen batch <dir> --style anime,ghibli,pixar   # N × M into one folder, named <input>-<style>.png
imgen batch <dir> -s anime --custom-prompt "..." # shared --custom-prompt augmentation across every input
imgen batch <dir> --style anime --yes          # skip the N×M confirm gate
imgen batch <dir> --enhance-prompt             # smart-prompt the whole batch (one model load, all images)
imgen batch <dir> --dry-run                    # show every mflux command without running

# Diagnostics
imgen doctor                                   # env + RAM forecast + cached models + backends + enhancer
imgen --list-styles                            # show presets
imgen --list-models                            # show built-in + user models from ~/.imgen/models.d/ + ~/.imgen/backends.d/
imgen --list-loras                             # show LoRAs each style ships with + HF cache state
imgen --dry-run <photo> -s anime               # show mflux command, don't run
imgen -v   /   imgen --version                 # print version

# Maintenance
imgen setup                                    # rerun setup (e.g. fix token)
imgen upgrade                                  # bootstrap: git pull + pip install -e . + mflux refresh
                                               # pipx: prints `pipx upgrade imgen` hint
imgen upgrade --latest                         # newest mflux (risky)
imgen clean                                    # delete stale partial downloads
imgen clean --all                              # delete cached models (with confirmation)

# History
imgen history                                  # last 20 generations
imgen history --last 50                        # more
imgen last                                     # repeat last with new seed
imgen replay 42                                # repeat #42
```

## Text-to-video (v0.9.0+)

`imgen video` generates short clips from text prompts via **LTX-Video** (Lightricks, ~2B-param DiT + ~5B-param T5-XXL text encoder) using HuggingFace `diffusers` on Apple Silicon's MPS backend. Outputs `.mp4` (libx264 via bundled ffmpeg). Single-shot per call in v0.9.0 — `--num-iterations N` is deferred to v0.9.x.

```bash
imgen video "a samurai walking through bamboo forest"   # canonical envelope
```

### Realistic envelope on M2 Pro 32 GB

LTX-Video's RAM math — peak RSS depends on resolution + frame count:

| Mode | Resolution × frames | RAM peak | Verdict on M2 Pro 32 GB |
|---|---|---|---|
| **Canonical** | 768×512 × 25 frames (~1 sec @ 24 fps) | ~17 GB | ✅ realistic with `available_gb >= 20` (close Chrome/IDE first) |
| **Heavy** | 1024×576 × 33 frames (~1.4 sec) | ~19 GB | ⚠️ tight; pass `--force` if preflight fails on 22 GB available |
| **Out of envelope** | 1280×720 × 121 frames (~5 sec) | ~29 GB | ❌ infeasible on 32 GB |

Check your current envelope with `imgen doctor` — the RAM forecast table includes a video section showing fits/tight/no per row.

### First-call lazy install

`imgen video` depends on three extra pip packages (`imageio==2.37.3`, `imageio-ffmpeg==0.6.0`, `sentencepiece==0.2.1`) that aren't part of the default `.venv-diffusers/`. The first `imgen video` invocation prompts to install them; subsequent calls return silently. For non-interactive contexts (cron, CI), set `IMGEN_INSTALL_VIDEO_DEPS=1` to opt in without prompting — an audit line still prints to stderr (never silent per security review §R.1).

### Architecture notes

- **Separate venv**: LTX runs in `.venv-diffusers/` (torch + diffusers + transformers + the three pinned video packages), kept isolated from the main `.venv/` to avoid the torch ↔ MLX dependency conflict.
- **CPU offload mandatory** on 32 GB Macs: LTX's T5-XXL encoder transient peaks ~18 GB if loaded to MPS directly; `force_cpu_offload=True` keeps the encoder on CPU and streams to MPS — peak drops to ~3 GB encoder + 14 GB transformer/activations.
- **8k+1 frame alignment**: LTX's temporal-VAE patch structure requires frame counts that satisfy `(n - 1) % 8 == 0` (1, 9, 17, 25, 33, 49, 65, ...). `--duration N` ceils UP to the nearest valid count so output is always ≥ requested duration.
- **History replay**: video entries roundtrip via `imgen replay <id>` — the `command="video"` discriminator + fields (`num_frames`, `fps`, `video_codec`, plus v0.9.3 `image_path` on i2v rows) flow through the existing history schema (v=4, additive — no migration). i2v rows replay back into i2v if the conditioning still still exists; if the file was moved/renamed the replay dies cleanly with the standard recovery hint.

## Image-to-video (v0.9.3+)

`imgen video --image PATH` animates an existing still per a motion-direction prompt. Same LTX-Video checkpoint as the v0.9.0 text-to-video default, different `diffusers` pipeline class (`LTXImageToVideoPipeline` instead of `LTXPipeline`) — no extra HuggingFace download.

```bash
# Conditioning still + motion-direction prompt → animated clip.
imgen video "wind blows softly through the bamboo" --image samurai.png

# Pipeline composition: generate the still with imgen draw, then animate.
imgen draw "samurai standing in fog, mountains behind" -o ~/samurai.png
imgen video "slow camera push-in, mist swirls" --image ~/samurai.png --duration 2
```

### Motion-prompt UX

LTX i2v on default `guidance=3` produces conservative motion (often a static-shimmer-only output on ambient prompts). When `--image` is set, `imgen video` bumps the default `guidance` to **5** and injects a motion-aware negative prompt (`"static, still, frozen, no motion"`) — both are overridable via `--guidance N` and `--negative-prompt "..."`.

Tips that work in practice:

- **Subject mid-action stills animate better than landscape stills.** A figure walking, leaning, or holding an object has implicit motion candidates LTX can amplify. A wide mountain vista without a focal subject often just shimmers.
- **Motion verbs in the prompt** ("walking", "drifting", "wind blows", "camera pans right") give LTX explicit direction. Style adjectives ("cinematic", "moody") don't carry motion intent.
- **Longer is more dynamic.** `--duration 2` (49 frames @ 24fps) gives the model more space to develop motion than the default 1-second / 25-frame canonical envelope.
- **Override `--guidance`** if the default 5 still feels frozen — try 6 or 7 for ambient prompts. Going above 8 starts to introduce artefacts.

### Supported input formats

`.png`, `.jpg`, `.jpeg` — verified through the LTX VAE encode path. `.webp` and `.heic` are refused (untested). Use `imgen draw` to produce a dim-matched still, or `sips -s format png in.heic --out out.png` to convert HEIC sources.

### History + replay

i2v rows store the conditioning path in an additive `image_path` field (no schema bump from v=4). `imgen replay <id>` reconstructs the run including the still; if the file has moved or been deleted, the replay dies cleanly with the standard "file not found" hint rather than silently mis-routing.

### Research-evaluated alternatives (not built-in)

Tested in 2026-05 research session on M2 Pro 32 GB. Findings descriptive below.

- **LTX-Video i2v** — promoted to first-class via `imgen video --image PATH` at v0.9.3 (see section above). Original research findings: proven on 512²×25f at ~3 min wall and 768²×25f at ~8 min wall on M2 Pro 32 GB; memory comfortable (~16 GB process RSS at 768²); conservative motion at default cfg=3 — v0.9.3 bumps i2v default to cfg=5 to mitigate.
- **Lance-3B-Video** (xocialize/lance-mlx, MLX-native t2v): painterly cinematic aesthetic by design (NOT photorealistic). Per-frame visual density at 384²+ is substantially higher than LTX 2B at comparable settings; per-call wall is ~2-4× longer. GPU utilization 80-97% (MLX-native = no MPS heap copies, no double-counting with system RAM). Strong candidate for v0.9.x second built-in t2v model; would need separate `.venv-lance/` + `MlxLanceEngine`.
- **CogVideoX-5b** (THUDM, diffusers + sequential CPU offload): highest single-frame photoreal density of the three but **bandwidth-bound by sequential offload** (~0% GPU utilization, ~23% CPU; ~19 min wall for 480×320×17f). Native 720×480×49f OOMs at 56 GiB attention buffer on 32 GB. Workable only at reduced params with offload; LTX is 3-5× faster for comparable output length on our hardware.

**TL;DR**: LTX-Video 2B remains the right default for v0.9.0 on M2 Pro 32 GB. Lance and CogVideoX are queued for v0.9.x as alternative built-ins. LTX 13B FP8 variant does NOT fit M2 Pro 32 GB (Apple Silicon has no native FP8 per public Metal docs → bf16 dequant pushes transformer alone to ~31 GB).

## Keeping prompts out of `ps`

Anything passed as `--custom-prompt "<text>"` lands in your shell's process arguments and is visible to other local users via `ps auxww`. For prompts you don't want exposed, use either:

```bash
# From a file (chmod 600 the file if it has secrets)
imgen photo.jpg --prompt-file ~/.imgen/private-prompt.txt -s anime --scope person

# From stdin — works with pipes, heredocs, pbpaste
echo "private prompt" | imgen photo.jpg --custom-prompt - -s anime
imgen photo.jpg --custom-prompt - <<< "$PROMPT"
pbpaste | imgen photo.jpg --custom-prompt -
```

Both paths cap at 64 KB; missing file, empty content, or specifying both `--custom-prompt` and `--prompt-file` fail early with a clear message. The effective prompt (not the source file path or "-") is what gets stored in `~/.imgen/history.jsonl` so `imgen replay <id>` reproduces the actual text.

### Don't paste `--dry-run` output into a shell

The pretty-printed command from `--dry-run` is for **reading**, not for re-executing. The quoting is structurally correct (shlex.quote — `$()`, backticks, newlines, semicolons all neutralized), so the line **would** run, but:

- It shows what mflux receives — including the **resolved** `--custom-prompt` text (so `--custom-prompt -` from stdin is displayed as the actual prompt, not as `-`) and the full mflux binary path — not the flags you originally typed.
- Pasting bypasses `imgen`'s preflight (memory forecast, disk check, mflux liveness) and runs mflux directly.

If you want to re-run with the same args, re-invoke `imgen` with the same flags rather than copy-pasting the displayed line.

## Tuning

| Flag                | Range  | Effect |
|---------------------|--------|--------|
| `--steps`           | 1-200  | More = better, slower. Sweet spot 15-30 |
| `--guidance` / `-g` | 0-15   | How strictly to follow prompt. 3.5-4.5 (0 = no CFG, for distilled models) |
| `--strength`        | 0-1    | How much to keep from original. 0.5-0.7 |
| `--quantize` / `-q` | 3,4,5,6,8 | Lower = smaller/faster, more artifacts |
| `--preview` / `-p`  | flag   | Q4, 8 steps, 768x — ~5x faster |
| `--scope`           | person/scene | default `scene` — restyle whole image with identity preserved; `person` keeps background unchanged |
| `--enhance-prompt`  | flag         | Expand the prompt via local AI before generating. See "Smart prompts" below |
| `--lora`            | REF[:WEIGHT] | Attach a LoRA weight delta on top of the style's stack (repeatable). See "LoRAs" below |
| `--no-lora`         | flag         | Drop the style's built-in LoRA stack — run the base model only |

For 32 GB Macs, **Q8** is recommended for FLUX Kontext.

## Smart prompts (`--enhance-prompt`)

Diffusion models produce noticeably better images when you give them rich, descriptive prompts. Three words like "wearing red kimono" work, but a sentence like "wearing an elegant red silk kimono with traditional floral patterns and an ornate obi sash, soft golden afternoon light" works **much** better.

Writing that level of detail by hand for every generation is tedious. Pass `--enhance-prompt` to let a small local AI model do it for you:

```bash
imgen photo.jpg --style anime --custom-prompt "wearing red kimono" --enhance-prompt
```

What happens:

1. `imgen` builds the usual prompt from your style preset + `--custom-prompt` + scope directive.
2. A local language model (Qwen2.5-7B-Instruct, ~4 GB, MLX-native) takes that prompt and expands it into a richer version — adding stylistic detail, lighting, materials, mood. Your face / identity / pose anchors stay verbatim; only descriptors get embellished.
3. The expanded prompt goes to FLUX, you get a better image.

**Real example.** With the input above, the unaltered prompt sent to FLUX is:

> *Restyle this person as a Japanese anime character, while preserving the facial identity ..., with cel-shaded illustration, ..., wearing red kimono*

The AI-enhanced version becomes:

> *Restyle this person as a Japanese anime character, while preserving the facial identity, exact facial features, and recognizable expression, with cel-shaded illustration, expressive large eyes, detailed line art, vibrant colors, clean shading, and manga aesthetic, and transform the background and surroundings into a hand-painted anime cel-shaded environment with vibrant skies, soft cloud shapes, and detailed illustrated scenery, **wearing a red kimono with intricate patterns and traditional Japanese motifs**.*

Same intent, much more for FLUX to work with.

### First-run download

The first time you pass `--enhance-prompt`, mlx-lm downloads Qwen2.5-7B-Instruct-4bit from HuggingFace into `~/.cache/huggingface/hub/` (~4 GB, ~5-10 minutes depending on connection). The download is one-time. `imgen doctor` reports whether the model is already cached.

If you've already got an HF token (you should, for FLUX), set `HF_TOKEN` in your shell before the first run for a much faster authenticated download:

```bash
export HF_TOKEN=$(cat ~/.imgen/hf_token)
imgen photo.jpg --style anime --enhance-prompt   # first run downloads ~4 GB
```

### Wall-clock cost

After the one-time download, the AI adds ~3-5 seconds per image to the FLUX wall-clock. Negligible on a 15-minute Q8 generation. For an `imgen batch` of 20 photos × 3 styles = 60 outputs, the AI loads **once** for the whole batch and amortises across all 60 prompts.

`--dry-run` with `--enhance-prompt` STILL runs the enhancer so the displayed mflux command matches what would actually execute. For a large batch this is non-trivial cost (~5-10 s cold load + ~3 s × N×M prompts) just to see the cmd printed. If you're sanity-checking the cmd structure and don't care about the enhanced prompt, drop `--enhance-prompt` (or pass `--no-enhance`) from the dry-run invocation.

### Determinism

At the default `temperature=0.0`, the expansion is deterministic: same input prompt + same AI model → same expanded prompt. This means `imgen replay <id>` reproduces the original image bit-for-bit when both runs use `--enhance-prompt`. If you want creative variation, pass `--enhance-temperature 0.5` (or higher).

### Opt-out

`--enhance-prompt` is opt-in by default. Add `[enhance] default = true` to `~/.imgen/config.toml` to make it default-on, and pass `--no-enhance` to disable for one run.

### When the AI hallucinates

The enhancer's system prompt explicitly forbids dropping identity-anchor language ("preserving the facial identity ..."), and `imgen` validates the AI's output against that anchor. If the AI ignores the instruction and drops it, `imgen` falls back to your original prompt and continues — you always get an image. `imgen doctor` reports the recent success rate.

### Override the AI model

Want to try a smaller / larger / different LLM?

```bash
imgen photo.jpg -s anime --enhance-prompt --enhance-model "mlx-community/Qwen2.5-3B-Instruct-4bit"
```

Or persist it in `~/.imgen/config.toml`:

```toml
[enhance]
default = false                                       # opt-in per run (default) or true for default-on
model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
temperature = 0.0
max_tokens = 200
timeout_s = 120
```

Smaller models (3B) are faster but produce flatter expansions. Larger (14B) won't fit alongside FLUX on 32 GB. The 7B 4-bit default is the sweet spot for M2/M3 Macs.

## LoRAs (`--lora`)

A LoRA is a small weight delta — typically 50-300 MB — trained on top of a base diffusion model to nudge it toward a specific style. Used at generation time, a LoRA *layers* on top of the base model, no full re-training, no need to swap models between styles.

Six built-in styles ship with curated LoRAs after v0.6.3 (research round 2 on Kontext-trained candidates):

| Style       | LoRA                                                         | Weight | Trigger             | License                                |
|-------------|--------------------------------------------------------------|--------|---------------------|----------------------------------------|
| `anime`     | `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style`    | 0.8    | `flat cartoon style`| non-commercial (see model card)        |
| `anime_alt` | `Kontext-Style/Irasutoya_lora`                               | 0.8    | `Irasutoya style`   | unspecified — see commercial-use note  |
| `pixar`     | `Kontext-Style/Poly_lora`                                    | 0.8    | `Poly style`        | unspecified — see commercial-use note  |
| `pixar_alt` | `Kontext-Style/3D_Chibi_lora`                                | 0.8    | `3D Chibi`          | unspecified — see commercial-use note  |
| `ghibli`    | `openfree/flux-chatgpt-ghibli-lora`                          | 0.8    | `Ghibli style`      | `flux-1-dev-non-commercial-license`    |
| `vangogh`   | `Kontext-Style/Oil_Painting_lora`                            | 0.8    | `Oil Painting`      | unspecified — see commercial-use note  |
| `pencil`    | `Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style`          | 0.8    | `sketch`            | non-commercial (see model card)        |

`simpsons` stays text-only — no Kontext-trained Simpsons LoRA on HF survived the visual A/B for that specific aesthetic.

`anime_alt` and `pixar_alt` exist because the same input photo lands differently under different LoRAs of the same family. Try the primary first; switch to `_alt` if you want the other aesthetic (Irasutoya = flatter Japanese illustration; 3D_Chibi = exaggerated chibi proportions).

> **Note on commercial use.** All built-in LoRAs above ride on top of FLUX.1-Kontext-dev, whose own license (FLUX-NC) gates commercial use. So even when an upstream LoRA's own license is permissive, running it through `imgen` inherits FLUX-NC restrictions. The `flux-1-dev-non-commercial-license` LoRAs (Shakker-Labs + openfree) are identical in spirit to FLUX-NC — non-commercial only. **Kontext-Style org LoRAs don't publish a license on their model cards** — `imgen` treats them as "unspecified, review before commercial use" but their FLUX-NC base inheritance means commercial use is gated regardless. The same FLUX-NC constraint applies to any LoRA you supply via `--lora` or `styles.d/*.toml`.

**Why is the table this short?** Most HuggingFace "Flux LoRA" weights were trained on **FLUX.1-dev** (the base text-to-image model). `imgen`'s default backend is **FLUX.1-Kontext-dev** — a different model in the same family, with a modified attention layer to accept image conditioning. Many FLUX.1-dev LoRAs *load* on Kontext (all tensor keys match) but **crash at the first denoise step** with an attention shape mismatch, because the rank-16 weight deltas don't fit Kontext's attention shape. v0.6.0 originally shipped LoRA mappings for `anime` (Flux-Animeo) and `pixar` (Canopus-Pixar-3D-FluxDev); v0.6.1 reverted both to text-only after they crashed in real runs. v0.6.3 went hunting for actual Kontext-trained replacements; the ones above are what survived a Phase-1 crash-screen against a real photo. **Even HF "Kontext"-tagged repos are not always Kontext-compat** — `prithivMLmods/Monochrome-Pencil` (328 dl, prominent) crashed with the same shape signature as the v0.6.0 controls. Per-LoRA Kontext compatibility must be verified by actual inference — name, tag, and key-match are not enough.

`imgen --list-loras` shows the active mapping plus which LoRAs are already cached locally vs. about to download:

```
$ imgen --list-loras
Available LoRAs
  Styles shipping LoRAs:
    anime          Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style @0.80  [flux-1] trigger="flat cartoon style"  (cached)
    anime_alt      Kontext-Style/Irasutoya_lora                            @0.80  [flux-1] trigger="Irasutoya style"     (cached)
    ghibli         openfree/flux-chatgpt-ghibli-lora                       @0.80  [flux-1] trigger="Ghibli style"        (cached)
    pencil         Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Sketch-Style       @0.80  [flux-1] trigger="sketch"               (cached)
    pixar          Kontext-Style/Poly_lora                                 @0.80  [flux-1] trigger="Poly style"           (cached)
    pixar_alt      Kontext-Style/3D_Chibi_lora                             @0.80  [flux-1] trigger="3D Chibi"             (cached)
    vangogh        Kontext-Style/Oil_Painting_lora                         @0.80  [flux-1] trigger="Oil Painting"         (cached)
  Text-only styles (no LoRA): simpsons
```

### A/B against the base model

To compare the LoRA-flavoured ghibli against the text-only baseline:

```bash
imgen photo.jpg --style ghibli                  # built-in: with the openfree-ghibli LoRA
imgen photo.jpg --style ghibli --no-lora        # text-only baseline (no LoRA)
```

Both runs write into the same timestamped `~/Desktop/imgen/<ts>/` folder, named `<input>-ghibli.png` — you'll have to move or rename the first before launching the second. Quick A/B without renaming:

```bash
imgen photo.jpg --style ghibli --output ~/Desktop/ghibli-with-lora.png
imgen photo.jpg --style ghibli --no-lora --output ~/Desktop/ghibli-text-only.png
```

### Attach an ad-hoc LoRA

Layer an additional LoRA on top of the style's stack (or any style's stack — including text-only ones):

```bash
imgen photo.jpg --style anime --lora "alvarobartt/flux-watercolor-lora:0.6"
imgen photo.jpg --style pencil --lora "/Users/me/loras/sketch-extra.safetensors:0.5"
```

`--lora REF[:WEIGHT]` is repeatable — pass it multiple times to stack:

```bash
imgen photo.jpg --style anime \
    --lora "strangerzonehf/Flux-Animeo-v1-LoRA:0.8" \
    --lora "Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style:0.3"
```

Style-declared LoRAs come first in argv, your CLI additions are appended after — order matters for some LoRA combinations, but rarely.

`REF` is either a HuggingFace repo id (`author/name`) or an absolute path to a local `.safetensors` file. mflux accepts both. Optional `:WEIGHT` is a float; 1.0 is full strength. The colon split is rightmost-only so paths with embedded colons (e.g. macOS Time Machine snapshot paths) parse correctly.

> **A note on shared logs.** Local `.safetensors` paths can disclose your `$HOME` layout to anyone you share batch logs / dry-run output with. `imgen` rewrites `$HOME` → `~` in its own rendered command output (dry-run + confirm-gate transcripts) and in `mflux`'s captured stderr — so a `/Users/me/loras/foo.safetensors` shows up as `~/loras/foo.safetensors` in logs. The subprocess itself still receives the real absolute path. If you redirect output through other tools or post-process logs externally, that rewrite doesn't follow — verify before sharing.
>
> The rewrite means a `~/foo` in a shared transcript is **your** `~`, not the reader's — recipients should treat `~` as a placeholder for the original author's home directory, not literally expand to their own `$HOME`. Same trade-off as token redaction: shared output prioritises "don't disclose" over "literally re-runnable on another machine".

### Trigger words

Many style LoRAs only activate when a specific phrase appears in the prompt — that phrase was used to label the LoRA's training set, and the model learnt to associate the weight delta with that token. `imgen` auto-prepends each LoRA's trigger to the prompt if it's not already there. For example:

```bash
imgen photo.jpg --style anime
# Effective prompt: "flat cartoon style, Restyle this person as a Japanese anime character, ..."
```

Triggers shown in `imgen --list-loras` (the `trigger=` column). If you set `--custom-prompt` and your custom text already includes the trigger phrase, `imgen` leaves it alone — no duplication. If you stack multiple LoRAs with different triggers, all triggers are prepended in stack order.

### Persist a custom LoRA stack via styles.d

If you want a LoRA stack always-on for one style without typing `--lora` every time, declare it in a user style TOML at `~/.imgen/styles.d/`. The shape mirrors the built-in dict — see [User-defined styles](#user-defined-styles) above. Example:

```toml
# ~/.imgen/styles.d/anime_strong.toml — anime with two Kontext-trained LoRAs stacked
prompt = "Restyle this person as a Japanese anime character, while preserving the facial identity, hairstyle, body proportions, and pose, with cel-shaded illustration, vibrant colors, clean shading, and manga aesthetic"
negative = "realistic photo, 3d render, deformed face, bad anatomy, blurry, watermark, text"
guidance = 4.0
strength = 0.60

[[loras]]
ref = "Shakker-Labs/FLUX.1-Kontext-dev-LoRA-Flat-Cartoon-Style"
weight = 0.7
compatible_with = ["flux-1"]
trigger = "flat cartoon style"

[[loras]]
ref = "Kontext-Style/Irasutoya_lora"
weight = 0.4
compatible_with = ["flux-1"]
trigger = "Irasutoya style"
```

Use it via `imgen photo.jpg --style anime_strong`. The `compatible_with` list controls which backends the LoRA can attach to (see below); `trigger` is optional but lets `imgen` auto-prepend the activation phrase. (Pick Kontext-trained refs — `*-FluxDev-*` or unverified FLUX.1-dev base LoRAs will crash mflux Kontext at the first denoise step.)

### Compatibility groups

LoRAs are architecture-bound. A LoRA trained for FLUX.1 will NOT load on Qwen-Image-Edit, and vice versa. `imgen` declares a `lora_compat_group` on each model (`"flux-1"` for the default `flux-kontext` model; `"qwen"` for `qwen-image-edit-v1`); when a LoRA's `compatible_with` list doesn't include the active model's group, `imgen` warns once and skips the LoRA, then continues with the rest of the stack (or fully text-only if all LoRAs are incompatible). It doesn't crash and doesn't silently apply a mismatched LoRA.

In practice all built-in LoRAs are flux-1 only. Switching `--model qwen-image-edit-v1` produces a warn-and-skip plus a text-only generation for those styles. Qwen-side LoRAs do exist; you'd attach them ad-hoc via `--lora REF` with `compatible_with = ["qwen"]` in a user style TOML.

### License model

**`imgen` ships no LoRA weights of its own.** The built-in style mappings reference HuggingFace repos by id, and `mflux` (via `huggingface_hub`) downloads them on first use into `~/.cache/huggingface/hub/` — the same cache that holds FLUX itself. `imgen clean --all` clears the whole HF cache including LoRAs.

Per-LoRA license (as published on the upstream HF model card):

- **Shakker-Labs Kontext LoRAs** (`Flat-Cartoon-Style`, `Sketch-Style`) — HF reports `license: other`; the model cards reference FLUX NC, so practically non-commercial. Check the model card directly for the latest wording.
- **`openfree/flux-chatgpt-ghibli-lora`** — `flux-1-dev-non-commercial-license`. Non-commercial only.
- **Kontext-Style org LoRAs** (`Poly_lora`, `3D_Chibi_lora`, `Oil_Painting_lora`, `Irasutoya_lora`) — **license unspecified** on the upstream model cards (no `license` field in HF metadata). `imgen` ships these for non-commercial use only (the FLUX-NC base license gates commercial use regardless). Review upstream model cards directly before any commercial use.

The blanket caveat in the LoRA section above (FLUX-NC base gates commercial use) applies to every LoRA invoked through `imgen` regardless of the LoRA's own upstream license, because the base model is what actually runs.

`--lora` references you supply yourself are entirely your responsibility — `imgen` doesn't check upstream licenses, and (as the Kontext-compatibility note above explains) most FLUX-LoRAs on HuggingFace target FLUX.1-dev base and will crash on Kontext. The same applies to user `styles.d/*.toml` LoRA entries.

## Models

> v0.8.0 renamed `--backend` → `--model` and two built-in names moved to honest
> v0.8 spellings (`flux` → `flux-kontext`, `qwen` → `qwen-image-edit-v1`).
> Old shell scripts using `--backend` get a hard error with the migration hint.
> See the v0.8.0 migration note at the top of the release notes for the rest of
> the rename surface (`backends.d/` → `models.d/`, etc.).

Built-in:

- `flux-kontext` (default for `imgen generate` / `batch`) — **FLUX.1 Kontext Dev** — best quality for style transfer. Gated, requires:
  - HF token (any classic Read token)
  - License acceptance at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- `qwen-image-edit-v1` — **Qwen-Image-Edit-2509** — open model, no token required. Lower quality at low quants.
- `flux-dev` — **FLUX.1-dev** — t2i base for `imgen draw`. Gated, same HF token + license as `flux-kontext`.
- `flux2-klein-edit-9b` — **FLUX.2-klein-9B** distilled edit — default for `imgen refine` (v0.7.5+). Native ~4 MP support (up to 2048²) past FLUX.1's 1.5K clean ceiling. Gated, accept license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B. Q4 default needs **~24 GB peak RAM** (real measurement at 1.5K-2K²; 32 GB Mac required, 16 GB will OOM). Internal `--guidance` pinned to 1.0 by mflux — `imgen refine` handles this automatically (the `--guidance` flag still works for `--model flux-kontext` fallback).

`imgen --list-models` shows the full set including any user-defined models below. The legacy `imgen --list-backends` form is detected and dies with a migration hint pointing at the new flag.

### Why these specific model versions?

**FLUX.1 Kontext for restyle, FLUX.2-klein for refine.** FLUX.2 (released Nov 2025; klein distilled variants Jan 2026) is two different model families: `klein-base` is text-to-image (doesn't take an input photo), and `klein-edit` is *instruction-based editing* ("make the sky blue") plus low-strength i2i, not the dense image-conditioning that drives style transfer. FLUX.1 Kontext was purpose-built for "rewrite this image while preserving identity / pose / composition" — exactly the load `imgen generate` / `imgen batch` carry, so they stay on Kontext. The six built-in style presets are prompt-tuned for Kontext's verb conventions ([BFL Kontext prompting guide](https://docs.bfl.ai/guides/prompting_guide_kontext_i2i)), so expect to retune prompts if you swap. v0.7.5 added `imgen refine` on **FLUX.2-klein-edit-9b** (via `mflux-generate-flux2-edit`) because the Hires-Fix workload — preserve composition, push detail, upsample past 1.5K — wants exactly the low-strength i2i FLUX.2-klein-edit is good at, AND its native 4 MP support clears FLUX.1's clean 1.5K ceiling without tiling artifacts. You can still swap refine onto `--model flux-kontext` if you want to stay under 1.5K with Kontext.

**Qwen-Image-Edit-2509, not 2511.** Qwen-Image-Edit-2511 (released 2025-12-17) is newer and arguably stronger, but mflux 0.17.5 — the only version this CLI is tested against — hardcodes `Qwen/Qwen-Image-Edit-2509` in its qwen-edit entrypoint. Bumping the pin requires upstream mflux support; tracked as a future release candidate.

### User-defined models

Drop `*.toml` files into `~/.imgen/models.d/` (v0.8.0+ canonical path). Filename becomes the `--model NAME`. Same drop-in pattern as styles.d, applied to the image-gen binaries imgen drives — useful for experimenting with new mflux-shaped models (future SDXL ports, your own wrapper script, etc.) without editing imgen's code.

> Pre-v0.8.0 the directory was `~/.imgen/backends.d/`. Files there still load with a DEPRECATED warn through v0.8.x **and v0.9.x** — run `mv ~/.imgen/backends.d/<NAME>.toml ~/.imgen/models.d/<NAME>.toml` per file to clear it. The legacy read is targeted for hard-removal in v0.10.0; v0.9.0's README originally claimed it dropped at v0.9.0 but the bridge survives the v0.9 arc for continuity.

```toml
# ~/.imgen/models.d/sdxl.toml
binary = "mflux-generate-sdxl"        # bare name (looked up in imgen's venv) OR absolute path
image_flag = "--image-path"           # "--image-path" or "--image-paths" (the two mflux shapes)
supports_strength = true              # model accepts --image-strength
supports_negative = false             # model accepts --negative-prompt
extra_args = ["--model", "sdxl"]      # appended unconditionally to every invocation

# Optional [secret] section — for models needing an API key/token
# in the subprocess env. Value comes from the parent shell's env;
# imgen forwards but does NOT store it.
[secret]
env_var = "MY_MODEL_API_KEY"          # name imgen looks up in os.environ
required = true                       # false → best-effort forward, no die on missing
```

Then:

```bash
# Verify the registry sees it + binary resolves + secret is set:
imgen doctor

# Or just list:
imgen --list-models

# Use it:
export MY_MODEL_API_KEY=...             # (only if [secret] declared with required=true)
imgen photo.jpg --model sdxl
```

> **Security:** `binary = ...` is exec'd as a subprocess by imgen. Treat models.d/ files **like shell scripts** — only drop in files you wrote yourself or got from a source you trust. This is a strictly higher trust level than `styles.d/`: a style TOML injects arguments to a known mflux binary, a model TOML controls *which binary runs at all*. A malicious models.d entry runs arbitrary code as your user.

Collisions with built-ins (`flux-kontext.toml`, `qwen-image-edit-v1.toml`) get a `_0001` suffix with a warning; built-ins always win on name. Mirrors the styles.d collision policy. Binary paths starting with `/` are used as-is; bare names resolve to `~/imgen/.venv/bin/<name>` (the venv that hosts mflux). Built-in fields you'll never see in a user TOML: `needs_token` (FLUX-specific HF token plumbing) is hard-coded `false` for user models — use the `[secret]` section above for non-HF tokens.

## Persistent config

`imgen setup` creates `~/.imgen/config.toml` with every key commented out. Uncomment what you want to override:

```toml
[defaults]
model = "qwen-image-edit-v1"  # see `imgen --list-models`
                              # (legacy `backend = "qwen"` still loads
                              #  with a DEPRECATED warn through v0.8.x
                              #  and v0.9.x; targeted for hard-removal
                              #  in v0.10.0.)
quantize = 4
steps = 12
guidance = 4.0
strength = 0.6
output_dir = "~/Pictures/imgen"

[ui]
open_in_preview = false     # don't auto-open results
color = "auto"              # "auto" | "always" | "never"
```

**Precedence:** CLI flag > `~/.imgen/config.toml` > built-in defaults. Bad value (e.g. `steps = 999`) → `imgen` warns and falls back to built-ins until you fix the file. Unknown keys are dropped with a warning so old `imgen` versions don't break on configs written by newer ones.

> **v0.8.0 migration:** `[defaults] style = ...` was REMOVED (soft-deprecated since v0.7.13, hard-removed at v0.8.0) — pass `--style NAME` per-invocation instead, or set `[enhance] default = true` to keep enhanced prompts as your default. `[defaults] backend = "flux"` still loads through v0.8.x **and v0.9.x** with a DEPRECATED warn that auto-maps to `[defaults] model = "flux-kontext"`. Rename in your config to clear the warn; the legacy key is targeted for hard-removal in v0.10.0 (not v0.9.0 — earlier README claimed v0.9.0 drop, but the bridge survives the v0.9 arc for continuity).

For `output_dir` specifically the resolution is **`--output-dir` CLI flag > `$IMGEN_OUTPUT_DIR` env > config > default**.

**Output layout.** Every run writes into a fresh timestamped folder under the resolved output root — `~/Desktop/imgen/2026-05-21-14-30-12/photo-pixar.png`. The folder name is `YYYY-MM-DD-HH-MM-SS`, sortable both alphabetically and chronologically. Files inside are named `<input-basename>-<style>.png`; `mtime` in Finder gives completion-time ordering. To bypass the folder entirely and write to one specific file, use `-o`/`--output FILE` (mutex with `--output-dir`).

**`imgen batch <dir>`** keeps the same flat folder layout — N inputs × M styles all land in one timestamped folder, named `<input.stem>-<style>.png`. Non-recursive (subdirectories ignored on purpose so `.photoslibrary` packages and mounted volumes can't leak in); dotfiles like `.DS_Store` skipped. Supported input formats: `jpg`/`jpeg`/`png`/`webp`/`heic`/`heif`/`bmp`/`tif`/`tiff`/`gif`. HEIC inputs are auto-converted to JPEG via macOS-native `sips` before mflux sees them — converted files live in a private `0o700` temp dir wiped on exit. Two inputs that would map to the same output stem (e.g. `IMG_1234.heic` + `IMG_1234.jpg`) fail fast with a "rename one" hint instead of silently overwriting.

**Color:** `[ui] color = "auto"` (default) emits ANSI only when stdout is a tty; `"always"` forces color (handy for piping into `less -R`); `"never"` disables. The `NO_COLOR` env var (https://no-color.org/) beats both — any non-empty value disables color regardless of config.

## Environment

| Variable / file        | Purpose |
|------------------------|---------|
| `~/.imgen/hf_token`    | HuggingFace token (chmod 600). Older installs at `~/.hf_token` auto-migrate on first run. |
| `$HF_TOKEN`            | Overrides `~/.imgen/hf_token` |
| `$IMGEN_OUTPUT_DIR`    | One-off override of output dir parent (beats config.toml; `--output-dir` flag beats env) |
| `$NO_COLOR`            | Any non-empty value disables ANSI color (https://no-color.org/) |
| `$HF_HOME` / `$HF_HUB_CACHE` / `$TRANSFORMERS_CACHE` | Override where HuggingFace caches model weights (FLUX, Qwen, and the prompt-enhancer LLM) |
| `~/.imgen/config.toml` | Persistent defaults — see [Persistent config](#persistent-config) |
| `~/.imgen/styles.d/*.toml` | User-defined style presets — see [User-defined styles](#user-defined-styles) |
| `~/.imgen/backends.d/*.toml` | User-defined image-gen backends — see [User-defined backends](#user-defined-backends) |
| `~/.imgen/history.jsonl` | Generation history (JSONL, schema-versioned). Includes pre- and post-enhance prompts when `--enhance-prompt` was used. |
| `~/.imgen/logs/<batch-id>.log` | Per-batch stderr log (multi-style runs only; mflux output with HF tokens redacted). Auto-pruned by `imgen clean` after 30 days. |
| `~/imgen/.venv/`       | bootstrap install — mflux + imgen venv |
| `~/.local/pipx/venvs/imgen/` | pipx install — mflux + imgen venv |

## Performance (M2 Pro 32 GB)

| Operation | Time |
|-----------|------|
| First-run FLUX download | ~30 min one-time (~24 GB) |
| First-run prompt-enhancer download | ~5-10 min one-time (~4 GB), only if you use `--enhance-prompt` |
| FLUX Kontext Q8, 20 steps, 1024px (default) | ~15 min |
| FLUX Kontext Q4, 8 steps, 768px (`--preview`) | ~3–3.5 min |
| Qwen Edit Q4, 20 steps, 1024px | ~18 min |
| FLUX.2-klein-edit Q4, 20 steps, 1536² (`imgen refine` default) | **~49 min** (real measurement; + ~15 GB first-run download) |
| FLUX.2-klein-edit Q4, 20 steps, 2048² (`imgen refine --scale 2`) | **~110 min** (real measurement at 330 s/iteration; memory-pressure bound) |
| `--enhance-prompt` overhead | ~3-5 s per image after warm-up |

Wall-clock figures measured on a quiet machine. First image after launch pays a one-time weight-load cost (~30–60 s of mmap); subsequent images in the same `imgen batch` reuse the loaded weights, so an N-image batch is roughly `30 s + N × 15 min`, not `N × 15.5 min`. With `--enhance-prompt`, the enhancer model loads once per `imgen` invocation and amortises across all prompts in a batch.

## Model cache

mflux downloads weights from HuggingFace into the standard `huggingface_hub` cache:

```
~/.cache/huggingface/hub/
├── models--black-forest-labs--FLUX.1-Kontext-dev/   # ~31 GB (FLUX, default)
│   ├── blobs/                                       # actual weight files
│   ├── snapshots/<commit-sha>/                      # symlinks to blobs
│   └── refs/main                                    # text file with sha
└── models--Qwen--Qwen-Image-Edit-2509/              # ~40 GB (Qwen, optional)
```

`imgen` does not move or duplicate these — it reads `~/.cache/huggingface/hub/` directly, same as anything else that uses `huggingface_hub`. To put the cache on another disk (e.g. external SSD because internal is tight), set `HF_HOME` before first run:

```bash
export HF_HOME=/Volumes/external-ssd/hf-cache    # subprocess inherits this
imgen photo.jpg --preview
```

`imgen` whitelists `HF_HOME`, `HF_HUB_CACHE`, and `TRANSFORMERS_CACHE` when launching mflux, so any of the three works.

### Pre-downloading models manually

Useful when first-run downloads are flaky, when you want to seed the cache from a phone tether, or when adding a new backend (see [User-defined backends](#user-defined-backends)). Two paths:

**`huggingface-cli` (recommended).** Resumable, integrity-checked, lays out the cache correctly:

```bash
pip install --user "huggingface_hub[cli]"
huggingface-cli login                                # paste your HF token
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev
# → drops into ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev/
```

`imgen photo.jpg --preview` after that will skip the download and go straight to generation.

**Browser download.** Go to the HF model page (e.g. https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), accept the license once, then "Files and versions" → download every file. Drop them into

```
~/.cache/huggingface/hub/models--<author>--<repo>/snapshots/<commit-sha>/
```

…and put the sha string into `refs/main`. Fiddly; the `huggingface-cli` path is much less error-prone for FLUX/Qwen-scale repos (dozens of safetensors shards). The browser route is only worth it for single-file LoRAs or if you cannot install `huggingface-cli`.

### Adding a custom model via backends.d/

Once weights are cached, wire them into `imgen` with a `~/.imgen/backends.d/*.toml` drop-in — no code change. Example for FLUX.1-dev (text-to-image, different from the default Kontext image-to-image):

```toml
# ~/.imgen/backends.d/flux-dev.toml
binary = "mflux-generate"            # mflux's text-to-image entrypoint
image_flag = "--image-path"          # required field, ignored by this binary
supports_strength = false
supports_negative = false
extra_args = ["--model", "dev"]      # tells mflux which HF repo to load
```

See `mflux-generate-* --help` or `.venv/bin/` (after install) for the full set of available binaries (`mflux-generate-flux2`, `mflux-generate-flux2-edit`, `mflux-generate-qwen`, etc.). Section [User-defined backends](#user-defined-backends) below has the full schema + security model.

## Resource preflight

Before every generation `imgen` checks:

- **RAM** — backend × quant has a known peak requirement. Blocks if insufficient
- **Disk** — warns if < 5 GB free
- **Battery** — warns if < 30% on battery
- **Parallel mflux** — blocks if another mflux process is running (would OOM)

Use `--force` to skip all checks at your own risk.

## Troubleshooting

```bash
imgen doctor                          # always start here
```

| Problem | Fix |
|---------|-----|
| `venv missing` | `./bootstrap.sh` or `imgen setup` |
| `HF token not found` | `imgen setup` (paste token) |
| `403 gated repo` | Accept license: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev |
| `Not enough RAM` | Close apps, or use `--quantize 4` / `--preview` / `--model qwen-image-edit-v1` |
| `Another mflux running` | Wait for it, or `--force` (will fight for GPU/RAM) |
| Black image | You're not using `imgen` — that's a ComfyUI/MPS issue. mflux uses MLX, no MPS bugs |
| Disk full | `imgen clean --all` |
| `--enhance-prompt` says "runner_error" | Run `imgen doctor` — check that mlx-lm is importable and the enhancer model is cached. First run downloads ~4 GB, can be slow on shared networks. |
| Enhancer "succeeded" but generation looks the same | Diffusion quality varies seed-to-seed; try a different `--seed`. If it consistently looks identical, your `--custom-prompt` may already be detailed enough that the enhancer has little to add. |

## Why not ComfyUI?

ComfyUI on Mac has well-documented PyTorch/MPS issues with Qwen and FLUX models — black images, NaN attention, BF16 emulation slowdowns. mflux uses Apple's native MLX framework, which sidesteps all of that.

## License

MIT — see [LICENSE](LICENSE).

Third-party model licenses (FLUX Kontext, FLUX.1-dev, FLUX.2-klein, Qwen Image Edit) apply to generated images. See LICENSE for details.

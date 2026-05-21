# imgen

Photo style-transfer CLI for Apple Silicon Macs. Wraps [mflux](https://github.com/filipstrand/mflux) (MLX-native FLUX Kontext / Qwen Image Edit) with sane defaults, presets, and zero MPS bugs.

```bash
imgen photo.jpg                              # Pixar style (default)
imgen photo.jpg --style anime
imgen photo.jpg --style simpsons --preview   # ~3 min fast test
imgen photo.jpg --custom-prompt "Mona Lisa painting style"
```

Output lands in `~/Desktop/imgen/<basename>_<style>_<timestamp>.png` and opens in Preview automatically.

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4) — MLX does not support Intel
- **Python 3.12** (install: `brew install python@3.12`)
- **32 GB unified memory recommended** (16 GB works with `--quantize 4`)
- **~60 GB free disk** (FLUX + Qwen models combined ~80 GB cached)
- **HuggingFace account** (for FLUX Kontext — gated model, needs license acceptance)

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

## All commands

```bash
# Generation
imgen <photo>                                  # default style (pixar)
imgen <photo> --style anime                    # preset
imgen <photo> --custom-prompt "..."            # free-form
imgen <photo> -s anime --preview               # fast mode (~3-10 min)
imgen <photo> -s anime --scope person          # only restyle person, keep bg photorealistic
imgen <photo> -s anime --scope scene           # restyle whole image
imgen <photo> --backend qwen                   # use Qwen Edit (no HF token needed)
imgen <photo> --force                          # skip resource preflight checks

# Diagnostics
imgen doctor                                   # env + RAM forecast + cached models
imgen --list-styles                            # show presets
imgen --dry-run <photo> -s anime               # show mflux command, don't run

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

## Tuning

| Flag                | Range  | Effect |
|---------------------|--------|--------|
| `--steps`           | 1-200  | More = better, slower. Sweet spot 15-30 |
| `--guidance` / `-g` | 0.5-15 | How strictly to follow prompt. 3.5-4.5 |
| `--strength`        | 0-1    | How much to keep from original. 0.5-0.7 |
| `--quantize` / `-q` | 3,4,5,6,8 | Lower = smaller/faster, more artifacts |
| `--preview` / `-p`  | flag   | Q4, 8 steps, 768x — ~5x faster |
| `--scope`           | person/scene | Modify prompt to focus on person or whole scene |

For 32 GB Macs, **Q8** is recommended for FLUX Kontext.

## Backends

- `flux` (default) — **FLUX.1 Kontext Dev** — best quality for style transfer. Gated, requires:
  - HF token (any classic Read token)
  - License acceptance at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- `qwen` — **Qwen-Image-Edit-2509** — open model, no token required. Lower quality at low quants.

## Environment

| Variable / file        | Purpose |
|------------------------|---------|
| `~/.hf_token`          | HuggingFace token (chmod 600) |
| `$HF_TOKEN`            | Overrides `~/.hf_token` |
| `$IMGEN_OUTPUT_DIR`    | Override default output dir |
| `~/.imgen/history.jsonl` | Generation history (JSONL, schema-versioned) |
| `~/imgen/.venv/`       | bootstrap install — mflux + imgen venv |
| `~/.local/pipx/venvs/imgen/` | pipx install — mflux + imgen venv |

## Performance (M2 Pro 32 GB)

| Operation | Time |
|-----------|------|
| First-run FLUX download | ~30 min one-time (~24 GB) |
| FLUX Kontext Q8, 20 steps, 1024px | ~50 min |
| FLUX Kontext Q4, 8 steps, 768px (`--preview`) | ~3 min |
| Qwen Edit Q4, 20 steps, 1024px | ~18 min |

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
| `Not enough RAM` | Close apps, or use `--quantize 4` / `--preview` / `--backend qwen` |
| `Another mflux running` | Wait for it, or `--force` (will fight for GPU/RAM) |
| Black image | You're not using `imgen` — that's a ComfyUI/MPS issue. mflux uses MLX, no MPS bugs |
| Disk full | `imgen clean --all` |

## Why not ComfyUI?

ComfyUI on Mac has well-documented PyTorch/MPS issues with Qwen and FLUX models — black images, NaN attention, BF16 emulation slowdowns. mflux uses Apple's native MLX framework, which sidesteps all of that.

## License

MIT — see [LICENSE](LICENSE).

Third-party model licenses (FLUX Kontext, Qwen Image Edit) apply to generated images. See LICENSE for details.

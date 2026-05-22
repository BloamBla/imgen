# imgen

Photo style-transfer CLI for Apple Silicon Macs. Wraps [mflux](https://github.com/filipstrand/mflux) (MLX-native FLUX Kontext / Qwen Image Edit) with sane defaults, presets, and zero MPS bugs.

```bash
imgen photo.jpg                              # Pixar style (default)
imgen photo.jpg --style anime
imgen photo.jpg --style simpsons --preview   # ~3 min fast test
imgen photo.jpg --custom-prompt "Mona Lisa painting style"
imgen batch ~/Desktop/holiday --style anime,ghibli   # v0.3.0: every photo in folder × every style
```

Every run creates a timestamped folder under `~/Desktop/imgen/` — e.g. `~/Desktop/imgen/2026-05-21-14-30-12/photo-pixar.png`. The result opens in Preview automatically. Change the parent with `--output-dir PATH` or pin an exact path with `--output FILE`.

**Multi-style** (v0.2.3+): pass `--style anime,ghibli,pixar` to generate N images from one input in a single run — all dropped into the same timestamped folder, named by `<input>-<style>.png`. Confirms with a `[y/N]` summary before starting (skip with `-y/--yes`).

**Batch a folder** (v0.3.0+): `imgen batch <dir>` applies M styles to every supported image directly under `<dir>` (non-recursive — `ls <dir>` ≈ what gets batched). Same flat output layout — all N×M results in one timestamped folder, named `<input>-<style>.png`. iPhone HEIC inputs are auto-converted via `sips` before mflux sees them; the same HEIC handling also fixes single-file `imgen generate vacation.heic`. Confirm gate shows N inputs × M styles + ETA; skip with `-y/--yes`.

**v0.3.2 defaults:** `--scope` is now `scene` by default (most photos are scenes, not portraits) — pass `--scope person` to focus on the subject and keep the background photorealistic. Also, mflux's `.metadata.json` sidecars are no longer written next to outputs (the data is already embedded in the PNG and stored in `~/.imgen/history.jsonl`), so the gallery folder stays clean.

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

**`--custom-prompt` augmentation** (v0.3.5+): with a full style, `--custom-prompt` AUGMENTS the style prompt rather than replacing it — the user's text is appended as a trailing detail. `imgen photo.jpg -s anime,ghibli,pixar --custom-prompt "wearing a red kimono"` runs three generations, each preserving its style's tuned prompt with the kimono detail tacked on. Without an explicit `--style`, `--custom-prompt` is the sole prompt content and the default style only contributes its tuning params (so `imgen photo.jpg --custom-prompt "sepia film still"` works without any style prompt blending in).

If a user style's filename clashes with a built-in (e.g. `styles.d/anime.toml`), it gets registered as `anime_0001` (`_0002`, `_0003`, …) with a warning, and the built-in stays accessible as `anime`. Built-ins always win on name; the suffix mechanism makes overrides explicit.

## All commands

```bash
# Generation
imgen <photo>                                  # default style (pixar)
imgen <photo> --style anime                    # one preset
imgen <photo> --style anime,ghibli,pixar       # multi-style — M images into one timestamped folder (asks [y/N])
imgen <photo> --style anime,ghibli --yes       # multi-style, skip the confirm gate
imgen <photo> --output-dir ~/Pictures/runs     # change parent of the timestamped run folder
imgen <photo> -o explicit.png                  # bypass run-folder layout (mutex with --output-dir)
imgen <photo> --custom-prompt "..."            # v0.3.5+: with --style → augments preset; without --style → sole prompt
imgen <photo> -s anime,ghibli --custom-prompt "wearing a red kimono"  # shared addition across all styles
imgen <photo> --custom-prompt -                # ← read prompt from stdin (hidden from ps)
imgen <photo> --prompt-file ~/prompts/x.txt    # ← read prompt from file (hidden from ps)
imgen <photo> -s anime --preview               # fast mode (~3-10 min)
imgen <photo> -s anime                         # v0.3.2: --scope=scene is the default — whole image restyled
imgen <photo> -s anime --scope person          # opt into person-focus: keep background photorealistic
imgen <photo> --backend qwen                   # use Qwen Edit (no HF token needed)
imgen <photo> --force                          # skip resource preflight checks

# Batch a folder (v0.3.0+) — same flags as generate except no -o/--output (always run-folder layout)
imgen batch <dir>                              # every photo × default style → one timestamped folder
imgen batch <dir> --style anime,ghibli,pixar   # N × M into one folder, named <input>-<style>.png
imgen batch <dir> -s anime --custom-prompt "..." # v0.3.5+: shared --custom-prompt augmentation across every input
imgen batch <dir> --style anime --yes          # skip the N×M confirm gate
imgen batch <dir> --dry-run                    # show every mflux command without running

# Diagnostics
imgen doctor                                   # env + RAM forecast + cached models + backend status
imgen --list-styles                            # show presets
imgen --list-backends                          # v0.4+: show built-in + user backends from ~/.imgen/backends.d/
imgen --dry-run <photo> -s anime               # show mflux command, don't run
imgen -v   /   imgen --version                 # v0.3.5+: both forms print the version

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
| `--guidance` / `-g` | 0.5-15 | How strictly to follow prompt. 3.5-4.5 |
| `--strength`        | 0-1    | How much to keep from original. 0.5-0.7 |
| `--quantize` / `-q` | 3,4,5,6,8 | Lower = smaller/faster, more artifacts |
| `--preview` / `-p`  | flag   | Q4, 8 steps, 768x — ~5x faster |
| `--scope`           | person/scene | v0.3.2+: default `scene` (transforms whole image); pass `person` to keep background photorealistic |

For 32 GB Macs, **Q8** is recommended for FLUX Kontext.

## Backends

Built-in:

- `flux` (default) — **FLUX.1 Kontext Dev** — best quality for style transfer. Gated, requires:
  - HF token (any classic Read token)
  - License acceptance at https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- `qwen` — **Qwen-Image-Edit-2509** — open model, no token required. Lower quality at low quants.

`imgen --list-backends` shows the full set including any user-defined backends below.

### User-defined backends

Drop `*.toml` files into `~/.imgen/backends.d/` (auto-created by `imgen setup`). Filename becomes the `--backend NAME`. Same drop-in pattern as styles.d, applied to the image-gen binaries imgen drives — useful for experimenting with new mflux-shaped models (future SDXL ports, your own wrapper script, etc.) without editing imgen's code.

```toml
# ~/.imgen/backends.d/sdxl.toml
binary = "mflux-generate-sdxl"        # bare name (looked up in imgen's venv) OR absolute path
image_flag = "--image-path"           # "--image-path" or "--image-paths" (the two mflux shapes)
supports_strength = true              # backend accepts --image-strength
supports_negative = false             # backend accepts --negative-prompt
extra_args = ["--model", "sdxl"]      # appended unconditionally to every invocation

# Optional [secret] section — for backends needing an API key/token
# in the subprocess env. Value comes from the parent shell's env;
# imgen forwards but does NOT store it.
[secret]
env_var = "MY_BACKEND_API_KEY"        # name imgen looks up in os.environ
required = true                       # false → best-effort forward, no die on missing
```

Then:

```bash
# Verify the registry sees it + binary resolves + secret is set:
imgen doctor

# Or just list:
imgen --list-backends

# Use it:
export MY_BACKEND_API_KEY=...           # (only if [secret] declared with required=true)
imgen photo.jpg --backend sdxl
```

> **Security:** `binary = ...` is exec'd as a subprocess by imgen. Treat backends.d/ files **like shell scripts** — only drop in files you wrote yourself or got from a source you trust. This is a strictly higher trust level than `styles.d/`: a style TOML injects arguments to a known mflux binary, a backend TOML controls *which binary runs at all*. A malicious backends.d entry runs arbitrary code as your user.

Collisions with built-ins (`flux.toml`, `qwen.toml`) get a `_0001` suffix with a warning; built-ins always win on name. Mirrors the styles.d collision policy. Binary paths starting with `/` are used as-is; bare names resolve to `~/imgen/.venv/bin/<name>` (the venv that hosts mflux). Built-in fields you'll never see in a user TOML: `needs_token` (FLUX-specific HF token plumbing) is hard-coded `false` for user backends — use the `[secret]` section above for non-HF tokens.

## Persistent config

`imgen setup` creates `~/.imgen/config.toml` with every key commented out. Uncomment what you want to override:

```toml
[defaults]
style = "anime"
backend = "qwen"            # save the FLUX token check for one-off --backend flux
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

For `output_dir` specifically the resolution is **`--output-dir` CLI flag > `$IMGEN_OUTPUT_DIR` env > config > default**. The CLI flag wins (added in v0.2.3 — beats env, which was the v0.1.x one-off channel); env is still the easiest way to redirect without touching config.

**Output layout (v0.2.3+):** every run writes into a fresh timestamped folder under the resolved output root — `~/Desktop/imgen/2026-05-21-14-30-12/photo-pixar.png`. The folder name is `YYYY-MM-DD-HH-MM-SS`, sortable both alphabetically and chronologically. Files inside are named `<input-basename>-<style>.png`; `mtime` in Finder gives completion-time ordering. To bypass the folder entirely and write to one specific file, use `-o`/`--output FILE` (mutex with `--output-dir`).

**`imgen batch <dir>`** (v0.3.0+) keeps the same flat folder layout — N inputs × M styles all land in one timestamped folder, named `<input.stem>-<style>.png`. Non-recursive (subdirectories ignored on purpose so `.photoslibrary` packages and mounted volumes can't leak in); dotfiles like `.DS_Store` skipped. Supported input formats: `jpg`/`jpeg`/`png`/`webp`/`heic`/`heif`/`bmp`/`tif`/`tiff`/`gif`. HEIC inputs are auto-converted to JPEG via macOS-native `sips` before mflux sees them — converted files live in a private `0o700` temp dir wiped on exit. Two inputs that would map to the same output stem (e.g. `IMG_1234.heic` + `IMG_1234.jpg`) fail fast with a "rename one" hint instead of silently overwriting.

**Color:** `[ui] color = "auto"` (default) emits ANSI only when stdout is a tty; `"always"` forces color (handy for piping into `less -R`); `"never"` disables. The `NO_COLOR` env var (https://no-color.org/) beats both — any non-empty value disables color regardless of config.

## Environment

| Variable / file        | Purpose |
|------------------------|---------|
| `~/.imgen/hf_token`    | HuggingFace token (chmod 600). v0.2.x used `~/.hf_token`; that path is still read as a fallback and auto-migrated on first run. |
| `$HF_TOKEN`            | Overrides `~/.imgen/hf_token` |
| `$IMGEN_OUTPUT_DIR`    | One-off override of output dir parent (beats config.toml; `--output-dir` flag beats env) |
| `$NO_COLOR`            | Any non-empty value disables ANSI color (https://no-color.org/) |
| `~/.imgen/config.toml` | Persistent defaults — see [Persistent config](#persistent-config) |
| `~/.imgen/styles.d/*.toml` | User-defined style presets — see [User-defined styles](#user-defined-styles) |
| `~/.imgen/backends.d/*.toml` | User-defined image-gen backends — see [User-defined backends](#user-defined-backends) |
| `~/.imgen/history.jsonl` | Generation history (JSONL, schema-versioned) |
| `~/.imgen/logs/<batch-id>.log` | Per-batch stderr log (multi-style runs only; mflux output with HF tokens redacted). Auto-pruned by `imgen clean` after 30 days. |
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

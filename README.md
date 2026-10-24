# ComfyUI-MossImage-owoinfer

ComfyUI nodes that run MOSS-Image (the 5B DiT with the Qwen3.5-9B text encoder and the FLUX.2 VAE) through **owoinfer (MossInfer)**, a native C++/CUDA engine with no PyTorch at run time.

The engine runs as its own process, `moss-server`, which the pack starts and talks to over pipes (JSON lines). This has three consequences:
- **Crash isolation.** A CUDA fault ends `moss-server`, not ComfyUI. The pack starts a new one for the next run.
- **No fight over VRAM.** owoinfer plans its own GPU memory, outside torch's caching allocator. By default the Generate node first unloads ComfyUI's own models.
- **One engine, shared by every node.** There is one `moss-server` per ComfyUI process. It is reused across runs, loads a model only when the loader's settings change, and runs one job at a time. ComfyUI stops it on exit.

The engine is not a ComfyUI model. Its nodes do not produce or take ComfyUI's `MODEL`, `CLIP`, `VAE` or `LATENT`, and they do not work with KSampler. The input is a prompt and optional images, and the output is an `IMAGE`.

## Supported now

| Area | What works |
|---|---|
| Models | MOSS-Image Base and SFT bundle folders from Hugging Face, and fine-tunes with the same architecture |
| Component files | a single-file DiT (ComfyUI / community INT8 ConvRot, INT4 ConvRot, bf16), text encoder (INT8 ConvRot, W4A8) or VAE, next to the bundle folder |
| Generation | text-to-image: prompt, negative prompt, size, steps, CFG, shift, seed, skip layers, skip-layer guidance, maximum prompt length |
| Samplers | `euler`, `ab2`, `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_2m_sde`, `er_sde`, `res_multistep`; 7 schedulers; noise layout and source, eta, s_noise, flow multiplier |
| Image input | img2img from an `IMAGE` with a denoise strength, inpainting with a `MASK`, crop / fit / stretch resizing |
| Hires fix | a second pass at a larger size, with its own strength and steps |
| LoRAs | any number, chained, each with a strength: kohya, PEFT / diffusers, LyCORIS LoHa and LoKr, full deltas |
| Control adapters | a control adapter file (e.g. the MOSS-Image color / depth control) with one conditioning image per kind, each with a strength and a step window |
| Lossy accelerators | CFG truncation and the First-Block step cache, off by default |
| Quality tiers | `auto`, `reference`, `high`, `turbo` (INT4 DiT files only), and the attention override |
| Memory modes | `balanced`, `min_memory`, `max_speed` |
| Progress | the node's progress bar and status text; latent previews on the node while sampling; overall progress and ETA from servers that report them |
| Cancel | ComfyUI's Cancel stops the engine within one layer or block |
| After a run | keep the model in VRAM, release the VRAM and keep the weights in RAM (servers with `release`), or unload |
| Platforms | wherever owoinfer runs: Linux x86-64 and Windows x64, NVIDIA compute capability 8.x and 12.x (RTX 30, 40 and 50 series, A100, A10, A40, L4, L40, RTX PRO Blackwell). The pack itself is tested on Linux. |

The nodes adapt to the `moss-server` they find. They ask it for its capabilities and offer only what it supports: for example, `after_run` lists `release` only with a server that has it. A setting the server lacks is refused with a message that names the capability, rather than sent anyway.

## Planned

| Feature | State |
|---|---|
| Distill bundles and the `dataspace` sampler | the engine supports them; the pack needs a bundle-folder picker on the loader and on Generate, and "unset" steps / CFG |
| Preload when the model is selected | the engine's `preload` (servers with that capability); the pack does not send it yet |
| More engine settings on the loader | VRAM budget and reserve, GPU index, LoRA strictness, text-encoder pinning, extra certificate folders, DiT read mode |
| A model folder made of small files only | needs an engine change (see "Model files") |

## Not supported

| Area | Status |
|---|---|
| ComfyUI's `MODEL` / `CLIP` / `VAE` / `LATENT`, KSampler, ComfyUI's LoRA loaders | not supported: the engine runs the whole pipeline in its own process |
| A single ComfyUI checkpoint or model file as the model | not supported: the model is a bundle folder; ComfyUI files load as component files next to it |
| Several images per Generate node (batch size) | not supported: one image per run. A batch on `init_image` uses its first image. |
| Several generations at once | not supported: the engine runs one job at a time; ComfyUI's queue runs them in turn |
| GGUF, FP8 (`fp8_scaled`, `float8_e4m3fn`), torchao, MLX, SDNQ and `.bin` / `.pt` files | not supported by the engine |
| NVFP4 DiT files | not supported by the engine in this version |
| Image editing models (MOSS-Image-Edit and similar) | not supported: the engine has no reference-image input |
| GPUs below compute capability 8.0, or 9.x to 11.x; AMD, Intel and Apple GPUs; macOS; CPU only | not supported by the engine |

## Install

1. Put this folder in `ComfyUI/custom_nodes/`, as `ComfyUI/custom_nodes/ComfyUI-MossImage-owoinfer`. It needs nothing beyond what ComfyUI already has (torch, numpy, Pillow).
2. Give it a `moss-server`. Download a MossInfer release for your OS and extract it into the pack's `bin/` folder:
   ```sh
   tar -xzf mossinfer-<version>-linux-cu13-x86_64.tar.gz -C ComfyUI/custom_nodes/ComfyUI-MossImage-owoinfer/bin/
   ```
   On Windows, extract the `.zip` into `bin\` the same way. Keep the archive's `bin/` and `lib/` together, because `moss-server` loads the engine library from `../lib`.
   - **Which archive:** `cu13` by default. Take `cu12` when the NVIDIA driver is older than 580. The `-system-cuda` archives are smaller and need a compatible CUDA toolkit installed.
3. Restart ComfyUI. The nodes are under **MOSS-Image (owoinfer)**.

**Where the pack looks for `moss-server`**, in this order:
1. the loader's `server_path` setting, when set;
2. the `MOSSINFER_SERVER` environment variable;
3. `bin/moss-server` in the pack, or `bin/*/bin/moss-server` (a release archive extracted into `bin/`);
4. `PATH`.

If none is found, the error names all of these places.

## Model files

**The model is a bundle folder, and it must be complete.** Download the official folder from Hugging Face into `ComfyUI/models/moss_image/`:

```sh
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir ComfyUI/models/moss_image/MOSS-Image-SFT --exclude 'assets/*'
hf download OpenMOSS-Team/MOSS-Image-Base --local-dir ComfyUI/models/moss_image/MOSS-Image-Base --exclude 'assets/*'
```

A folder is listed by the loader when it holds `tokenizer/` and `transformer/config.json`, `text_encoder/config.json` and `vae/config.json`, up to three levels below `models/moss_image/`. Each folder is about 27 GB. To save disk with both models, hard-link SFT's `text_encoder/*.safetensors` and `vae/*.safetensors` to Base's: they are byte-identical.

**Component files** replace one part of the folder at run time. Put them in ComfyUI's usual folders:

| Loader input | ComfyUI folder | Example |
|---|---|---|
| `dit_file` | `models/diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` (Comfy-Org), `moss_image_sft_int4_convrot.safetensors` (toxicdog) |
| `text_encoder_file` | `models/text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors` (Comfy-Org) |
| `vae_file` | `models/vae/` | `flux2-vae.safetensors`, `flux2-vae_small.safetensors` (Comfy-Org) |
| LoRA `lora_name`, Control `adapter` | `models/loras/` | `moss_image_color_depth_control.safetensors` (Comfy-Org) |

Be precise about what the engine needs:
- **The folder is still required in full.** The tokenizer and configs always come from it. owoinfer also reads the folder's own weight files (`transformer/`, `text_encoder/`, `vae/`) when it indexes the model, even when component files replace all three. A folder with only the configs and tokenizer (no large `.safetensors` files) is refused at the first generation with a `model` error, e.g. `…/text_encoder: expected one .safetensors file or a shard index, found 0 files and no index`. So ComfyUI users cannot yet get by with the small files of the folder plus Comfy component files.
- **A single file as the model is refused** (`not_supported`). The loader therefore picks a folder, never a file.
- **INT4 DiT files run only under tier `turbo`, and `turbo` runs only INT4 files.** The loader checks the file's header: with an INT4 file, `auto` becomes `turbo`; `turbo` with any other DiT, or `reference` / `high` with an INT4 file, is refused before the engine starts. `reference` with an INT8 file is refused by the engine, with a message that names the conflict.
- **Wrong files are refused by the engine** at the first generation, with a message that names the file and what was found (another model, GGUF, FP8 and so on).

**Other model folders.** `models/moss_image` is a regular ComfyUI model folder, so `extra_model_paths.yaml` can add more:

```yaml
my_models:
  base_path: /data/models
  moss_image: MOSS-Image          # bundle folders under /data/models/MOSS-Image/
  diffusion_models: comfy-org
```

An entry can also name one bundle folder directly.

## Nodes

### MOSS-Image Model Loader → `MOSS_MODEL`

A plain description of the model: the engine is global, and loads it on the first run.

| Input | Values |
|---|---|
| `model_folder` | the bundle folders found under `models/moss_image/` |
| `dit_file`, `text_encoder_file`, `vae_file` | `(from folder)` (default) or a file from `diffusion_models`, `text_encoders`, `vae` |
| `mode` | `balanced` (default), `min_memory`, `max_speed` |
| `tier` | `auto` (default), `reference`, `high`, `turbo` |
| `attention` (advanced) | `default` (the tier's choice), `exact`, `q8_fp16`, `q8_fp16_f32acc`, `q8_fp8`, `q8_fp8_f32acc` |
| `server_path` (advanced) | another `moss-server` binary; empty uses the search order above |

### MOSS-Image LoRA: `MOSS_MODEL` → `MOSS_MODEL`

`lora_name` from `models/loras/` and a `strength` (default 1). Chain several; they are applied in order, per generation, so changing them does not reload the model.

### MOSS-Image Control → `MOSS_CONTROL`

`adapter` (from `models/loras/`), `kind` (one of the adapter's kinds, e.g. `depth` or `color`), `image`, `strength` (0 to 10), `start` and `end` (the share of the steps it is active in), `lora_strength` (the adapter's LoRA part), and an optional `control` to chain another kind for the same adapter. Connect the last one to Generate's `control`. A run with a control adapter is uncertified.

### MOSS-Image Generate → `IMAGE`

| Input | Default | Notes |
|---|---|---|
| `moss_model` | | from the loader or a LoRA node |
| `prompt`, `negative_prompt` | | multiline |
| `width`, `height` | 1024 | multiples of 16, at most 4096 per side and 4.2 MP in all |
| `steps` | 50 | |
| `cfg` | 3.0 | |
| `seed` | 42 | 0 to 2^53 - 1 |
| `sampler`, `scheduler` | `euler`, `moss` | `moss` with `euler` is the official pipeline |
| `shift` | 5.0 | |
| `init_image`, `mask` | | optional: img2img and inpainting (1 repaints, 0 keeps; the init image's size) |
| `control` | | optional, from MOSS-Image Control |
| `denoise_strength`, `resize` | 0.75, `crop` | used with an init image |
| advanced: `uncond_skip_layers`, `slg_layers`, `slg_scale`, `slg_start`, `slg_stop`, `max_sequence_length`, `flow_multiplier`, `noise_layout`, `noise_source`, `eta`, `s_noise` | the engine's | |
| advanced: `hires_width`, `hires_height`, `hires_strength`, `hires_steps` | 0, 0, 0.45, 0 | hires fix; 0 × 0 is off, `hires_steps` 0 uses `steps` |
| advanced: `cfg_truncation`, `step_cache`, `step_cache_threshold` | 1, `off`, 0 | lossy accelerators; a threshold of 0 uses the engine's default |
| advanced: `preview` | on | latent previews on the node |
| advanced: `after_run` | `keep` | `keep`, `release` (servers with `release`: VRAM freed, weights kept in RAM for a fast reload), `unload` |
| advanced: `free_comfy_vram` | on | unload ComfyUI's own models before the run |

The engine checks the values itself. When it refuses one, ComfyUI shows its message as it is, e.g. `Height and width must be positive multiples of 16.` Engine warnings (a truncated prompt, LoRA layers it could not apply exactly, lossy accelerators) appear in ComfyUI's console log.

## Example workflow

`example_workflows/MOSS-Image_txt2img.json`: Model Loader → Generate → Save Image, for a bundle folder named `MOSS-Image-SFT`. Drop the file onto ComfyUI, or open it from the templates browser.

## Files the pack writes

- `ComfyUI/user/ComfyUI-MossImage-owoinfer/ComfyUI-MossImage-owoinfer.moss-server.log`: the engine's log (its stderr) for this ComfyUI session; the previous session's is kept as `.log.1`. Attach it to bug reports.
- `ComfyUI/user/ComfyUI-MossImage-owoinfer/cache/`: the engine's prompt-embedding cache and hardware profile.
- `ComfyUI/temp/ComfyUI-MossImage-owoinfer/`: the PNG files of a run (init image, mask, control images, previews, the output), deleted when it ends.

## Security notes

- `moss-server` reads and writes the paths it is given. The pack sends only paths it chose itself: files from ComfyUI's model folders and temp PNGs it wrote.
- Images reach the engine only as PNGs that the pack encodes from ComfyUI tensors. The engine's own image decoder never sees an uploaded or downloaded file.
- `server_path` comes from the workflow, so it must name a file called `moss-server` (`moss-server.exe` on Windows). A shared workflow cannot make ComfyUI run another program.

## Tests

From the pack folder, with ComfyUI's Python:

```sh
python -m pytest tests                      # unit tests and the headless ComfyUI e2e, with a fake moss-server; no GPU
OWOINFER_GPU_TESTS=1 python -m pytest tests # also the GPU smoke tests with a real moss-server (MOSS_SERVER=...)
```

The e2e tests start ComfyUI from `COMFYUI_DIR` (its code and venv) with a scratch base directory and `--cpu`, and change nothing in that install.

## License

Apache-2.0, like MossInfer. The MOSS-Image weights are published by OpenMOSS under Apache-2.0 and are downloaded separately.

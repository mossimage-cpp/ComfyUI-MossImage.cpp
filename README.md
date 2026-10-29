# ComfyUI-MossImage-owoinfer

ComfyUI nodes for [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (the 5B DiT with the Qwen3.5-9B text encoder and the FLUX.2 VAE), running on [**owoinfer**](https://github.com/xycld/owoinfer) (MossInfer), a native C++/CUDA inference engine with no PyTorch at run time.

The engine runs as its own process, `moss-server`, which the pack starts at the first run and talks to over pipes. There is one per ComfyUI process: it is reused across runs, loads a model only when the loader's settings change, and stops when ComfyUI exits. The nodes take a prompt and optional images and return an `IMAGE`. They do not use ComfyUI's `MODEL`, `CLIP`, `VAE` or `LATENT`, so they work in place of KSampler, not with it.

## Why use this instead of ComfyUI's native MOSS-Image support?

ComfyUI runs MOSS-Image natively, and if that works well for you, there is no need for this pack. Use it when:

- **VRAM is tight.** owoinfer plans its memory against the GPU it finds and streams the model from RAM or disk when it does not fit, instead of failing. It targets cards with 8 GiB of VRAM or more. On an RTX 5090 its peak was 0.34–0.66× of ComfyUI's (see [Performance](#performance)).
- **You want it faster.** With the same files, it was 1.04–1.25× as fast as ComfyUI on an RTX 5090 (see [Performance](#performance)).
- **You want crash isolation.** A CUDA fault ends `moss-server`, not ComfyUI; the next run starts a new one.
- **You want the official pipeline.** The `moss` scheduler with the `euler` sampler and tier `reference` (the folder's bf16 models) reproduces the official MOSS-Image pipeline bit for bit.

The cost is that the nodes are a closed pipeline. You can't mix them with ComfyUI's samplers, latent nodes or LoRA loaders (see [Limitations](#limitations)).

## Features

- Text-to-image, img2img with a denoise strength, and inpainting with a `MASK`.
- **Samplers:** `euler`, `ab2`, `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_2m_sde`, `er_sde`, `res_multistep`.
- **Schedulers:** 7, including `moss`.
- **Sampler options:** skip layers, skip-layer guidance, and the hires fix.
- **Models:** MOSS-Image Base and SFT folders, fine-tunes with the same architecture, and single-file DiT, text encoder and VAE files, such as the Comfy-Org INT8 ConvRot files.
- **LoRAs:** any number, chained. Formats: kohya, PEFT / diffusers, LyCORIS LoHa and LoKr, and full deltas.
- **Control adapters**, such as the colour/depth control, and the lossy accelerators (CFG truncation, First-Block cache).
- **Quality tiers:** `reference` (bf16, exact attention), `high` (the default: the DiT's INT8 or bf16 weights with 8-bit attention), `turbo` (INT4 and NVFP4 DiT files). **Memory modes:** `balanced`, `min_memory`, `max_speed`.
- Progress bar over the whole run (text encoder and model loads included) with the time remaining, latent previews (off when ComfyUI's preview method is `none`) and the engine's warnings on the node. ComfyUI's Cancel stops a run within one layer or block.
- **After a run:** keep the model in VRAM, release the VRAM and keep the weights in RAM, or unload it.

Two checks run before anything loads:
- **Files:** the engine inspects each model file the run uses (DiT, text encoder, VAE, LoRAs, control adapter) and says what it is. A file in the wrong slot, a LoRA the engine can't use, a tier the DiT doesn't run, or a control `kind` the adapter doesn't have stops the run with the engine's reason. Results are kept per file until the file or the engine changes.
- **Names:** the sampler, scheduler, attention, mode, tier and the other choices are checked against the names the server accepts, so a server that lacks one says so and lists what it has.

## Performance

Benchmarks against ComfyUI's native MOSS-Image support:

- **GPU:** RTX 5090 (a rental) at its 575 W power cap.
- **ComfyUI:** 0.42.0 (`7cebca5`), torch 2.14.1+cu130, comfy-kitchen 0.2.40, SageAttention 2.2.0. The official `image_moss_image_t2i` template.
- **owoinfer:** `moss-server`, mode `balanced`. Measured with MossInfer development builds before 2.0: `920bcf5` (bf16 and INT8, 2026-10-24/25) and `97c63d9` (NVFP4, 2026-10-28). The tier column names the 2.0 tier that runs the same configuration.
- **Files:** the same DiT files on both sides: Comfy-Org's `moss_image_sft_bf16` and `moss_image_sft_int8_convrot`, and BennyDaBall's `moss_image_sft_nvfp4`. The VAE is `flux2-vae` on both.
- **Run:** text to image, 50 steps, CFG 3. Time is the median of 3 warm runs (models loaded), from the request to the written PNG. VRAM is the process's peak in `nvidia-smi`.

| Size | DiT | ComfyUI attention | owoinfer tier | ComfyUI | owoinfer | Speed-up | Peak VRAM: ComfyUI / owoinfer |
|---|---|---|---|---|---|---|---|
| 1024² | bf16 | default (SDPA) | `reference` | 27.66 s | 23.09 s | 1.20× | 25.9 / 12.2 GiB |
| 1024² | bf16 | `--use-sage-attention` | `high` | 25.29 s | 20.96 s | 1.21× | 25.9 / 12.2 GiB |
| 1024² | INT8 | default (comfy-kitchen INT8) | `high` | 10.87 s | 9.86 s | 1.10× | 21.5 / 7.8 GiB |
| 1024² | NVFP4 | default | `turbo` | 10.58 s | 8.45 s | 1.25× | 20.7 / 7.0 GiB |
| 2048² | bf16 | default (SDPA) | `reference` | 162.10 s | 136.43 s | 1.19× | 28.0 / 18.4 GiB |
| 2048² | bf16 | `--use-sage-attention` | `high` | 122.68 s | 101.44 s | 1.21× | 28.0 / 18.4 GiB |
| 2048² | INT8 | default (comfy-kitchen INT8) | `high` | 61.88 s | 59.71 s | 1.04× | 23.6 / 13.9 GiB |
| 2048² | NVFP4 | default | `turbo` | 56.92 s | 53.09 s | 1.07× | 22.7 / 13.1 GiB |

Speed-up is ComfyUI's time divided by owoinfer's.

- **Speed:** owoinfer was 1.19–1.21× as fast with bf16 weights, 1.04–1.10× with INT8 and 1.07–1.25× with NVFP4. The bf16 rows pair the same kind of attention; Comfy-Org's INT8 file makes ComfyUI use comfy-kitchen's INT8 attention, so those rows are 8-bit on both sides. Which attention ComfyUI ran with the NVFP4 file wasn't checked.
- **VRAM:** owoinfer's peak was 0.34–0.66× of ComfyUI's. Part of ComfyUI's figure is cache, not a requirement: its dynamic VRAM keeps the text encoder resident.
- **The samplers differ:** ComfyUI ran the template's `er_sde` + `simple`, owoinfer its default `euler` + `moss`. Both make the same number of model evaluations per step. owoinfer skips layer 22 in the negative-prompt pass, as the official pipeline does (`uncond_skip_layers`); without that it is 1.3 % slower.
- **Scope:** one card, one rental. Other GPUs and the INT4 files haven't been compared.

## Requirements

- **GPU:** NVIDIA, compute capability 8.x or 12.x: RTX 30, 40 and 50 series, A100, A10, A40, L4, L40, RTX PRO Blackwell.
- **NVIDIA driver:**
  - for the `cu13` build: Linux ≥ 580.65.06, Windows ≥ 580.88;
  - for the `cu12` build: Linux ≥ 525.60.13, Windows ≥ 527.41;
  - RTX 50 series cards need the 570 series or newer either way.
- **OS:** Linux x86-64. The code is written for Windows x64 too, but it hasn't been tested there.
- **ComfyUI** with the V3 node API (`comfy_api.latest`); tested with 0.33.0.
- **owoinfer (MossInfer) 2.0 or later**, for `moss-server` (see [Getting moss-server](#getting-moss-server)).
- **Disk:** about 27 GB per model folder, or about 20 MB for a folder of configs and tokenizer next to single files (see [Models](#models)).
- **RAM:** the engine aims to stay within 4 GiB (`min_memory`), 8 GiB (`balanced`) or what is free minus 1 GiB (`max_speed`). Free RAM beyond that caches the model files.

## Installation

### Manual install

Clone the repository into `ComfyUI/custom_nodes`. It needs nothing beyond what ComfyUI already has (torch, numpy, Pillow), so there is no `pip install` step:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/xycld/ComfyUI-MossImage-owoinfer
```

For the Windows portable build, run this in the `ComfyUI_windows_portable` folder:

```
git clone https://github.com/xycld/ComfyUI-MossImage-owoinfer ComfyUI\custom_nodes\ComfyUI-MossImage-owoinfer
```

### Getting `moss-server`

Download an owoinfer release for your OS from [github.com/xycld/owoinfer/releases](https://github.com/xycld/owoinfer/releases), check it against the release's `SHA256SUMS`, and extract it into the pack's `bin/` folder:

```sh
mkdir -p ComfyUI/custom_nodes/ComfyUI-MossImage-owoinfer/bin
tar -xzf mossinfer-2.0.0-linux-cu13-x86_64.tar.gz -C ComfyUI/custom_nodes/ComfyUI-MossImage-owoinfer/bin/
```

- **Windows:** extract the `.zip` into `bin\` the same way.
- **Keep the archive's `bin/` and `lib/` together:** `moss-server` loads the engine library from `../lib`.
- **Which archive:**
  - `cu13` by default;
  - `cu12` when the driver is too old for `cu13` (see [Requirements](#requirements));
  - the `-system-cuda` archives are smaller and need a compatible CUDA toolkit installed.

The pack looks for `moss-server` in this order:
1. the `MOSSINFER_SERVER` environment variable, set where ComfyUI is started;
2. `bin/moss-server`, or `bin/*/bin/moss-server` (an extracted release; the newest version first);
3. `PATH`.

Restart ComfyUI. The nodes are under **MOSS-Image (owoinfer)**.

## Models

Download a model folder into `ComfyUI/models/moss_image/`. These are the revisions owoinfer was tested with:

```sh
hf download OpenMOSS-Team/MOSS-Image-SFT  --revision 2c7954437269461767f533b006823e8f85973a60 --local-dir ComfyUI/models/moss_image/MOSS-Image-SFT --exclude 'assets/*'
hf download OpenMOSS-Team/MOSS-Image-Base --revision 32edbc8a53ff3117ed52af34ce27a031129f2d0e --local-dir ComfyUI/models/moss_image/MOSS-Image-Base --exclude 'assets/*'
```

The Model Loader lists every folder up to three levels below `models/moss_image/` that has a `tokenizer/` and `transformer/config.json`, `text_encoder/config.json` and `vae/config.json`. With both models, you can save disk by hard-linking SFT's `text_encoder/*.safetensors` and `vae/*.safetensors` to Base's; they are byte-identical.

Optional single files replace one part of the folder. They go in ComfyUI's usual folders:

| File | Folder | Example |
|---|---|---|
| DiT | `models/diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` ([Comfy-Org](https://huggingface.co/Comfy-Org/MOSS-Image)), `moss_image_sft_int4_convrot.safetensors` ([toxicdog](https://huggingface.co/toxicdog/MOSS-Image-SFT-INT4-ConvRot-ComfyUI)) |
| Text encoder | `models/text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors` (Comfy-Org) |
| VAE | `models/vae/` | `flux2-vae.safetensors`, `flux2-vae_small.safetensors` (Comfy-Org) |
| LoRA, control adapter | `models/loras/` | `moss_image_color_depth_control.safetensors` (Comfy-Org) |

**Already have ComfyUI's single files?** When the loader takes all three parts (DiT, text encoder and VAE) from files, the folder needs only its configs and tokenizer, about 20 MB; it makes the same images as the full folder. Set `dit_file`, `text_encoder_file` and `vae_file` on the loader (the example workflow takes all three from the folder):

```sh
hf download OpenMOSS-Team/MOSS-Image-SFT --revision 2c7954437269461767f533b006823e8f85973a60 --local-dir ComfyUI/models/moss_image/MOSS-Image-SFT --include 'tokenizer/*' 'model_index.json' '*/config.json'
```

`moss_image` also works in `extra_model_paths.yaml`, as a folder of model folders or as one model folder:

```yaml
my_models:
  base_path: /data/models
  moss_image: MOSS-Image          # model folders under /data/models/MOSS-Image/
  diffusion_models: comfy-org
```

## Usage

Load `example_workflows/MOSS-Image_txt2img.json` (Model Loader → Generate → Save Image) by dropping it onto ComfyUI or from the templates browser. It expects a model folder named `MOSS-Image-SFT`.

### Nodes

**MOSS-Image Model Loader** → `MOSS_MODEL`

| Input | Values |
|---|---|
| `model_folder` | the folders found under `models/moss_image/` |
| `dit_file`, `text_encoder_file`, `vae_file` | `(from folder)` (default), or a file from `diffusion_models`, `text_encoders`, `vae` |
| `mode` | `balanced` (default), `min_memory`, `max_speed` |
| `tier` | `high` (default), `reference` (bf16 DiT and text encoder), `turbo` (INT4 and NVFP4 DiT files; NVFP4 on RTX 50 series cards only) |
| `attention` (advanced) | `default` (`reference`: exact; `high` and `turbo`: `q8_fp8`, or `q8_fp16` on GPUs without FP8: RTX 30 series, A100, A10, A40), `exact`, `q8_fp16`, `q8_fp16_f32acc`, `q8_fp8`, `q8_fp8_f32acc` |

The loader only describes the model; the engine loads it at the first run. A model file replaced under the same name makes the loader run again, and the engine loads the new file.

**MOSS-Image LoRA**: `MOSS_MODEL` → `MOSS_MODEL`.
- Inputs: `lora_name` from `models/loras/`, and a `strength` (default 1).
- Chain several; they are applied in order. Changing them does not reload the model, and a LoRA file replaced under the same name is read anew.

**MOSS-Image Control** → `MOSS_CONTROL`. Connect the last node of a chain to Generate's `control`.

| Input | Values |
|---|---|
| `adapter` | from `models/loras/` |
| `kind` | one of the adapter's kinds, e.g. `depth` or `color` |
| `image` | the conditioning image |
| `strength` | 0 to 10 |
| `start`, `end` | the share of the steps it is active in |
| `lora_strength` | the adapter's LoRA part; the same on every node of a chain |
| `control` | optional: chain another kind for the same adapter |

**MOSS-Image Generate** → `IMAGE`

| Input | Default | Notes |
|---|---|---|
| `moss_model` | | from the loader or a LoRA node |
| `prompt`, `negative_prompt` | | multiline |
| `width`, `height` | 1024 | multiples of 16, at most 4096 per side and 4.2 MP in all |
| `steps`, `cfg`, `seed` | 50, 3.0, 42 | seed 0 to 2^53 - 1 |
| `sampler`, `scheduler` | `euler`, `moss` | `moss` with `euler` and tier `reference` is the official pipeline |
| `shift` | 5.0 | |
| `init_image`, `mask` | | optional: img2img and inpainting (mask 1 repaints, 0 keeps) |
| `control` | | optional, from MOSS-Image Control |
| `denoise_strength`, `resize` | 0.75, `crop` | with an init image |
| advanced: hires fix | off | `hires_width`, `hires_height` (0 × 0 is off), `hires_strength` 0.45, `hires_steps` (0 uses `steps`) |
| advanced: accelerators | off | `cfg_truncation`, `step_cache`, `step_cache_threshold` (0 uses the engine's default); lossy |
| advanced: `after_run` | `keep` | `keep`, `release` (VRAM freed, weights kept in RAM), `unload` |
| advanced: `free_comfy_vram` | on | unload ComfyUI's own models before the run |
| advanced: other | the engine's | `uncond_skip_layers`, `slg_*`, `max_sequence_length`, `flow_multiplier`, `noise_layout`, `noise_source`, `eta`, `s_noise`, `preview` |

The engine checks the values itself. When it refuses one, ComfyUI shows its message as is, e.g. `Height and width must be positive multiples of 16.` The engine's warnings (a truncated prompt, LoRA layers it could not apply exactly, lossy accelerators) appear on the node and in ComfyUI's console log. The mask has the init image's size. Cancel also stops an `after_run` `release`.

## Limitations

- **A single checkpoint file can't be the model.** The loader picks a folder (at least its configs and tokenizer); ComfyUI files load next to it.
- **The tier must suit the files.**
  - `reference` runs bf16 only: a quantized DiT or text encoder file, or an 8-bit `attention`, is refused.
  - `high` refuses an INT4 or NVFP4 DiT, and `exact` attention with a bf16 DiT (that is `reference`).
  - `turbo` runs only an INT4 or NVFP4 DiT file.

  A tier the DiT file doesn't run is refused before the run loads anything, with the engine's reason, e.g. `the loaded DiT is INT4 ConvRot, which runs the Turbo tier; use tier turbo`. The other combinations, and any with the folder's own DiT, are refused by the engine at the first run, also with its reason.
- **One image per run, one run at a time.** A batch on `init_image`, `mask` or a Control `image` uses its first item, with a warning. ComfyUI's queue runs generations in turn.
- **Not with ComfyUI's model nodes:** no `MODEL` / `CLIP` / `VAE` / `LATENT`, no KSampler, no ComfyUI LoRA loaders.
- **Formats the engine does not read:**
  - GGUF;
  - FP8 (`fp8_scaled`, `float8_e4m3fn`);
  - torchao, MLX and SDNQ;
  - `.bin` / `.pt` files;
  - NVFP4 DiT files on GPUs other than the RTX 50 series.

  The engine refuses them with a message that names what it found.
- **No image editing models** (MOSS-Image-Edit and similar): the engine has no reference-image input.
- **Other GPUs aren't supported:** below compute capability 8.0, 9.x to 11.x, AMD, Intel, Apple, macOS, and CPU only.

### Sharing the GPU

- **With ComfyUI's own models.** ComfyUI's memory manager can't see the VRAM the engine holds. With `after_run` `keep`, a torch-heavy node later in the workflow (an upscaler, another model) can run out of memory. Set `after_run` to `release` on the last Generate before such nodes. `free_comfy_vram` covers the other direction: ComfyUI's models leave VRAM before the engine runs.
- **With another engine.** Two ComfyUI instances on one GPU, or ComfyUI and another owoinfer application such as MossStudio, each plan against the VRAM free at the time. Each can push the other out of memory or into slower streaming. Run one instance per GPU, or use `min_memory` and `after_run` `release` in each. Several `moss-server` processes, of any engine version, can share one cache folder safely; they also share its prompt-embedding hits.

## Troubleshooting

- **"moss-server not found":** extract a release into the pack's `bin/` or set `MOSSINFER_SERVER` (see [Getting moss-server](#getting-moss-server)). The error names the places it looked.
- **`out_of_memory`:** use `min_memory` mode or a smaller image, and check that nothing else holds the GPU (see [Sharing the GPU](#sharing-the-gpu)).
- **`model` error at the first run:** the folder is incomplete or not a MOSS-Image folder, or a file is in a format the engine doesn't read. The message names the file and what was found.
- **"`dit_file` … is a LoRA, not a DiT"** (or another slot and kind): the file is in the wrong ComfyUI folder or picked in the wrong input. Move it, or pick it where it belongs: DiTs in `diffusion_models`, text encoders in `text_encoders`, VAEs in `vae`, LoRAs and control adapters in `loras`.
- **"LoRA … can't be used: …":** the engine read the file and refuses it, e.g. a LoRA for another model; the rest of the message is the engine's reason.
- **"control kind '…': … has the kinds …":** set the Control node's `kind` to one of the kinds listed.
- **"… is not supported by this moss-server (…); it accepts: …":** the node offers a name that this `moss-server` lacks. Pick one from the list, or update `moss-server`.
- **"… does not speak protocol 2 …; the pack needs a MossInfer 2.x moss-server …":** the `moss-server` found is from another major version, e.g. a 1.x release. Extract a 2.x release (see [Getting moss-server](#getting-moss-server)).
- **"tier: 'auto' not in ['reference', 'high', 'turbo']" when queueing:** the workflow was saved with tier `auto`, which MossInfer 2.0 removed. Pick a tier on the Model Loader: `high` for a bf16 or INT8 DiT, `turbo` for an INT4 or NVFP4 one.
- **Engine log:** `ComfyUI/user/ComfyUI-MossImage-owoinfer/ComfyUI-MossImage-owoinfer.moss-server.log`, with the previous session's log as `.log.1`. Attach it to bug reports.

**Files the pack writes:**
- the log above;
- the engine's prompt-embedding cache and hardware profile in `ComfyUI/user/ComfyUI-MossImage-owoinfer/cache/`;
- a run's temporary PNGs (init image, mask, control images, previews, the output) in `ComfyUI/temp/ComfyUI-MossImage-owoinfer/`, deleted when the run ends.

**Security:**
- **Which program runs:** only the person who starts ComfyUI decides, through `MOSSINFER_SERVER`, the pack's `bin/` or `PATH`. No workflow value, and no file under ComfyUI's `user/` folder (which ComfyUI's web API can write), takes part.
- **Paths:** the pack sends `moss-server` only paths it chose itself, from ComfyUI's model folders and its own temp files.
- **Images:** images reach the engine only as PNGs re-encoded from ComfyUI tensors, so the engine's own image decoder never sees an uploaded or downloaded file.

## Roadmap

- **Distill bundles** and the `dataspace` sampler: the engine supports them; the pack needs a bundle-folder picker and unset steps / CFG.
- **Preload** when the model is selected (the engine's `preload`).
- **More loader settings:** VRAM budget and reserve, GPU index, LoRA strictness, text-encoder pinning, DiT read mode.
- **`MODEL`, `CLIP` and `VAE` nodes that work with KSampler**, once the engine's step-by-step API lands. Generate stays the fast path.
- **GGUF DiT files** and **MOSS-Image-Edit adapters**, once the engine reads them.

## Development

Run the tests from the pack folder with ComfyUI's Python:

```sh
python -m pytest tests                      # unit and headless-ComfyUI tests with a fake moss-server; no GPU
OWOINFER_GPU_TESTS=1 python -m pytest tests # adds the GPU smoke tests (MOSS_SERVER=<path to moss-server>)
```

With `MOSS_SERVER` set to a MossInfer 2.0 `moss-server`, the plain run also checks the engine's `inspect` answers on the community model files and that every name the nodes offer is one the server accepts; this needs no GPU. These tests need the model files under `MOSS_MODELS` (default `/workspace/Models`: the MOSS-Image folders and the Comfy-Org, toxicdog and akhaliq files), and without `MOSS_SERVER` they use the GPU tests' default server path (`tests/helpers.py`, `GPU_SERVER`); they skip, saying why, when either is missing or the server is older than 2.0.

The headless tests start ComfyUI from `COMFYUI_DIR` with a scratch base directory, `--cpu`, their own database and bytecode writing off. A test checks that nothing in the install changed. ComfyUI always reads `<install>/extra_model_paths.yaml` if one exists, so its folders show up in the tests too; the tests warn when it exists.

## Credits

- [MOSS-Image](https://huggingface.co/OpenMOSS-Team) by the OpenMOSS team.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- The single-file models from [Comfy-Org](https://huggingface.co/Comfy-Org/MOSS-Image), [toxicdog](https://huggingface.co/toxicdog) and [BennyDaBall](https://huggingface.co/BennyDaBall/MOSS-Image-SFT-NVFP4).

## License

Apache-2.0, like owoinfer. The MOSS-Image weights are published by OpenMOSS under Apache-2.0 and downloaded separately.

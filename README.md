# ComfyUI-MossImage.cpp

ComfyUI nodes for [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) (the 5B DiT with the Qwen3.5-9B text encoder and the FLUX.2 VAE), running on [**mossimage.cpp**](https://github.com/mossimage-cpp/mossimage.cpp) (formerly owoinfer; its release archives are still named `mossinfer-*`), a native C++/CUDA inference engine with no PyTorch at run time.

The engine runs as its own process, `moss-server`, which the pack starts at the first run and talks to over pipes. There is one per ComfyUI process: it is reused across runs, loads a model only when the loader's settings change, and stops when ComfyUI exits. The nodes take a prompt and optional images and return an `IMAGE`. They do not use ComfyUI's `MODEL`, `CLIP`, `VAE` or `LATENT`, so they work in place of KSampler, not with it.

## Why use this instead of ComfyUI's native MOSS-Image support?

ComfyUI runs MOSS-Image natively, and if that works well for you, there is no need for this pack. Use it when:

- **VRAM is tight.** mossimage.cpp plans its memory against the GPU it finds and streams the model from RAM or disk when it does not fit, instead of failing. It targets cards with 8 GiB of VRAM or more. On an RTX 5090 its peak was 0.34–0.66× of ComfyUI's (see [Performance](#performance)).
- **You want it faster.** With the same files, it was 1.04–1.25× as fast as ComfyUI on an RTX 5090 (see [Performance](#performance)).
- **You want crash isolation.** A CUDA fault ends `moss-server`, not ComfyUI; the next run starts a new one.
- **You want the official pipeline.** The `moss` scheduler with the `euler` sampler and tier `reference` (the folder's bf16 models) reproduces the official MOSS-Image pipeline bit for bit.

The cost is that the nodes are a closed pipeline. You can't mix them with ComfyUI's samplers, latent nodes or LoRA loaders (see [Limitations](#limitations)).

## Features

- Text-to-image, img2img with a denoise strength, and inpainting with a `MASK`.
- **Samplers:** `euler`, `ab2`, `euler_ancestral`, `heun`, `dpmpp_2m`, `dpmpp_2m_sde`, `er_sde`, `res_multistep`.
- **Schedulers:** 7, including `moss`.
- **Sampler options:** skip layers, skip-layer guidance, and the hires fix.
- **Models:** MOSS-Image Base and SFT folders, fine-tunes with the same architecture, and single-file DiT, text encoder and VAE files, such as the Comfy-Org INT8 ConvRot files. GGUF DiT files, converted once (see [GGUF DiT files](#gguf-dit-files)).
- **LoRAs:** any number, chained. Formats: kohya, PEFT / diffusers, LyCORIS LoHa and LoKr, and full deltas.
- **Image editing:** edit adapters such as `chinoll/Moss-Image-Edit`, with up to 4 reference images (see [Editing images](#editing-images)).
- **Control adapters**, such as the colour/depth control, and the lossy accelerators (CFG truncation, First-Block cache).
- **Quality tiers:** `reference` (bf16, exact attention), `high` (the default: the DiT's INT8 or bf16 weights with 8-bit attention), `turbo` (INT4 and NVFP4 DiT files). **Memory modes:** `balanced`, `min_memory`, `max_speed`.
- Progress bar over the whole run (text encoder and model loads included) with the time remaining, latent previews (off when ComfyUI's preview method is `none`) and the engine's warnings on the node. ComfyUI's Cancel stops a run within one layer or block.
- **After a run:** keep the model in VRAM, release the VRAM and keep the weights in RAM, or unload it.

Two checks run before anything loads:
- **Files:** the engine inspects each model file the run uses (DiT, text encoder, VAE, LoRAs, control adapter) and says what it is. A file in the wrong slot, a LoRA the engine can't use, a tier the DiT doesn't run, or a control `kind` the adapter doesn't have stops the run with the engine's reason. Results are kept per file until the file or the engine changes.
- **Names:** the sampler, scheduler, attention, mode, tier and the other choices are checked against the names the server accepts, so a server that lacks one says so and lists what it has.

## Performance

Benchmarks against ComfyUI's native MOSS-Image support. They were measured before 2.0 and have not been re-run with 2.1. For the engine's current numbers, see its [README](https://github.com/mossimage-cpp/mossimage.cpp#readme) and [integration.md](https://github.com/mossimage-cpp/mossimage.cpp/blob/main/docs/integration.md#performance).

- **GPU:** RTX 5090 (a rental) at its 575 W power cap.
- **ComfyUI:** 0.42.0 (`7cebca5`), torch 2.14.1+cu130, comfy-kitchen 0.2.40, SageAttention 2.2.0. The official `image_moss_image_t2i` template.
- **mossimage.cpp:** `moss-server`, mode `balanced`. Measured with development builds before 2.0: `920bcf5` (bf16 and INT8, 2026-10-24/25) and `97c63d9` (NVFP4, 2026-10-28). The tier column names the 2.0 tier that runs the same configuration.
- **Files:** the same DiT files on both sides: Comfy-Org's `moss_image_sft_bf16` and `moss_image_sft_int8_convrot`, and BennyDaBall's `moss_image_sft_nvfp4`. The VAE is `flux2-vae` on both.
- **Run:** text to image, 50 steps, CFG 3. Time is the median of 3 warm runs (models loaded), from the request to the written PNG. VRAM is the process's peak in `nvidia-smi`.

| Size | DiT | ComfyUI attention | mossimage.cpp tier | ComfyUI | mossimage.cpp | Speed-up | Peak VRAM: ComfyUI / mossimage.cpp |
|---|---|---|---|---|---|---|---|
| 1024² | bf16 | default (SDPA) | `reference` | 27.66 s | 23.09 s | 1.20× | 25.9 / 12.2 GiB |
| 1024² | bf16 | `--use-sage-attention` | `high` | 25.29 s | 20.96 s | 1.21× | 25.9 / 12.2 GiB |
| 1024² | INT8 | default (comfy-kitchen INT8) | `high` | 10.87 s | 9.86 s | 1.10× | 21.5 / 7.8 GiB |
| 1024² | NVFP4 | default | `turbo` | 10.58 s | 8.45 s | 1.25× | 20.7 / 7.0 GiB |
| 2048² | bf16 | default (SDPA) | `reference` | 162.10 s | 136.43 s | 1.19× | 28.0 / 18.4 GiB |
| 2048² | bf16 | `--use-sage-attention` | `high` | 122.68 s | 101.44 s | 1.21× | 28.0 / 18.4 GiB |
| 2048² | INT8 | default (comfy-kitchen INT8) | `high` | 61.88 s | 59.71 s | 1.04× | 23.6 / 13.9 GiB |
| 2048² | NVFP4 | default | `turbo` | 56.92 s | 53.09 s | 1.07× | 22.7 / 13.1 GiB |

Speed-up is ComfyUI's time divided by mossimage.cpp's.

- **Speed:** mossimage.cpp was 1.19–1.21× as fast with bf16 weights, 1.04–1.10× with INT8 and 1.07–1.25× with NVFP4. The bf16 rows pair the same kind of attention; Comfy-Org's INT8 file makes ComfyUI use comfy-kitchen's INT8 attention, so those rows are 8-bit on both sides. Which attention ComfyUI ran with the NVFP4 file wasn't checked.
- **VRAM:** mossimage.cpp's peak was 0.34–0.66× of ComfyUI's. Part of ComfyUI's figure is cache, not a requirement: its dynamic VRAM keeps the text encoder resident.
- **The samplers differ:** ComfyUI ran the template's `er_sde` + `simple`, mossimage.cpp its default `euler` + `moss`. Both make the same number of model evaluations per step. mossimage.cpp skips layer 22 in the negative-prompt pass, as the official pipeline does (`uncond_skip_layers`); without that it is 1.3 % slower.
- **Scope:** one card, one rental. Other GPUs and the INT4 files haven't been compared.

## Requirements

- **GPU:** NVIDIA.
  - The standard packages cover compute capability 8.x and 12.x: RTX 30, 40 and 50 series, A100, A10, A40, L4, L40, RTX PRO Blackwell.
  - Turing (7.5: RTX 20 series, T4, GTX 16 series): the Windows `-sm75` package, or a build with `-DMOSSINFER_SM75=ON`.
  - Volta (7.0: Tesla V100): a CUDA 12 build with the preset `linux-cuda12-sm70`. It was merged after v2.1.1, so it is not among that release's assets.
  - On Turing and Volta only text-to-image without LoRAs runs: img2img, inpainting, the hires fix, references, LoRAs and control adapters answer `not_supported`. Attention `default` resolves to `exact`, an explicit 8-bit mode is refused, and so is NVFP4. Tier `high` needs an INT8 DiT file there: with the folder's bf16 models the engine refuses it, so use `reference`. The example workflow uses `high` with the folder's models, so change one or the other.
- **NVIDIA driver:**
  - for the `cu13` build: Linux ≥ 580.65.06, Windows ≥ 580.88;
  - for the `cu12` build: Linux ≥ 525.60.13, Windows ≥ 527.41;
  - RTX 50 series cards need the 570 series or newer either way.
- **OS:** the engine ships Linux x86-64 and Windows x64 packages. This pack is tested on Linux only; its code is written for Windows too.
- **ComfyUI** with the V3 node API (`comfy_api.latest`); tested with 0.33.0.
- **mossimage.cpp 2.0 or later**, for `moss-server`; 2.1 for GGUF DiT files and reference images; tested with 2.1.1 (see [Getting moss-server](#getting-moss-server)).
- **Disk:** about 27 GB per model folder, or about 20 MB for a folder of configs and tokenizer next to single files (see [Models](#models)). A GGUF DiT adds its converted file (see [GGUF DiT files](#gguf-dit-files)).
- **RAM:** the engine aims to stay within 4 GiB (`min_memory`), 8 GiB (`balanced`) or what is free minus 1 GiB (`max_speed`). Free RAM beyond that caches the model files.

## Installation

### Manual install

Clone the repository into `ComfyUI/custom_nodes`. It needs nothing beyond what ComfyUI already has (torch, numpy, Pillow), so there is no `pip install` step:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/mossimage-cpp/ComfyUI-MossImage.cpp
```

For the Windows portable build, run this in the `ComfyUI_windows_portable` folder:

```
git clone https://github.com/mossimage-cpp/ComfyUI-MossImage.cpp ComfyUI\custom_nodes\ComfyUI-MossImage.cpp
```

### Getting `moss-server`

Download a mossimage.cpp release for your OS from [github.com/mossimage-cpp/mossimage.cpp/releases](https://github.com/mossimage-cpp/mossimage.cpp/releases), check it against the release's `SHA256SUMS`, and extract it into the pack's `bin/` folder:

```sh
mkdir -p ComfyUI/custom_nodes/ComfyUI-MossImage.cpp/bin
tar -xzf mossinfer-2.1.1-linux-cu13-x86_64.tar.gz -C ComfyUI/custom_nodes/ComfyUI-MossImage.cpp/bin/
```

- **Top folder:** the tar has one, which `bin/*/bin/moss-server` below handles.
- **Windows:** extract the `.zip` into `bin\` the same way.
- **Keep the archive's `bin/` and `lib/` together:** `moss-server` loads the engine library from `../lib`.
- **Which archive:**
  - `cu13` by default;
  - `cu12` when the driver is too old for `cu13` (see [Requirements](#requirements));
  - the `-system-cuda` archives are smaller and need a compatible CUDA toolkit installed;
  - `-sm75` (Windows, `cu13`) adds Turing cards (see [Requirements](#requirements)).

The pack looks for `moss-server` in this order:
1. the `MOSSIMAGE_CPP_SERVER` environment variable, set where ComfyUI is started;
2. `bin/moss-server`, or `bin/*/bin/moss-server` (an extracted release; the newest version first);
3. `PATH`.

Restart ComfyUI. The nodes are under **MOSS-Image (mossimage.cpp)**.

## Coming from ComfyUI-MossImage-owoinfer

- **URLs:** the old GitHub URLs redirect. To update a clone: `git remote set-url origin https://github.com/mossimage-cpp/ComfyUI-MossImage.cpp`
- **Folder:** the old folder name still works.
- **Workflows:** the node ids changed from `OwoInferMossImage*` to `MossImageCpp*`, so a saved workflow shows its nodes as missing until you update it. Replace the old prefix in the workflow's JSON file, e.g. `sed -i 's/OwoInferMossImage/MossImageCpp/g' my_workflow.json`, or the same replacement in a text editor. The menu category changed to **MOSS-Image (mossimage.cpp)**.
- **Environment:** `MOSSINFER_SERVER` is now `MOSSIMAGE_CPP_SERVER`, and the old name is ignored. If you set it, rename it, or the pack looks in its `bin/` folder and on `PATH` instead. The engine's own `MOSSINFER_CUDA_LIBS` and `MOSSINFER_CUBLASLT` keep their names.
- **Files:** the log and cache now live in `user/ComfyUI-MossImage.cpp/`, so `user/ComfyUI-MossImage-owoinfer/` can be deleted; the embedding cache and hardware profile regenerate.

## Models

Download a model folder into `ComfyUI/models/moss_image/`. These are the revisions mossimage.cpp was tested with:

```sh
hf download OpenMOSS-Team/MOSS-Image-SFT  --revision 2c7954437269461767f533b006823e8f85973a60 --local-dir ComfyUI/models/moss_image/MOSS-Image-SFT --exclude 'assets/*'
hf download OpenMOSS-Team/MOSS-Image-Base --revision 32edbc8a53ff3117ed52af34ce27a031129f2d0e --local-dir ComfyUI/models/moss_image/MOSS-Image-Base --exclude 'assets/*'
```

The Model Loader lists every folder up to three levels below `models/moss_image/` that has a `tokenizer/` and `transformer/config.json`, `text_encoder/config.json` and `vae/config.json`. With both models, you can save disk by hard-linking SFT's `text_encoder/*.safetensors` and `vae/*.safetensors` to Base's; they are byte-identical.

Optional single files replace one part of the folder. They go in ComfyUI's usual folders:

| File | Folder | Example |
|---|---|---|
| DiT | `models/diffusion_models/` | `moss_image_sft_int8_convrot.safetensors` ([Comfy-Org](https://huggingface.co/Comfy-Org/MOSS-Image)), `moss_image_sft_int4_convrot.safetensors` ([toxicdog](https://huggingface.co/toxicdog/MOSS-Image-SFT-INT4-ConvRot-ComfyUI)) |
| GGUF DiT | `models/diffusion_models/` | `MOSS-Image-SFT-Q8_0.gguf` ([leejet](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF)), [unsloth](https://huggingface.co/unsloth/MOSS-Image-SFT-GGUF) |
| Text encoder | `models/text_encoders/` | `qwen3.5_9b_int8_convrot.safetensors`, `qwen3.5_9b_w4a8_gptq.safetensors` (Comfy-Org) |
| VAE | `models/vae/` | `flux2-vae.safetensors`, `flux2-vae_small.safetensors` (Comfy-Org) |
| LoRA, control adapter, edit adapter | `models/loras/` | `moss_image_color_depth_control.safetensors` (Comfy-Org), `adapter_model.safetensors` ([chinoll](https://huggingface.co/chinoll/Moss-Image-Edit)) |

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

### GGUF DiT files

The Model Loader offers the `.gguf` files in `diffusion_models`; ComfyUI's own loaders don't list them. The engine has no GGUF runtime: the first load converts the file into one of its own formats, and later loads use the converted file. **VRAM and speed are those of the format it runs as, not of the GGUF type:** a Q4_K file takes INT8's VRAM and runs at INT8's speed. What a GGUF saves is download size.

| Tensor types in the file | tier `high` | tier `reference` | tier `turbo` |
|---|---|---|---|
| BF16, F16, F32 only | runs as INT8 ConvRot | runs as bf16 | refused |
| any of Q8_0, Q6_K, Q5_K, Q5_1, Q5_0, Q4_K, Q4_1, Q4_0 | runs as INT8 ConvRot | refused | refused |
| any other, e.g. Q2_K, Q3_K, IQ\*, TQ\*, MXFP4, custom types | refused | refused | refused |

- **First load:** the node shows `converting GGUF DiT (first load)`. The engine's figure is about 30 to 35 s on a 32-thread laptop CPU. Cancel stops it, and the next load starts over.
- **Disk:** 5.4 GB (INT8) or 10.2 GB (bf16) per GGUF and tier, in `ComfyUI/user/ComfyUI-MossImage.cpp/cache/gguf/`. There is no size cap. To reclaim the space, delete the files by hand while no engine runs; the next load converts again.
- **Sweep:** each conversion, once its file is in place, removes converted files whose GGUF is gone from its folder, the older file of the same GGUF path and tier, and leftovers of stopped conversions. A GGUF you moved is therefore converted again.
- **DiT only:** GGUF text encoders and VAEs are refused.

## Usage

Load `example_workflows/MOSS-Image_txt2img.json` (Model Loader → Generate → Save Image) by dropping it onto ComfyUI or from the templates browser. It expects a model folder named `MOSS-Image-SFT`.

### Editing images

An edit adapter is an ordinary LoRA file trained to edit reference images.

1. Put the adapter in `models/loras/` and add it with the MOSS-Image LoRA node.
2. Add one MOSS-Image Reference node per image, up to 4, chained with the source image first, and connect the last one to Generate's `references`. `max_edge` is 512 by default (the engine notes that the adapter was trained at 256 and that its authors' workflow uses 512).
3. Describe the edit in the prompt.

The engine refuses `slg_scale` above 0 with reference images, a run in which no sampling step runs (an init image at `denoise_strength` 0 without a sampling hires pass), and more than 16384 reference tokens in all, where an image of w × h has (w/16)(h/16): two 2048² references are over, even though the node allows that `max_edge`.

On an INT8 DiT the adapter has a cost. These are the engine's numbers, from its dev box:
- **Side branch** (the default on INT8): about 12 % per step.
- **`lora_placement` `merge`:** nothing per step, but it requantizes at about 50 s per load, and moves the edit further from the reference implementation than the side branch or a baked file.
- **A baked file:** `moss-infer quantize <bundle> -o edit_int8.safetensors --lora <adapter>`, about 2 minutes once. Load it as `dit_file`, keep the Reference nodes, and drop the LoRA node.

On a bf16 DiT the default merges the adapter at load; changing its strength reloads the DiT.

### Nodes

**MOSS-Image Model Loader** → `MOSS_MODEL`

| Input | Values |
|---|---|
| `model_folder` | the folders found under `models/moss_image/` |
| `dit_file`, `text_encoder_file`, `vae_file` | `(from folder)` (default), or a file from `diffusion_models` (`.gguf` included), `text_encoders`, `vae` |
| `mode` | `balanced` (default), `min_memory`, `max_speed` |
| `tier` | `high` (default), `reference` (bf16 DiT and text encoder), `turbo` (INT4 and NVFP4 DiT files; NVFP4 on RTX 50 series cards only) |
| `attention` (advanced) | `default` (`reference`: exact; `high` and `turbo`: `q8_fp8`, or `q8_fp16` on GPUs without FP8: RTX 30 series, A100, A10, A40; Turing and Volta: `exact`), `exact`, `q8_fp16`, `q8_fp16_f32acc`, `q8_fp8`, `q8_fp8_f32acc` |

The loader only describes the model; the engine loads it at the first run. A model file replaced under the same name makes the loader run again, and the engine loads the new file.

**MOSS-Image LoRA**: `MOSS_MODEL` → `MOSS_MODEL`.
- Inputs: `lora_name` from `models/loras/`, and a `strength` (default 1).
- Chain several; they are applied in order. Changing a side-branch LoRA does not reload the model; changing a merged one does (an edit adapter on bf16 by default, LoHa, LoKr and full deltas; see `lora_placement`). A LoRA file replaced under the same name is read anew.

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

**MOSS-Image Reference** → `MOSS_REFERENCES`. Connect the last node of a chain to Generate's `references`.

| Input | Values |
|---|---|
| `image` | the reference image |
| `max_edge` | 16 to 2048, default 512; the same on every node of a chain |
| `references` | optional: the images before this one, so the first node of a chain is the source image |

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
| `references` | | optional, from MOSS-Image Reference |
| `denoise_strength`, `resize` | 0.75, `crop` | with an init image |
| advanced: hires fix | off | `hires_width`, `hires_height` (0 × 0 is off), `hires_strength` 0.45, `hires_steps` (0 uses `steps`) |
| advanced: accelerators | off | `cfg_truncation`, `step_cache`, `step_cache_threshold` (0 uses the engine's default); lossy |
| advanced: `after_run` | `keep` | `keep`, `release` (VRAM freed, weights kept in RAM), `unload` |
| advanced: `free_comfy_vram` | on | unload ComfyUI's own models before the run |
| advanced: `lora_placement` | `default` | where low-rank adapters go: `default` (the side branch; an edit adapter on bf16 is merged), `merge`, `side_branch`. Changing what is merged reloads the DiT |
| advanced: other | the engine's | `uncond_skip_layers`, `slg_*`, `max_sequence_length`, `flow_multiplier`, `noise_layout`, `noise_source`, `eta`, `s_noise`, `preview` |

The engine checks the values itself. When it refuses one, ComfyUI shows its message as is, e.g. `Height and width must be positive multiples of 16.` The engine's warnings (a truncated prompt, LoRA layers it could not apply exactly, lossy accelerators) appear on the node and in ComfyUI's console log. The mask has the init image's size. Cancel also stops an `after_run` `release`.

## Limitations

- **A single checkpoint file can't be the model.** The loader picks a folder (at least its configs and tokenizer); ComfyUI files load next to it.
- **The tier must suit the files.**
  - `reference` runs bf16 only: a quantized DiT or text encoder file, or an 8-bit `attention`, is refused.
  - `high` refuses an INT4 or NVFP4 DiT, and `exact` attention with a bf16 DiT (that is `reference`).
  - `turbo` runs only an INT4 or NVFP4 DiT file.

  A tier the DiT file doesn't run is refused before the run loads anything, with the engine's reason, e.g. `the loaded DiT is INT4 ConvRot, which runs the Turbo tier; use tier turbo`. The other combinations, and any with the folder's own DiT, are refused by the engine at the first run, also with its reason.
- **One image per run, one run at a time.** A batch on `init_image`, `mask`, or a Control or Reference `image` uses its first item, with a warning. ComfyUI's queue runs generations in turn.
- **Not with ComfyUI's model nodes:** no `MODEL` / `CLIP` / `VAE` / `LATENT`, no KSampler, no ComfyUI LoRA loaders.
- **Formats the engine does not read:**
  - GGUF DiT files with a type outside the [table](#gguf-dit-files), and GGUF text encoders and VAEs;
  - FP8 (`fp8_scaled`, `float8_e4m3fn`);
  - torchao, MLX and SDNQ;
  - `.bin` / `.pt` files;
  - NVFP4 DiT files on GPUs other than the RTX 50 series.

  The engine refuses them with a message that names what it found.
- **Turing and Volta run text-to-image without LoRAs only** (see [Requirements](#requirements)).
- **Other GPUs aren't supported:** below compute capability 7.0, 9.x to 11.x, AMD, Intel, Apple, macOS, and CPU only.

### Sharing the GPU

- **With ComfyUI's own models.** ComfyUI's memory manager can't see the VRAM the engine holds. With `after_run` `keep`, a torch-heavy node later in the workflow (an upscaler, another model) can run out of memory. Set `after_run` to `release` on the last Generate before such nodes. `free_comfy_vram` covers the other direction: ComfyUI's models leave VRAM before the engine runs.
- **With another engine.** Two ComfyUI instances on one GPU, or ComfyUI and another mossimage.cpp application such as MossStudio, each plan against the VRAM free at the time. Each can push the other out of memory or into slower streaming. Run one instance per GPU, or use `min_memory` and `after_run` `release` in each. Several `moss-server` processes, of any engine version, can share one cache folder safely; they also share its prompt-embedding hits.

## Troubleshooting

- **"moss-server not found":** extract a release into the pack's `bin/` or set `MOSSIMAGE_CPP_SERVER` (see [Getting moss-server](#getting-moss-server)). The error names the places it looked.
- **`out_of_memory`:** use `min_memory` mode or a smaller image, and check that nothing else holds the GPU (see [Sharing the GPU](#sharing-the-gpu)).
- **`model` error at the first run:** the folder is incomplete or not a MOSS-Image folder, or a file is in a format the engine doesn't read. The message names the file and what was found.
- **"`dit_file` … is a LoRA, not a DiT"** (or another slot and kind): the file is in the wrong ComfyUI folder or picked in the wrong input. Move it, or pick it where it belongs: DiTs in `diffusion_models`, text encoders in `text_encoders`, VAEs in `vae`, LoRAs and control adapters in `loras`.
- **"`dit_file` ….gguf: tier reference runs bf16 weights, but the GGUF DiT is …":** the tier refuses this GGUF's tensor types. Pick the tier the message names (see [GGUF DiT files](#gguf-dit-files)).
- **`io` error at a GGUF's first load:** the engine can't write to the cache folder, or it has no room for the converted file. The message names the folder.
- **`not_supported` on a Turing or Volta card:** img2img, inpainting, the hires fix, references, LoRAs and control adapters don't run there (see [Requirements](#requirements)). Remove them from the workflow.
- **"tier high with attention exact on a bf16 DiT and text encoder is the Reference configuration …":** on a Turing or Volta card, use tier `reference` with the folder's bf16 models, or an INT8 DiT file with `high` (see [Requirements](#requirements)).
- **"LoRA … can't be used: …":** the engine read the file and refuses it, e.g. a LoRA for another model; the rest of the message is the engine's reason.
- **"control kind '…': … has the kinds …":** set the Control node's `kind` to one of the kinds listed.
- **"… is not supported by this moss-server (…); it accepts: …":** the node offers a name that this `moss-server` lacks. Pick one from the list, or update `moss-server`.
- **"… does not speak protocol 2 …; the pack needs a mossimage.cpp 2.x moss-server …":** the `moss-server` found is from another major version, e.g. a 1.x release. Extract a 2.x release (see [Getting moss-server](#getting-moss-server)).
- **"tier: 'auto' not in ['reference', 'high', 'turbo']" when queueing:** the workflow was saved with tier `auto`, which the engine removed in 2.0. Pick a tier on the Model Loader: `high` for a bf16 or INT8 DiT, `turbo` for an INT4 or NVFP4 one.
- **Engine log:** `ComfyUI/user/ComfyUI-MossImage.cpp/ComfyUI-MossImage.cpp.moss-server.log`, with the previous session's log as `.log.1`. Attach it to bug reports.

**Files the pack writes:**
- the log above;
- the engine's prompt-embedding cache and hardware profile in `ComfyUI/user/ComfyUI-MossImage.cpp/cache/`;
- converted GGUF DiT files in `ComfyUI/user/ComfyUI-MossImage.cpp/cache/gguf/`, kept until you delete them or the engine sweeps them (see [GGUF DiT files](#gguf-dit-files));
- a run's temporary PNGs (init image, mask, control and reference images, previews, the output) in `ComfyUI/temp/ComfyUI-MossImage.cpp/`, deleted when the run ends.

**Security:**
- **Which program runs:** only the person who starts ComfyUI decides, through `MOSSIMAGE_CPP_SERVER`, the pack's `bin/` or `PATH`. No workflow value, and no file under ComfyUI's `user/` folder (which ComfyUI's web API can write), takes part.
- **Paths:** the pack sends `moss-server` only paths it chose itself, from ComfyUI's model folders and its own temp files.
- **Images:** images, reference images included, reach the engine only as PNGs re-encoded from ComfyUI tensors, so the engine's own image decoder never sees an uploaded or downloaded file.
- **Converted GGUFs:** the engine trusts the files in `cache/gguf/` as the model, without checking them against the GGUF, so that folder must be writable only by the account that runs ComfyUI. The pack's folder is outside the `user/<id>/` folders that ComfyUI's userdata API serves.

## Roadmap

- **Distill bundles** and the `dataspace` sampler: the engine supports them; the pack needs a bundle-folder picker and unset steps / CFG.
- **Preload** when the model is selected (the engine's `preload`).
- **More loader settings:** VRAM budget and reserve, GPU index, LoRA strictness, text-encoder pinning, DiT read mode.
- **`MODEL`, `CLIP` and `VAE` nodes that work with KSampler**, using the engine's component session (the `components` capability). Generate stays the fast path.
- **A dry run before loading,** with the engine's `check` (the `check` capability).

## Development

Run the tests from the pack folder with ComfyUI's Python:

```sh
python -m pytest tests                      # unit and headless-ComfyUI tests with a fake moss-server; no GPU
MOSSIMAGE_CPP_GPU_TESTS=1 python -m pytest tests # adds the GPU smoke tests (MOSS_SERVER=<path to moss-server>)
```

The GPU test of a GGUF DiT needs `MOSS_TEST_CACHE` set to a folder with 5.4 GB free, because `/tmp` may be tmpfs; without it, that test skips and says why.

With `MOSS_SERVER` set to a mossimage.cpp 2.0 `moss-server`, the plain run also checks the engine's `inspect` answers on the community model files and that every name the nodes offer is one the server accepts; this needs no GPU. These tests need the model files under `MOSS_MODELS` (default `/workspace/Models`: the MOSS-Image folders and the Comfy-Org, toxicdog and akhaliq files), and without `MOSS_SERVER` they use the GPU tests' default server path (`tests/helpers.py`, `GPU_SERVER`); they skip, saying why, when either is missing or the server is older than 2.0. The GGUF and edit adapter checks need a 2.1 server and the leejet and chinoll files, and skip the same way.

The headless tests start ComfyUI from `COMFYUI_DIR` with a scratch base directory, `--cpu`, their own database and bytecode writing off. A test checks that nothing in the install changed. ComfyUI always reads `<install>/extra_model_paths.yaml` if one exists, so its folders show up in the tests too; the tests warn when it exists.

## Credits

- [MOSS-Image](https://huggingface.co/OpenMOSS-Team) by the OpenMOSS team.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- The single-file models from [Comfy-Org](https://huggingface.co/Comfy-Org/MOSS-Image), [toxicdog](https://huggingface.co/toxicdog) and [BennyDaBall](https://huggingface.co/BennyDaBall/MOSS-Image-SFT-NVFP4).
- The GGUF files from [leejet](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF) and [unsloth](https://huggingface.co/unsloth/MOSS-Image-SFT-GGUF).
- The edit adapter from [chinoll](https://huggingface.co/chinoll/Moss-Image-Edit).

## License

Apache-2.0, like mossimage.cpp. The MOSS-Image weights are published by OpenMOSS under Apache-2.0 and downloaded separately.

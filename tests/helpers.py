"""Shared test helpers (importable as `helpers`: conftest puts tests/ on sys.path)."""

import json
import os
import stat
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE = os.path.join(ROOT, "tests", "fake_moss_server.py")


def make_fake_server(directory, python=sys.executable):
    """A `moss-server` executable in `directory` that runs the fake server."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "moss-server")
    with open(path, "w") as f:
        f.write(f'#!/bin/sh\nexec "{python}" "{FAKE}" "$@"\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def recorded(env):
    """The requests the fake received, as (method, params) pairs; ("argv", [...]) for each start."""
    out = []
    with open(env["FAKE_MOSS_RECORD"], encoding="utf-8") as f:
        for line in f:
            if line.startswith("argv "):
                out.append(("argv", json.loads(line[5:])))
            else:
                msg = json.loads(line)
                out.append((msg.get("method"), msg.get("params", {})))
    return out


def make_bundle(path):
    """A folder that passes the pack's bundle check (configs and tokenizer, no weights)."""
    for c in ("transformer", "text_encoder", "vae"):
        os.makedirs(os.path.join(path, c), exist_ok=True)
        with open(os.path.join(path, c, "config.json"), "w") as f:
            f.write("{}")
    os.makedirs(os.path.join(path, "tokenizer"), exist_ok=True)
    return path


MOSSINFER = "/workspace/Projects/MossInfer"
GPU_SERVER = os.environ.get("MOSS_SERVER", os.path.join(MOSSINFER, "build-g", "moss-server"))
MODELS = os.environ.get("MOSS_MODELS", "/workspace/Models")
SFT = os.path.join(MODELS, "MOSS-Image", "sft")
COMFY_ORG = os.path.join(MODELS, "community", "comfy-org")
DIT_INT8 = os.path.join(COMFY_ORG, "moss_image_sft_int8_convrot.safetensors")
TE_INT8 = os.path.join(COMFY_ORG, "qwen3.5_9b_int8_convrot.safetensors")
VAE_FULL = os.path.join(COMFY_ORG, "flux2-vae.safetensors")
LORA_4STEP = os.path.join(MODELS, "community", "akhaliq", "MOSS-Image-SFT-4-steps", "lora.safetensors")
GGUF_Q8 = os.path.join(MODELS, "community", "gguf", "leejet", "MOSS-Image-SFT-GGUF", "MOSS-Image-SFT-Q8_0.gguf")
EDIT_ADAPTER = os.path.join(MODELS, "community", "chinoll", "Moss-Image-Edit", "adapter_model.safetensors")
MIN_FREE_RAM_GIB = 10


def gpu_skip_reason(server=GPU_SERVER, files=(SFT, DIT_INT8, TE_INT8)):
    """Why the GPU tests cannot run here, or "". Never touches other processes."""
    import shutil
    import subprocess
    if os.environ.get("MOSSIMAGE_CPP_GPU_TESTS") != "1":
        return "GPU tests are opt-in: set MOSSIMAGE_CPP_GPU_TESTS=1 (run under the GPU lock)"
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi: no NVIDIA GPU"
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"nvidia-smi failed: {e}"
    users = [line for line in apps.splitlines() if "moss-server" in line]
    if users:
        return f"a moss-server not started by these tests holds VRAM ({'; '.join(users)}); not touching it"
    if not os.access(server, os.X_OK):
        return f"no moss-server at {server} (set MOSS_SERVER)"
    for f in files:
        if not os.path.exists(f):
            return f"model file not on this machine: {f}"
    with open("/proc/meminfo") as f:
        avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable:")) / 2**20
    if avail < MIN_FREE_RAM_GIB:
        return f"only {avail:.1f} GiB RAM available (need {MIN_FREE_RAM_GIB})"
    return ""


def write_safetensors(path, tensors, metadata=None, blobs=None):
    """tensors: name -> (dtype, shape); blobs: name -> bytes for U8 tensors."""
    blobs = blobs or {}
    header, data, offset = {}, b"", 0
    for name, (dtype, shape) in tensors.items():
        payload = blobs.get(name, b"\0" * 4)
        header[name] = {"dtype": dtype, "shape": shape if name not in blobs else [len(payload)],
                        "data_offsets": [offset, offset + len(payload)]}
        data += payload
        offset += len(payload)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + data)


def fake_model_file(path, inspect=None, tensors=None, metadata=None):
    """A small safetensors file that the fake server's `inspect` describes as `inspect`."""
    meta = dict(metadata or {})
    if inspect is not None:
        meta["fake_inspect"] = json.dumps(inspect)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    write_safetensors(str(path), tensors or {"w": ("BF16", [2])}, metadata=meta or None)
    return str(path)


Q8 = "8-bit attention (q8_fp8; q8_fp16 on GPUs without FP8)"
INT8_DIT = {"kind": "dit", "loadable": True, "precision": "int8", "quantization": "int8_convrot256", "tiers": ["high"],
            "tier_reasons": {"reference": "tier reference runs bf16 weights exactly, but the loaded DiT or text "
                                          "encoder file is quantized; load the bundle's bf16 models or use tier high",
                             "high": f"High tier: INT8 ConvRot GEMMs, {Q8}", "turbo": "the loaded DiT is INT8"}}
INT4_DIT = {"kind": "dit", "loadable": True, "precision": "w4a4", "quantization": "int4_convrot256", "tiers": ["turbo"],
            "tier_reasons": {"reference": "the loaded DiT is quantized",
                             "high": "the loaded DiT is INT4 ConvRot, which runs the Turbo tier; use tier turbo",
                             "turbo": f"Turbo tier: INT4 ConvRot GEMMs, {Q8}"}}
LORA = {"kind": "lora", "loadable": True, "base_model": "", "adapter_types": ["lora"], "lora_rank": 16,
        "lora_adapters": 254, "lora_unknown_keys": 0}
CONTROL = dict(LORA, kind="control_adapter", control_kinds=["color", "depth"], base_model="moss-base")

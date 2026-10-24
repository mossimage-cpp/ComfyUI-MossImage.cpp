"""Shared test helpers (importable as `helpers`: conftest puts tests/ on sys.path)."""

import json
import os
import stat
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
MIN_FREE_RAM_GIB = 10


def gpu_skip_reason(server=GPU_SERVER, files=(SFT, DIT_INT8, TE_INT8)):
    """Why the GPU tests cannot run here, or "". Never touches other processes."""
    import shutil
    import subprocess
    if os.environ.get("OWOINFER_GPU_TESTS") != "1":
        return "GPU tests are opt-in: set OWOINFER_GPU_TESTS=1 (run under the GPU lock)"
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

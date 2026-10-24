"""GPU smoke tests through headless ComfyUI with the real moss-server.

Small on purpose: 256x256, 2 to 4 steps. Opt-in with OWOINFER_GPU_TESTS=1, run
under the GPU lock (`flock /workspace/Projects/MossInfer/.gpu.lock pytest ...`).
Skipped, with the reason, when a moss-server the tests did not start holds VRAM.
"""

import os
import re
import shutil
import subprocess
import time

import numpy as np
import pytest
from PIL import Image

from comfy_harness import ComfyUI, comfyui_missing, find, load_workflow_prompt, make_base
from helpers import COMFY_ORG, GPU_SERVER, LORA_4STEP, MODELS, gpu_skip_reason

pytestmark = pytest.mark.gpu

DIT = "moss_image_sft_int8_convrot.safetensors"
TE = "qwen3.5_9b_int8_convrot.safetensors"


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    reason = comfyui_missing() or gpu_skip_reason(files=(os.path.join(MODELS, "MOSS-Image", "sft"),
                                                         os.path.join(COMFY_ORG, DIT), os.path.join(COMFY_ORG, TE)))
    if reason:
        pytest.skip(reason)
    tmp = tmp_path_factory.mktemp("e2e-gpu")
    base = make_base(str(tmp / "base"))
    yaml = tmp / "extra_model_paths.yaml"
    yaml.write_text(
        f"owoinfer_gpu:\n  base_path: {MODELS}\n  moss_image: MOSS-Image\n"
        "  diffusion_models: community/comfy-org\n  text_encoders: community/comfy-org\n"
        "  vae: community/comfy-org\n  loras: community/akhaliq/MOSS-Image-SFT-4-steps\n")
    c = ComfyUI(base, dict(os.environ, MOSSINFER_SERVER=GPU_SERVER),
                extra_args=["--extra-model-paths-config", str(yaml)])
    try:
        c.wait_ready()
        yield c
    finally:
        c.stop(timeout=120)


def workflow(comfy, **gen):
    prompt = load_workflow_prompt(comfy)
    find(prompt, "OwoInferMossImageModelLoader").update(model_folder="sft", dit_file=DIT, text_encoder_file=TE)
    g = find(prompt, "OwoInferMossImageGenerate")
    g.update(width=256, height=256, steps=4)
    g.update(gen)
    return prompt


def output_image(comfy, entry, node="3"):
    image = entry["outputs"][node]["images"][0]
    path = os.path.join(comfy.base, "output", image["subfolder"], image["filename"])
    with Image.open(path) as img:
        return path, np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def assert_ok(entry, comfy):
    assert entry["status"]["status_str"] == "success", (entry["status"], comfy.log()[-3000:])


def test_txt2img(comfy):
    entry = comfy.wait_history(comfy.queue(workflow(comfy)), timeout=900)
    assert_ok(entry, comfy)
    path, pixels = output_image(comfy, entry)
    assert pixels.shape == (256, 256, 3)
    assert pixels.std() > 0.05, "a flat image"
    os.makedirs(os.path.join(comfy.base, "input"), exist_ok=True)
    shutil.copy(path, os.path.join(comfy.base, "input", "gpu_init.png"))
    assert "MOSS-Image: seed 42" in comfy.log()


def test_img2img(comfy):
    prompt = workflow(comfy, prompt="the same scene as a watercolor painting", denoise_strength=0.5, seed=3)
    prompt["20"] = {"class_type": "LoadImage", "inputs": {"image": "gpu_init.png"}}
    find(prompt, "OwoInferMossImageGenerate")["init_image"] = ["20", 0]
    entry = comfy.wait_history(comfy.queue(prompt), timeout=900)
    assert_ok(entry, comfy)
    _, pixels = output_image(comfy, entry)
    assert pixels.shape == (256, 256, 3) and pixels.std() > 0.05


def test_lora(comfy):
    if not os.path.isfile(LORA_4STEP):
        pytest.skip(f"LoRA not on this machine: {LORA_4STEP}")
    prompt = workflow(comfy, cfg=1.0, seed=4)
    prompt["21"] = {"class_type": "OwoInferMossImageLoRA",
                    "inputs": {"moss_model": ["1", 0], "lora_name": "lora.safetensors", "strength": 1.0}}
    find(prompt, "OwoInferMossImageGenerate")["moss_model"] = ["21", 0]
    entry = comfy.wait_history(comfy.queue(prompt), timeout=900)
    assert_ok(entry, comfy)
    _, pixels = output_image(comfy, entry)
    assert pixels.std() > 0.05
    assert "MOSS-Image: seed 4, tier" in comfy.log() and "uncertified" in comfy.log().split("MOSS-Image: seed 4")[1]


def test_after_run_release(comfy):
    spec = comfy.get("/object_info")["OwoInferMossImageGenerate"]["input"]["required"]["after_run"]
    options = spec[1]["options"] if spec[0] == "COMBO" else spec[0]
    if "release" not in options:
        pytest.skip(f"{GPU_SERVER} has no 'release' capability; set MOSS_SERVER to a 1.1 build")
    entry = comfy.wait_history(comfy.queue(workflow(comfy, steps=2, seed=12, after_run="release")), timeout=600)
    assert_ok(entry, comfy)
    assert "MOSS-Image: released" in comfy.log()


def test_cancel_mid_run(comfy):
    # The txt2img prompt again: an embedding-cache hit, and the DiT is resident, so denoise starts at once.
    steps = 400
    prompt_id = comfy.queue(workflow(comfy, steps=steps, seed=9))
    comfy.wait_running(prompt_id)
    time.sleep(8)
    comfy.post("/interrupt")
    entry = comfy.wait_history(prompt_id, timeout=120)
    assert "execution_interrupted" in [m[0] for m in entry["status"]["messages"]]
    found = re.findall(r"moss-server: generate cancelled after (\d+) steps", comfy.log())
    assert found, comfy.log()[-3000:]
    assert 0 < int(found[-1]) < steps
    # the engine is reused after the cancel
    entry = comfy.wait_history(comfy.queue(workflow(comfy, steps=2, seed=10)), timeout=600)
    assert_ok(entry, comfy)


def test_exit_stops_the_server(comfy):
    comfy.stop(timeout=120)
    time.sleep(1)
    left = subprocess.run(["pgrep", "-f", "-x", f"{GPU_SERVER} --parent-pid {comfy.proc.pid}"],
                          capture_output=True, text=True).stdout.split()
    assert left == [], f"moss-server still running: {left}"

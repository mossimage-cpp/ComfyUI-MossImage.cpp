"""Headless ComfyUI with the fake moss-server: the real nodes, no GPU.

Starts the user's ComfyUI (its code and venv, a scratch base directory, --cpu),
posts the example workflow to /prompt and checks the IMAGE output.
"""

import copy
import json
import os
import time

import pytest
from PIL import Image

from comfy_harness import ComfyUI, comfyui_missing, find, load_workflow_prompt, make_base
from helpers import make_bundle, make_fake_server, recorded

pytestmark = pytest.mark.e2e

if comfyui_missing():
    pytest.skip(comfyui_missing(), allow_module_level=True)

NODE_IDS = ["OwoInferMossImageModelLoader", "OwoInferMossImageLoRA", "OwoInferMossImageControl", "OwoInferMossImageGenerate"]


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e-fake")
    base = make_base(str(tmp / "base"))
    make_bundle(os.path.join(base, "models", "moss_image", "MOSS-Image-SFT"))
    make_bundle(str(tmp / "extra" / "bundles" / "Extra-Bundle"))
    os.makedirs(os.path.join(base, "models", "loras"), exist_ok=True)
    with open(os.path.join(base, "models", "loras", "test_lora.safetensors"), "wb") as f:
        f.write(b"fake")
    os.makedirs(os.path.join(base, "input"), exist_ok=True)
    Image.new("RGB", (96, 64), (10, 200, 30)).save(os.path.join(base, "input", "init.png"))
    yaml = tmp / "extra_model_paths.yaml"
    yaml.write_text(f"owoinfer_e2e:\n  base_path: {tmp / 'extra'}\n  moss_image: bundles\n")
    env = dict(os.environ, MOSSINFER_SERVER=make_fake_server(str(tmp / "bin")),
               FAKE_MOSS_RECORD=str(tmp / "requests.jsonl"), FAKE_MOSS_STEP_DELAY="0.02")
    c = ComfyUI(base, env, extra_args=["--extra-model-paths-config", str(yaml)])
    c.env = env
    try:
        c.wait_ready()
        yield c
    finally:
        c.stop()


def small(prompt, **gen):
    p = copy.deepcopy(prompt)
    inputs = find(p, "OwoInferMossImageGenerate")
    inputs.update({"width": 64, "height": 48, "steps": 4})
    inputs.update(gen)
    return p


def generate_requests(comfy):
    return [p for m, p in recorded(comfy.env) if m == "generate"]


def test_nodes_registered(comfy):
    info = comfy.get("/object_info")
    for node_id in NODE_IDS:
        assert info[node_id]["category"] == "MOSS-Image (owoinfer)"
    assert info["OwoInferMossImageModelLoader"]["display_name"] == "MOSS-Image Model Loader"
    assert info["OwoInferMossImageModelLoader"]["output"] == ["MOSS_MODEL"]
    folders = info["OwoInferMossImageModelLoader"]["input"]["required"]["model_folder"]
    options = folders[1]["options"] if folders[0] == "COMBO" else folders[0]
    # models/moss_image/ and the extra_model_paths.yaml folder
    assert "MOSS-Image-SFT" in options and "Extra-Bundle" in options
    after_run = info["OwoInferMossImageGenerate"]["input"]["required"]["after_run"]
    options = after_run[1]["options"] if after_run[0] == "COMBO" else after_run[0]
    assert options == ["keep", "unload"]  # the fake 1.0.0 has no `release`


def test_example_workflow_makes_an_image(comfy):
    prompt = small(load_workflow_prompt(comfy))
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    images = entry["outputs"]["3"]["images"]
    assert len(images) == 1
    path = os.path.join(comfy.base, "output", images[0]["subfolder"], images[0]["filename"])
    with Image.open(path) as img:
        assert img.size == (64, 48)
        assert img.getpixel((0, 0)) == (42, 0, 100)  # the fake's colour for seed 42, txt2img
    sent = generate_requests(comfy)[-1]
    assert sent["prompt"].startswith("a red fox") and sent["seed"] == 42 and sent["save_metadata"] is False
    assert sent["preview_every"] == 1 and "init_image" not in sent
    load = [p for m, p in recorded(comfy.env) if m == "load"]
    assert len(load) == 1 and load[0]["model"].endswith(os.path.join("moss_image", "MOSS-Image-SFT"))
    assert load[0]["tier"] == "auto" and load[0]["cache_dir"].endswith(os.path.join("ComfyUI-MossImage-owoinfer", "cache"))
    # the job's temp files are gone
    assert os.listdir(os.path.join(comfy.base, "temp", "ComfyUI-MossImage-owoinfer")) == []
    assert "MOSS-Image: seed 42" in comfy.log()


def test_lora_chain_and_img2img(comfy):
    prompt = small(load_workflow_prompt(comfy), seed=5, denoise_strength=0.5)
    prompt["10"] = {"class_type": "OwoInferMossImageLoRA",
                    "inputs": {"moss_model": ["1", 0], "lora_name": "test_lora.safetensors", "strength": 0.8}}
    prompt["11"] = {"class_type": "OwoInferMossImageLoRA",
                    "inputs": {"moss_model": ["10", 0], "lora_name": "test_lora.safetensors", "strength": 0.3}}
    prompt["12"] = {"class_type": "LoadImage", "inputs": {"image": "init.png"}}
    gen = find(prompt, "OwoInferMossImageGenerate")
    gen["moss_model"] = ["11", 0]
    gen["init_image"] = ["12", 0]
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    sent = generate_requests(comfy)[-1]
    lora_path = os.path.join(comfy.base, "models", "loras", "test_lora.safetensors")
    assert sent["loras"] == [{"path": lora_path, "strength": 0.8}, {"path": lora_path, "strength": 0.3}]
    assert sent["denoise_strength"] == 0.5 and sent["resize"] == "crop" and sent["init_image"].endswith("-init.png")
    # LoRAs are per generation: the engine was not loaded again
    assert [m for m, _ in recorded(comfy.env)].count("load") == 1


def test_after_run_unload(comfy):
    loads = [m for m, _ in recorded(comfy.env)].count("load")
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=21, after_run="unload")))
    assert entry["status"]["status_str"] == "success"
    methods = [m for m, _ in recorded(comfy.env)]
    assert methods[-1] == "unload" and methods[-2] == "generate"
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=22)))
    assert entry["status"]["status_str"] == "success"
    assert [m for m, _ in recorded(comfy.env)].count("load") == loads + 1  # loaded again after the unload


def test_after_run_release_is_not_offered(comfy):
    """The fake 1.0.0 has no `release`: ComfyUI refuses the value before the node runs."""
    with pytest.raises(RuntimeError, match="after_run"):
        comfy.queue(small(load_workflow_prompt(comfy), seed=23, after_run="release"))


def test_server_message_shown_as_is(comfy):
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), width=1000, seed=6)))
    assert entry["status"]["status_str"] == "error"
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert errors[0]["exception_type"].endswith("MossInferError")
    assert errors[0]["exception_message"].rstrip("\n") == "Height and width must be positive multiples of 16."


def test_interrupt_cancels_the_job(comfy):
    prompt_id = comfy.queue(small(load_workflow_prompt(comfy), steps=400, seed=7))
    comfy.wait_running(prompt_id)
    time.sleep(1.0)
    comfy.post("/interrupt")
    entry = comfy.wait_history(prompt_id, timeout=60)
    kinds = [m[0] for m in entry["status"]["messages"]]
    assert "execution_interrupted" in kinds, kinds
    gen_id = [json.loads(line)["id"] for line in open(comfy.env["FAKE_MOSS_RECORD"])
              if not line.startswith("argv") and json.loads(line).get("method") == "generate"][-1]
    assert {"id": gen_id} in [p for m, p in recorded(comfy.env) if m == "cancel"]
    assert "moss-server: generate cancelled after" in comfy.log()
    # the next run works
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=8)))
    assert entry["status"]["status_str"] == "success"


def test_exit_shuts_the_server_down(comfy):
    comfy.stop()
    lines = [m for m, _ in recorded(comfy.env)]
    assert lines[-1] == "shutdown"
    # one long-lived server for every run (plus the one-shot capability probe)
    assert [p for m, p in recorded(comfy.env) if m == "argv"].count(["--parent-pid", str(comfy.proc.pid)]) == 1

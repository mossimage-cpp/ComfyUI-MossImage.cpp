"""Headless ComfyUI with a fake moss-server that has `inspect` and hello `values` (MossInfer 1.1).

The checks run in Generate, before anything loads: a non-auto tier the DiT does not
take, a file of the wrong kind in a slot, a LoRA the engine can't use, a control kind
the adapter lacks, and a name the server's `values` don't list.
"""

import copy
import json
import os

import pytest
from PIL import Image

from comfy_harness import ComfyUI, comfyui_missing, find, install_changes, load_workflow_prompt, make_base
from helpers import CONTROL, INT4_DIT, INT8_DIT, LORA, fake_model_file, make_bundle, make_fake_server, recorded

pytestmark = pytest.mark.e2e

if comfyui_missing():
    pytest.skip(comfyui_missing(), allow_module_level=True)

INT4_HEADER = {"_quantization_metadata": json.dumps({"layers": {"layers.0.proj.q": {"format": "convrot_w4a4"}}})}
INT8_HEADER = {"_quantization_metadata": json.dumps({"layers": {"layers.0.proj.q": {"format": "int8_tensorwise"}}})}


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e-inspect")
    base = make_base(str(tmp / "base"))
    models = os.path.join(base, "models")
    make_bundle(os.path.join(models, "moss_image", "MOSS-Image-SFT"))
    fake_model_file(os.path.join(models, "diffusion_models", "int8.safetensors"), INT8_DIT, metadata=INT8_HEADER)
    fake_model_file(os.path.join(models, "diffusion_models", "int4.safetensors"), INT4_DIT, metadata=INT4_HEADER)
    fake_model_file(os.path.join(models, "diffusion_models", "lora_as_dit.safetensors"), LORA)
    fake_model_file(os.path.join(models, "vae", "te_as_vae.safetensors"), {"kind": "text_encoder", "loadable": True,
                                                                         "precision": "int8", "quantization": "int8_convrot"})
    fake_model_file(os.path.join(models, "loras", "good.safetensors"), LORA)
    fake_model_file(os.path.join(models, "loras", "bad.safetensors"),
                    dict(LORA, loadable=False, note="bad.safetensors: a LoRA for another model (fake)"))
    fake_model_file(os.path.join(models, "loras", "control.safetensors"), CONTROL)
    os.makedirs(os.path.join(base, "input"))
    Image.new("RGB", (64, 48), (128, 128, 128)).save(os.path.join(base, "input", "depth.png"))
    env = dict(os.environ, MOSSINFER_SERVER=make_fake_server(str(tmp / "bin")),
               FAKE_MOSS_RECORD=str(tmp / "requests.jsonl"), FAKE_MOSS_STEP_DELAY="0.01",
               FAKE_MOSS_CAPS="preview,model_files,lora,tiers,samplers,image_input,hires,control,accel,release,"
                              "progress_fraction,inspect",
               FAKE_MOSS_VERSION="1.1.0", FAKE_MOSS_VALUES="1", FAKE_MOSS_VALUES_DROP="sampler:heun")
    marker = tmp / "before-start"
    marker.write_text("")
    c = ComfyUI(base, env, extra_args=["--preview-method", "auto"])
    c.env, c.marker = env, str(marker)
    try:
        c.wait_ready()
        yield c
    finally:
        c.stop()


def workflow(comfy, seed, loader=None, **gen):
    prompt = copy.deepcopy(load_workflow_prompt(comfy))
    find(prompt, "OwoInferMossImageModelLoader").update(loader or {})
    g = find(prompt, "OwoInferMossImageGenerate")
    g.update({"width": 64, "height": 48, "steps": 3, "seed": seed})
    g.update(gen)
    return prompt


def methods(comfy):
    return [m for m, _ in recorded(comfy.env)] if os.path.exists(comfy.env["FAKE_MOSS_RECORD"]) else []


def run_error(comfy, prompt):
    """The run's error message; asserts nothing was loaded or generated for it."""
    before = methods(comfy)
    entry = comfy.wait_history(comfy.queue(prompt))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert errors, entry["status"]
    sent = methods(comfy)[len(before):]
    assert "load" not in sent and "generate" not in sent, sent
    return errors[0]["exception_message"]


def ok(comfy, prompt):
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]


def test_tier_the_dit_does_not_take(comfy):
    msg = run_error(comfy, workflow(comfy, 1, {"dit_file": "int8.safetensors", "tier": "reference"}))
    assert "dit_file int8.safetensors: " + INT8_DIT["tier_reasons"]["reference"] in msg
    # INT4 with high: the header check lets it through at queue time; inspect refuses it
    msg = run_error(comfy, workflow(comfy, 2, {"dit_file": "int4.safetensors", "tier": "high"}))
    assert INT4_DIT["tier_reasons"]["high"] in msg


def test_tiers_the_dit_takes(comfy):
    ok(comfy, workflow(comfy, 3, {"dit_file": "int8.safetensors", "tier": "auto"}))
    ok(comfy, workflow(comfy, 4, {"dit_file": "int4.safetensors", "tier": "turbo"}))


def test_wrong_kind_in_a_slot(comfy):
    msg = run_error(comfy, workflow(comfy, 5, {"dit_file": "lora_as_dit.safetensors"}))
    assert "dit_file lora_as_dit.safetensors is a LoRA, not a DiT" in msg
    msg = run_error(comfy, workflow(comfy, 6, {"vae_file": "te_as_vae.safetensors"}))
    assert "vae_file te_as_vae.safetensors is a text encoder, not a VAE" in msg


def with_lora(prompt, name):
    prompt["40"] = {"class_type": "OwoInferMossImageLoRA",
                    "inputs": {"moss_model": ["1", 0], "lora_name": name, "strength": 1.0}}
    find(prompt, "OwoInferMossImageGenerate")["moss_model"] = ["40", 0]
    return prompt


def test_loras(comfy):
    msg = run_error(comfy, with_lora(workflow(comfy, 7), "bad.safetensors"))
    assert "LoRA bad.safetensors can't be used: bad.safetensors: a LoRA for another model (fake)" in msg
    msg = run_error(comfy, with_lora(workflow(comfy, 8), "control.safetensors"))
    assert "LoRA control.safetensors is a control adapter, not a LoRA" in msg
    ok(comfy, with_lora(workflow(comfy, 9), "good.safetensors"))


def with_control(prompt, kind):
    prompt["41"] = {"class_type": "LoadImage", "inputs": {"image": "depth.png"}}
    prompt["42"] = {"class_type": "OwoInferMossImageControl",
                    "inputs": {"adapter": "control.safetensors", "kind": kind, "image": ["41", 0], "strength": 1.0,
                               "start": 0.0, "end": 1.0, "lora_strength": 1.0}}
    find(prompt, "OwoInferMossImageGenerate")["control"] = ["42", 0]
    return prompt


def test_control_kinds(comfy):
    msg = run_error(comfy, with_control(workflow(comfy, 10), "normal"))
    assert "control kind 'normal': control.safetensors has the kinds color, depth" in msg
    ok(comfy, with_control(workflow(comfy, 11), "depth"))
    sent = [p for m, p in recorded(comfy.env) if m == "generate"][-1]
    assert sent["control"]["images"][0]["kind"] == "depth"


def test_name_the_server_lacks(comfy):
    msg = run_error(comfy, workflow(comfy, 12, sampler="heun"))
    assert msg.rstrip("\n") == ("sampler 'heun' is not supported by this moss-server (1.1.0); it accepts: "
                                "euler, ab2, euler_ancestral, dpmpp_2m, dpmpp_2m_sde, er_sde, res_multistep, dataspace")
    ok(comfy, workflow(comfy, 13, sampler="euler_ancestral"))


def test_inspect_results_are_cached(comfy):
    def inspected():
        return [p["path"] for m, p in recorded(comfy.env) if m == "inspect"]
    ok(comfy, workflow(comfy, 14, {"dit_file": "int8.safetensors"}))
    before = inspected()
    ok(comfy, workflow(comfy, 15, {"dit_file": "int8.safetensors"}))
    assert inspected() == before  # same file, same server: no new inspect
    assert any(p.endswith("int8.safetensors") for p in before)


def test_nothing_written_into_the_install(comfy):
    comfy.stop()
    assert install_changes(comfy.marker) == []

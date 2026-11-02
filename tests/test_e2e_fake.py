"""Headless ComfyUI with the fake moss-server: the real nodes, no GPU.

Starts the user's ComfyUI (its code and venv, a scratch base directory, --cpu),
posts the example workflow to /prompt and checks the IMAGE output.
"""

import copy
import json
import os
import time
import warnings

import pytest
from PIL import Image

from comfy_harness import (ComfyUI, comfyui_missing, find, install_changes, install_extra_paths,
                           load_workflow_prompt, make_base)
from helpers import make_bundle, make_fake_server, recorded, write_safetensors

pytestmark = pytest.mark.e2e

if comfyui_missing():
    pytest.skip(comfyui_missing(), allow_module_level=True)

NODE_IDS = ["MossImageCppModelLoader", "MossImageCppLoRA", "MossImageCppControl",
            "MossImageCppReference", "MossImageCppGenerate"]
PACK_DIR = "ComfyUI-MossImage.cpp"


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
    dits = os.path.join(base, "models", "diffusion_models")
    os.makedirs(dits)
    write_safetensors(os.path.join(dits, "test_int4.safetensors"), {"layers.0.proj.q.weight": ("I8", [4])},
                      metadata={"_quantization_metadata": json.dumps(
                          {"layers": {"layers.0.proj.q": {"format": "convrot_w4a4"}}})})
    with open(os.path.join(dits, "test_q8.gguf"), "wb") as f:
        f.write(b"GGUF")
    os.makedirs(tmp / "extra" / "dits")
    (tmp / "extra" / "dits" / "extra_q4.gguf").write_bytes(b"GGUF")
    yaml = tmp / "extra_model_paths.yaml"
    yaml.write_text(f"mossimage_e2e:\n  base_path: {tmp / 'extra'}\n  moss_image: bundles\n  diffusion_models: dits\n")
    if install_extra_paths():
        warnings.warn(f"ComfyUI also reads {install_extra_paths()}; its folders appear in these tests", stacklevel=2)
    env = dict(os.environ, MOSSIMAGE_CPP_SERVER=make_fake_server(str(tmp / "bin")),
               FAKE_MOSS_RECORD=str(tmp / "requests.jsonl"), FAKE_MOSS_STEP_DELAY="0.02")
    marker = tmp / "before-start"
    marker.write_text("")
    c = ComfyUI(base, env, extra_args=["--extra-model-paths-config", str(yaml), "--preview-method", "auto"])
    c.env = env
    c.marker = str(marker)
    try:
        c.wait_ready()
        yield c
    finally:
        c.stop()


def small(prompt, **gen):
    p = copy.deepcopy(prompt)
    inputs = find(p, "MossImageCppGenerate")
    inputs.update({"width": 64, "height": 48, "steps": 4})
    inputs.update(gen)
    return p


def generate_requests(comfy):
    return [p for m, p in recorded(comfy.env) if m == "generate"]


def test_nodes_registered(comfy):
    info = comfy.get("/object_info")
    for node_id in NODE_IDS:
        assert info[node_id]["category"] == "MOSS-Image (mossimage.cpp)"
    assert info["MossImageCppModelLoader"]["display_name"] == "MOSS-Image Model Loader"
    assert info["MossImageCppModelLoader"]["output"] == ["MOSS_MODEL"]
    folders = info["MossImageCppModelLoader"]["input"]["required"]["model_folder"]
    options = folders[1]["options"] if folders[0] == "COMBO" else folders[0]
    # models/moss_image/ and the extra_model_paths.yaml folder
    assert "MOSS-Image-SFT" in options and "Extra-Bundle" in options
    after_run = info["MossImageCppGenerate"]["input"]["required"]["after_run"]
    options = after_run[1]["options"] if after_run[0] == "COMBO" else after_run[0]
    assert options == ["keep", "release", "unload"]  # always listed; checked against the server at run time
    # nothing spawned moss-server at import or schema time
    assert not os.path.exists(comfy.env["FAKE_MOSS_RECORD"])


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
    assert load[0]["tier"] == "high" and load[0]["cache_dir"].endswith(os.path.join(PACK_DIR, "cache"))
    # the job's temp files are gone
    assert os.listdir(os.path.join(comfy.base, "temp", PACK_DIR)) == []
    assert "MOSS-Image: seed 42" in comfy.log()


def test_lora_chain_and_img2img(comfy):
    prompt = small(load_workflow_prompt(comfy), seed=5, denoise_strength=0.5)
    prompt["10"] = {"class_type": "MossImageCppLoRA",
                    "inputs": {"moss_model": ["1", 0], "lora_name": "test_lora.safetensors", "strength": 0.8}}
    prompt["11"] = {"class_type": "MossImageCppLoRA",
                    "inputs": {"moss_model": ["10", 0], "lora_name": "test_lora.safetensors", "strength": 0.3}}
    prompt["12"] = {"class_type": "LoadImage", "inputs": {"image": "init.png"}}
    gen = find(prompt, "MossImageCppGenerate")
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


def test_after_run_release_needs_the_capability(comfy):
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=23, after_run="release")))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert "after_run release needs a moss-server with the 'release' capability; this one (2.1.1)" in errors[0]["exception_message"]


def loader(prompt, **kw):
    find(prompt, "MossImageCppModelLoader").update(kw)
    return prompt


def test_saved_tier_auto_is_refused_when_queued(comfy):
    """A workflow saved with 1.x's tier `auto`: ComfyUI's own combo check refuses it."""
    prompt = loader(small(load_workflow_prompt(comfy), seed=24), tier="auto")
    with pytest.raises(RuntimeError) as e:
        comfy.queue(prompt)
    assert "tier: 'auto' not in ['reference', 'high', 'turbo']" in str(e.value)
    prompt = loader(small(load_workflow_prompt(comfy), seed=24), dit_file="nope.safetensors")
    with pytest.raises(RuntimeError, match="dit_file: 'nope.safetensors' not in"):
        comfy.queue(prompt)


def test_linked_tier_is_checked_by_the_server(comfy):
    """ComfyUI does not check a linked combo input: the server refuses the name at `load`."""
    prompt = loader(small(load_workflow_prompt(comfy), seed=29), dit_file="test_int4.safetensors")
    # ComfyUI's If/Else Switch passes any type through, so it can feed a combo input
    prompt["30"] = {"class_type": "ComfySwitchNode", "inputs": {"switch": True, "on_true": "auto", "on_false": "auto"}}
    find(prompt, "MossImageCppModelLoader")["tier"] = ["30", 0]
    entry = comfy.wait_history(comfy.queue(prompt))  # accepted: nothing to check before it runs
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert errors and errors[0]["node_type"] == "MossImageCppGenerate"
    assert '"tier" must be one of high, reference, turbo' in errors[0]["exception_message"]
    prompt["30"]["inputs"]["on_true"] = "turbo"
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]


def test_turbo_with_int4_is_sent_as_chosen(comfy):
    prompt = loader(small(load_workflow_prompt(comfy), seed=25), dit_file="test_int4.safetensors", tier="turbo")
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    load = [p for m, p in recorded(comfy.env) if m == "load"][-1]
    assert load["tier"] == "turbo" and load["dit_file"].endswith("test_int4.safetensors")


def test_replaced_file_runs_again(comfy):
    """fingerprint_inputs: the same workflow is cached, until a model file is replaced."""
    prompt = loader(small(load_workflow_prompt(comfy), seed=26), dit_file="test_int4.safetensors", tier="turbo")
    comfy.wait_history(comfy.queue(prompt))
    before = recorded(comfy.env)
    assert comfy.wait_history(comfy.queue(prompt))["status"]["status_str"] == "success"
    assert recorded(comfy.env) == before  # cached by ComfyUI: nothing sent
    path = os.path.join(comfy.base, "models", "diffusion_models", "test_int4.safetensors")
    with open(path, "ab") as f:
        f.write(b"\0" * 8)  # the file is replaced (another size and mtime)
    assert comfy.wait_history(comfy.queue(prompt))["status"]["status_str"] == "success"
    methods = [m for m, _ in recorded(comfy.env)][len(before):]
    assert methods == ["load", "generate"]  # the loader ran again, and the engine loaded the new file


def combo_options(info, node_id, name):
    spec = info[node_id]["input"]["required"][name]
    return spec[1]["options"] if spec[0] == "COMBO" else spec[0]


def test_gguf_is_offered_by_the_loader_only(comfy):
    info = comfy.get("/object_info")
    offered = combo_options(info, "MossImageCppModelLoader", "dit_file")
    assert offered[0] == "(from folder)" and offered[1:] == sorted(offered[1:])
    # models/diffusion_models/ and the extra_model_paths.yaml folder
    assert "test_q8.gguf" in offered and "extra_q4.gguf" in offered and "test_int4.safetensors" in offered
    native = combo_options(info, "UNETLoader", "unet_name")
    assert "test_int4.safetensors" in native
    assert not [name for name in native if name.endswith(".gguf")]


def test_gguf_reaches_the_server(comfy):
    prompt = loader(small(load_workflow_prompt(comfy), seed=33), dit_file="extra_q4.gguf")
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    load = [p for m, p in recorded(comfy.env) if m == "load"][-1]
    assert load["dit_file"].endswith(os.path.join("extra", "dits", "extra_q4.gguf"))
    assert load["cache_dir"].endswith(os.path.join(PACK_DIR, "cache"))


def test_replaced_gguf_runs_again(comfy):
    prompt = loader(small(load_workflow_prompt(comfy), seed=34), dit_file="test_q8.gguf")
    comfy.wait_history(comfy.queue(prompt))
    before = recorded(comfy.env)
    assert comfy.wait_history(comfy.queue(prompt))["status"]["status_str"] == "success"
    assert recorded(comfy.env) == before
    with open(os.path.join(comfy.base, "models", "diffusion_models", "test_q8.gguf"), "ab") as f:
        f.write(b"\0" * 8)
    assert comfy.wait_history(comfy.queue(prompt))["status"]["status_str"] == "success"
    assert [m for m, _ in recorded(comfy.env)][len(before):] == ["load", "generate"]


def with_references(prompt, *sizes, max_edges=None):
    """A chain of Reference nodes on EmptyImage, one per (width, height, batch), into Generate."""
    last = None
    for i, (width, height, batch) in enumerate(sizes):
        prompt[str(50 + i)] = {"class_type": "EmptyImage",
                               "inputs": {"width": width, "height": height, "batch_size": batch, "color": 0}}
        node = {"image": [str(50 + i), 0], "max_edge": (max_edges or [256] * len(sizes))[i]}
        if last is not None:
            node["references"] = [last, 0]
        last = str(60 + i)
        prompt[last] = {"class_type": "MossImageCppReference", "inputs": node}
    find(prompt, "MossImageCppGenerate")["references"] = [last, 0]
    return prompt


def test_reference_chain(comfy):
    prompt = with_references(small(load_workflow_prompt(comfy), seed=35), (96, 64, 1), (32, 48, 1))
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    sent = generate_requests(comfy)[-1]
    assert sent["references"]["max_edge"] == 256 and sent["lora_placement"] == "default"
    assert [os.path.basename(p).split("-", 1)[1] for p in sent["references"]["images"]] == ["reference0.png", "reference1.png"]
    # upstream first, each image re-encoded to a PNG the server could read during the run
    assert [p for m, p in recorded(comfy.env) if m == "references_seen"][-1]["sizes"] == [[96, 64], [32, 48]]
    assert os.listdir(os.path.join(comfy.base, "temp", PACK_DIR)) == []


def test_reference_batch_uses_the_first(comfy):
    prompt = with_references(small(load_workflow_prompt(comfy), seed=36), (48, 48, 2))
    entry = comfy.wait_history(comfy.queue(prompt))
    assert entry["status"]["status_str"] == "success", entry["status"]
    assert [p for m, p in recorded(comfy.env) if m == "references_seen"][-1]["sizes"] == [[48, 48]]
    assert "MOSS-Image Reference: image is a batch of 2; using the first" in comfy.log()


def test_reference_chain_must_agree_on_max_edge(comfy):
    prompt = with_references(small(load_workflow_prompt(comfy), seed=37), (48, 48, 1), (32, 32, 1), max_edges=[256, 512])
    entry = comfy.wait_history(comfy.queue(prompt))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert "chained MOSS-Image Reference nodes must use the same max_edge (256 and 512)" in errors[0]["exception_message"]


def test_too_many_references(comfy):
    prompt = with_references(small(load_workflow_prompt(comfy), seed=38), *[(16 * (i + 1), 16, 1) for i in range(5)])
    entry = comfy.wait_history(comfy.queue(prompt))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert "too many reference images: the engine takes at most 4" in errors[0]["exception_message"]


def test_workflow_saved_without_lora_placement(comfy):
    """The example workflow predates it: it has no widget value, and the API prompt no input."""
    info = comfy.get("/object_info")["MossImageCppGenerate"]
    assert info["input_order"]["optional"][-1] == "lora_placement"
    assert "lora_placement" not in info["input"]["required"]
    assert "lora_placement" not in find(load_workflow_prompt(comfy), "MossImageCppGenerate")
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=39)))
    assert entry["status"]["status_str"] == "success", entry["status"]
    assert generate_requests(comfy)[-1]["lora_placement"] == "default"
    entry = comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=40, lora_placement="side_branch")))
    assert entry["status"]["status_str"] == "success", entry["status"]
    assert generate_requests(comfy)[-1]["lora_placement"] == "side_branch"


def test_preview_method_none_sends_no_preview(comfy):
    comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=27), extra_data={"preview_method": "none"}))
    assert "preview_output" not in generate_requests(comfy)[-1]
    comfy.wait_history(comfy.queue(small(load_workflow_prompt(comfy), seed=28)))
    assert "preview_output" in generate_requests(comfy)[-1]


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
    assert "inspect" not in lines  # a server without the capability: the engine checks the files at the run
    # one long-lived server for every run, started by the first one
    assert [p for m, p in recorded(comfy.env) if m == "argv"].count(["--parent-pid", str(comfy.proc.pid)]) == 1


@pytest.fixture(scope="module")
def comfy_release(tmp_path_factory):
    """A second ComfyUI whose fake moss-server has `release`."""
    tmp = tmp_path_factory.mktemp("e2e-fake-release")
    base = make_base(str(tmp / "base"))
    make_bundle(os.path.join(base, "models", "moss_image", "MOSS-Image-SFT"))
    env = dict(os.environ, MOSSIMAGE_CPP_SERVER=make_fake_server(str(tmp / "bin")),
               FAKE_MOSS_RECORD=str(tmp / "requests.jsonl"), FAKE_MOSS_STEP_DELAY="0.02",
               FAKE_MOSS_CAPS="preview,model_files,lora,tiers,samplers,image_input,release",
               FAKE_MOSS_RELEASE_DELAY="0.5")
    c = ComfyUI(base, env)
    c.env = env
    try:
        c.wait_ready()
        yield c
    finally:
        c.stop()


def test_release_after_an_interrupted_run_is_not_cancelled(comfy_release):
    """The interrupt that stopped the job is still flagged when after_run releases: only a new one cancels it."""
    c = comfy_release
    prompt_id = c.queue(small(load_workflow_prompt(c), steps=400, seed=31, after_run="release"))
    c.wait_running(prompt_id)
    time.sleep(1.0)
    c.post("/interrupt")
    entry = c.wait_history(prompt_id, timeout=60)
    assert "execution_interrupted" in [m[0] for m in entry["status"]["messages"]]
    calls = [(m, p) for m, p in recorded(c.env)]
    release_at = [i for i, (m, _) in enumerate(calls) if m == "release"]
    assert release_at, calls
    assert [m for m, _ in calls[release_at[-1]:]] == ["release"]  # no cancel sent to the release
    assert "MOSS-Image: released 1.0 GiB of VRAM" in c.log()


def test_release_after_a_normal_run(comfy_release):
    c = comfy_release
    entry = c.wait_history(c.queue(small(load_workflow_prompt(c), seed=32, after_run="release")))
    assert entry["status"]["status_str"] == "success"
    assert [m for m, _ in recorded(c.env)][-2:] == ["generate", "release"]


def test_references_need_the_capability(comfy_release):
    c = comfy_release
    prompt = with_references(small(load_workflow_prompt(c), seed=41), (48, 48, 1))
    entry = c.wait_history(c.queue(prompt))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert errors and ("A reference image needs a moss-server with the 'references' capability; this one (2.1.1) "
                       "does not have it") in errors[0]["exception_message"]
    assert os.listdir(os.path.join(c.base, "temp", PACK_DIR)) == []
    # at its default it is left out, and such a server takes the request
    assert c.wait_history(c.queue(small(load_workflow_prompt(c), seed=42)))["status"]["status_str"] == "success"
    assert "lora_placement" not in [p for m, p in recorded(c.env) if m == "generate"][-1]
    entry = c.wait_history(c.queue(small(load_workflow_prompt(c), seed=43, lora_placement="merge")))
    errors = [m[1] for m in entry["status"]["messages"] if m[0] == "execution_error"]
    assert "lora_placement 'merge' needs a moss-server with the 'references' capability" in errors[0]["exception_message"]


def test_nothing_written_into_the_install(comfy):
    comfy.stop()
    assert install_changes(comfy.marker) == []

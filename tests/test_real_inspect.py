"""`inspect` and hello `values` from a real moss-server, on the community files. No GPU.

Runs when MOSS_SERVER names a moss-server with the `inspect` capability (MossInfer 1.1);
skipped, saying so, otherwise. inspect reads file headers only and needs no engine.
"""

import os

import pytest

from helpers import COMFY_ORG, DIT_INT8, GPU_SERVER, LORA_4STEP, MODELS, TE_INT8
from owoinfer_comfy import logic
from owoinfer_comfy.logic import ControlImage, MossControl, MossModel
from owoinfer_comfy.server import MossInferError, ServerManager

DIT_INT4 = os.path.join(MODELS, "community", "toxicdog", "moss_image_sft_int4_convrot.safetensors")
CONTROL = os.path.join(COMFY_ORG, "moss_image_color_depth_control.safetensors")
VAE = os.path.join(COMFY_ORG, "flux2-vae.safetensors")
FILES = [DIT_INT8, DIT_INT4, LORA_4STEP, CONTROL, TE_INT8, VAE]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    if not os.access(GPU_SERVER, os.X_OK):
        pytest.skip(f"no moss-server at {GPU_SERVER} (set MOSS_SERVER)")
    missing = [f for f in FILES if not os.path.isfile(f)]
    if missing:
        pytest.skip(f"model files not on this machine: {missing}")
    m = ServerManager(log_dir=str(tmp_path_factory.mktemp("real-inspect")))
    try:
        with m.session(GPU_SERVER) as s:
            if not s.has("inspect"):
                pytest.skip(f"{GPU_SERVER} ({s.version}) has no 'inspect' capability; set MOSS_SERVER to a 1.1 build")
            yield s
    finally:
        m.shutdown()


def test_files(session):
    r = session.inspect(FILES)
    assert r[DIT_INT8]["kind"] == "dit" and r[DIT_INT8]["loadable"] and r[DIT_INT8]["precision"] == "int8"
    assert "high" in r[DIT_INT8]["tiers"] and "reference" not in r[DIT_INT8]["tiers"]
    assert r[DIT_INT4]["kind"] == "dit" and r[DIT_INT4]["tiers"] == ["turbo"]
    assert r[LORA_4STEP]["kind"] == "lora" and r[LORA_4STEP]["loadable"] and r[LORA_4STEP]["lora_rank"] > 0
    assert r[CONTROL]["kind"] == "control_adapter" and set(r[CONTROL]["control_kinds"]) == {"color", "depth"}
    assert r[TE_INT8]["kind"] == "text_encoder" and r[TE_INT8]["precision"] == "int8"
    assert r[VAE]["kind"] == "vae"


def test_errors(session, tmp_path):
    r = session.inspect([str(tmp_path / "missing.safetensors"), str(tmp_path)])
    assert isinstance(r[str(tmp_path / "missing.safetensors")], MossInferError)
    assert r[str(tmp_path / "missing.safetensors")].code == "io"
    assert r[str(tmp_path)].code == "invalid_params"


def test_the_pack_checks_on_real_results(session):
    r = session.inspect(FILES)
    int8 = MossModel("/m", dit_file=DIT_INT8, tier="reference")
    problems = logic.inspected_problems(int8, None, r)
    assert len(problems) == 1 and problems[0].startswith("dit_file moss_image_sft_int8_convrot.safetensors: ")
    assert "reference" in problems[0]
    # auto with INT4: never refused from inspect (no certificates looked up)
    assert logic.inspected_problems(MossModel("/m", dit_file=DIT_INT4, tier="auto"), None, r) == []
    assert logic.inspected_problems(MossModel("/m", dit_file=DIT_INT4, tier="turbo"), None, r) == []
    # wrong kinds
    swapped = MossModel("/m", dit_file=LORA_4STEP, vae_file=TE_INT8).with_lora(CONTROL, 1.0)
    assert logic.inspected_problems(swapped, None, r) == [
        "dit_file lora.safetensors is a LoRA, not a DiT",
        "vae_file qwen3.5_9b_int8_convrot.safetensors is a text encoder, not a VAE",
        "LoRA moss_image_color_depth_control.safetensors is a control adapter, not a LoRA"]
    control = MossControl(CONTROL).with_image(ControlImage("normal", None))
    assert logic.inspected_problems(MossModel("/m"), control, r) == [
        "control kind 'normal': moss_image_color_depth_control.safetensors has the kinds color, depth"]


def test_the_node_lists_are_the_servers(session):
    """Every name the nodes offer is one this server accepts."""
    values = session.values
    assert values, "a server with inspect has hello values"
    for name, offered in (("mode", logic.MODES), ("tier", logic.TIERS), ("attention", logic.ATTENTIONS),
                          ("sampler", logic.SAMPLERS), ("scheduler", logic.SCHEDULERS),
                          ("noise_layout", logic.NOISE_LAYOUTS), ("noise_source", logic.NOISE_SOURCES),
                          ("resize", logic.RESIZE_MODES), ("step_cache", logic.STEP_CACHES)):
        assert set(offered) <= set(values[name]), (name, offered, values[name])

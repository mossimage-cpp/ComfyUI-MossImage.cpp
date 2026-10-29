"""`inspect` and hello `values` from a real moss-server, on the community files. No GPU.

Runs when MOSS_SERVER names a MossInfer 2.0 or later moss-server; skipped, saying so,
otherwise. inspect reads file headers only and needs no engine.
"""

import os

import pytest

from helpers import COMFY_ORG, DIT_INT8, GPU_SERVER, LORA_4STEP, MODELS, TE_INT8
from owoinfer_comfy import logic
from owoinfer_comfy.logic import ControlImage, MossControl, MossModel
from owoinfer_comfy.server import MossInferError, ServerManager

DIT_INT4 = os.path.join(MODELS, "community", "toxicdog", "moss_image_sft_int4_convrot.safetensors")
DIT_NVFP4 = os.path.join(MODELS, "community", "bennydaball", "moss_image_sft_nvfp4.safetensors")
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
        try:
            with m.session(GPU_SERVER) as s:
                yield s
        except MossInferError as e:
            if e.code != "unsupported_protocol":
                raise
            pytest.skip(f"{e}; set MOSS_SERVER to a 2.0 build")
    finally:
        m.shutdown()


def test_files(session):
    r = session.inspect(FILES)
    assert r[DIT_INT8]["kind"] == "dit" and r[DIT_INT8]["loadable"] and r[DIT_INT8]["precision"] == "int8"
    assert r[DIT_INT8]["tiers"] == ["high"] and set(r[DIT_INT8]["tier_reasons"]) == set(logic.TIERS)
    assert r[DIT_INT4]["kind"] == "dit" and r[DIT_INT4]["tiers"] == ["turbo"]
    for reason in (r[DIT_INT8]["tier_reasons"]["high"], r[DIT_INT4]["tier_reasons"]["turbo"]):
        assert "8-bit attention" in reason and "certif" not in reason
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
    high = logic.inspected_problems(MossModel("/m", dit_file=DIT_INT4, tier="high"), None, r)
    assert high == ["dit_file moss_image_sft_int4_convrot.safetensors: " + r[DIT_INT4]["tier_reasons"]["high"]]
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
    assert values["tier"] == logic.TIERS  # 2.0: exactly these, no `auto`


def test_nvfp4(session):
    if not os.path.isfile(DIT_NVFP4):
        pytest.skip(f"model file not on this machine: {DIT_NVFP4}")
    r = session.inspect([DIT_NVFP4])[DIT_NVFP4]
    if not r["loadable"] and "this build of MossInfer does not include" in r.get("note", ""):
        pytest.skip(f"this moss-server has no NVFP4 kernels: {r['note']}")
    assert r["kind"] == "dit" and r["precision"] == "nvfp4" and r["tiers"] == ["turbo"]
    assert "RTX 50 only" in r["tier_reasons"]["turbo"]

"""`inspect` and hello `values` from a real moss-server, on the community files. No GPU.

Runs when MOSS_SERVER names a mossimage.cpp 2.0 or later moss-server (2.1 for the GGUF
and edit adapter tests); skipped, saying so, otherwise. inspect reads file headers only and needs no engine.
"""

import os

import pytest

from helpers import COMFY_ORG, DIT_INT8, EDIT_ADAPTER, GGUF_Q8, GPU_SERVER, LORA_4STEP, MODELS, TE_INT8
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


def needs_21(session, *files):
    if not session.has("references"):
        pytest.skip(f"moss-server {session.version} is older than 2.1: no GGUF DiTs or reference images")
    for f in files:
        if not os.path.isfile(f):
            pytest.skip(f"model file not on this machine: {f}")


def test_gguf_dit(session):
    needs_21(session, GGUF_Q8)
    r = session.inspect([GGUF_Q8])[GGUF_Q8]
    assert r["kind"] == "dit" and r["loadable"] and r["precision"] == "int8"
    assert r["quantization"] == "GGUF Q8_0 + BF16 + F32, runs as INT8 ConvRot"
    assert r["tiers"] == ["high"] and set(r["tier_reasons"]) == set(logic.TIERS)
    name = os.path.basename(GGUF_Q8)
    assert logic.inspected_problems(MossModel("/m", dit_file=GGUF_Q8, tier="high"), None, {GGUF_Q8: r}) == []
    assert logic.inspected_problems(MossModel("/m", dit_file=GGUF_Q8, tier="reference"), None, {GGUF_Q8: r}) == [
        f"dit_file {name}: " + r["tier_reasons"]["reference"]]


def test_edit_adapter(session):
    needs_21(session, EDIT_ADAPTER)
    r = session.inspect([EDIT_ADAPTER])[EDIT_ADAPTER]
    assert r["kind"] == "lora" and r["loadable"] and r["lora_rank"] > 0 and r["lora_unknown_keys"] == 0
    assert logic.inspected_problems(MossModel("/m").with_lora(EDIT_ADAPTER, 1.0), None, {EDIT_ADAPTER: r}) == []


def test_lora_placements_are_the_servers(session):
    needs_21(session)
    assert session.values["lora_placement"] == logic.LORA_PLACEMENTS


def test_nvfp4(session):
    if not os.path.isfile(DIT_NVFP4):
        pytest.skip(f"model file not on this machine: {DIT_NVFP4}")
    r = session.inspect([DIT_NVFP4])[DIT_NVFP4]
    if not r["loadable"] and "this build of MossInfer does not include" in r.get("note", ""):
        pytest.skip(f"this moss-server has no NVFP4 kernels: {r['note']}")
    assert r["kind"] == "dit" and r["precision"] == "nvfp4" and r["tiers"] == ["turbo"]
    assert "RTX 50 only" in r["tier_reasons"]["turbo"]

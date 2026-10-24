"""Node logic without ComfyUI: folders, DiT kinds, tiers, request building, progress."""

import json
import os
import struct

import pytest

from helpers import make_bundle, recorded, write_safetensors
from owoinfer_comfy import logic
from owoinfer_comfy.logic import ControlImage, MossControl, MossModel, ProgressMapper
from owoinfer_comfy.server import MossInferError, ServerManager

COMMUNITY = "/workspace/Models/community"

ALL = {c: True for c in ("preview", "model_files", "lora", "tiers", "samplers", "image_input", "hires",
                         "distill", "control", "accel", "release", "preload", "progress_fraction")}
OLD = {"preview": True, "model_files": True, "lora": True, "tiers": True}  # a minimal build


def values(**kw):
    v = {"prompt": "p", "negative_prompt": "", "width": 256, "height": 256, "steps": 4, "cfg": 3.0, "shift": 5.0,
         "seed": 1, "uncond_skip_layers": "22", "slg_layers": "", "slg_scale": 0.0, "slg_start": 0.0,
         "slg_stop": 1.0, "max_sequence_length": 4096, "sampler": "euler", "scheduler": "moss",
         "flow_multiplier": 1.0, "noise_layout": "official", "noise_source": "philox", "eta": 1.0, "s_noise": 1.0,
         "denoise_strength": 0.75, "resize": "crop", "hires_width": 0, "hires_height": 0, "hires_strength": 0.45,
         "hires_steps": 0, "cfg_truncation": 1.0, "step_cache": "off", "step_cache_threshold": 0.0}
    v.update(kw)
    return v


# folders ----------------------------------------------------------------------------------

def test_find_bundles(tmp_path):
    root = tmp_path / "moss_image"
    make_bundle(str(root / "sft"))
    make_bundle(str(root / "MOSS-Image" / "base"))
    make_bundle(str(root / "sft" / "inner"))  # inside a bundle: not listed
    os.makedirs(root / "empty" / "x")
    (root / "sft-incomplete" / "transformer").mkdir(parents=True)
    other = tmp_path / "other"
    make_bundle(str(other / "sft"))  # same name in a later root: the first wins
    make_bundle(str(other / "extra"))
    direct = tmp_path / "direct-bundle"
    make_bundle(str(direct))
    found = logic.find_bundles([str(root), str(other), str(direct), str(tmp_path / "missing")])
    assert found == {"MOSS-Image/base": str(root / "MOSS-Image" / "base"), "direct-bundle": str(direct),
                     "extra": str(other / "extra"), "sft": str(root / "sft")}


def test_find_bundles_depth_limit(tmp_path):
    make_bundle(str(tmp_path / "a" / "b" / "c" / "d"))
    assert logic.find_bundles([str(tmp_path)], max_depth=3) == {}
    assert list(logic.find_bundles([str(tmp_path)], max_depth=4)) == ["a/b/c/d"]


# DiT kinds and tiers -----------------------------------------------------------------------

def test_dit_quant_kind_synthetic(tmp_path):
    int4 = tmp_path / "int4.safetensors"
    write_safetensors(int4, {"layers.0.proj.q.weight": ("I8", [4])},
                      metadata={"_quantization_metadata": json.dumps(
                          {"layers": {"layers.0.proj.q": {"format": "convrot_w4a4"}}})})
    int8 = tmp_path / "int8.safetensors"
    q = json.dumps({"format": "int8_tensorwise", "convrot": True}).encode()
    write_safetensors(int8, {"layers.0.proj.q.weight": ("I8", [4]), "layers.0.proj.q.comfy_quant": ("U8", [0])},
                      blobs={"layers.0.proj.q.comfy_quant": q})
    nvfp4 = tmp_path / "nvfp4.safetensors"
    write_safetensors(nvfp4, {"a.comfy_quant": ("U8", [0]), "b.comfy_quant": ("U8", [0])},
                      blobs={"a.comfy_quant": q, "b.comfy_quant": json.dumps({"format": "nvfp4"}).encode()})
    bf16 = tmp_path / "bf16.safetensors"
    write_safetensors(bf16, {"layers.0.proj.q.weight": ("BF16", [2]), "x": ("F32", [1])})
    fp8 = tmp_path / "fp8.safetensors"
    write_safetensors(fp8, {"layers.0.proj.q.weight": ("F8_E4M3", [4])})
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert [logic.dit_quant_kind(str(p)) for p in (int4, int8, nvfp4, bf16, fp8, junk, tmp_path / "missing")] == \
        ["int4", "int8", "nvfp4", "bf16", "fp8", "unknown", "unknown"]


REAL_FILES = {
    "toxicdog/moss_image_sft_int4_convrot.safetensors": "int4",
    "comfy-org/moss_image_sft_int8_convrot.safetensors": "int8",
    "toxicdog/moss_image_sft_turbo_int8_convrot.safetensors": "int8",
    "bennydaball/moss_image_sft_nvfp4.safetensors": "nvfp4",
}


@pytest.mark.parametrize("name,kind", REAL_FILES.items())
def test_dit_quant_kind_community_files(name, kind):
    path = os.path.join(COMMUNITY, name)
    if not os.path.isfile(path):
        pytest.skip(f"community file not on this machine: {path}")
    assert logic.dit_quant_kind(path) == kind


def test_dit_quant_kind_malformed_headers(tmp_path):
    """Crafted headers give "unknown", never an exception."""
    cases = {
        "qmeta_dict": {"__metadata__": {"_quantization_metadata": {"layers": {}}}, "w": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}},
        "qmeta_layers_list": {"__metadata__": {"_quantization_metadata": json.dumps({"layers": [1, 2]})}},
        "qmeta_not_json": {"__metadata__": {"_quantization_metadata": "{oops"}},
        "metadata_list": {"__metadata__": ["x"]},
        "offsets_null": {"a.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": None}},
        "offsets_strings": {"a.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": ["0", "4"]}},
        "offsets_backwards": {"a.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": [8, 2]}},
        "tensor_not_object": {"w": 5},
    }
    for name, header in cases.items():
        raw = json.dumps(header).encode()
        path = tmp_path / f"{name}.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 16)
        assert logic.dit_quant_kind(str(path)) in ("unknown", "bf16", "fp8"), name
    top_list = tmp_path / "list.safetensors"
    top_list.write_bytes(struct.pack("<Q", 2) + b"[]")
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", logic.MAX_HEADER_BYTES + 1) + b"{}")
    assert logic.dit_quant_kind(str(top_list)) == "unknown" and logic.dit_quant_kind(str(huge)) == "unknown"


def test_tier_problem():
    assert logic.tier_problem("auto", None) == ""
    assert logic.tier_problem("high", "int8") == ""
    assert logic.tier_problem("reference", "int8") == ""  # the engine refuses it, with its message
    assert logic.tier_problem("turbo", "int4") == ""
    # auto is sent unchanged, never rewritten: the user picks turbo
    assert logic.tier_problem("auto", "int4") == "INT4 DiT file: choose tier turbo (lossy, uncertified)"
    assert logic.tier_problem("high", "int4") == ""  # the engine decides
    assert "not the folder's own DiT" in logic.tier_problem("turbo", None)
    for kind in ("int8", "bf16", "fp8", "nvfp4"):
        assert f"not this DiT file ({kind})" in logic.tier_problem("turbo", kind)
    assert logic.tier_problem("turbo", "unknown") == ""  # a file the pack cannot read goes to the engine


def test_file_and_model_stamps(tmp_path):
    bundle = make_bundle(str(tmp_path / "b"))
    dit = tmp_path / "dit.safetensors"
    dit.write_bytes(b"one")
    m = MossModel(bundle, dit_file=str(dit))
    first = m.stamp()
    assert first == m.stamp()
    assert logic.file_stamp(str(dit))[2] == 3
    dit.write_bytes(b"three")  # replaced: another size and mtime
    assert m.stamp() != first
    second = m.stamp()
    with open(os.path.join(bundle, "transformer", "diffusion_pytorch_model.safetensors"), "wb") as f:
        f.write(b"w")  # a weight file added to the folder
    assert m.stamp() != second
    assert logic.file_stamp(str(tmp_path / "missing")) is None


def test_find_bundles_is_cached_until_a_folder_changes(tmp_path, monkeypatch):
    root = tmp_path / "moss_image"
    make_bundle(str(root / "a"))
    walks = []
    real_walk = os.walk
    monkeypatch.setattr(logic.os, "walk", lambda *a, **kw: (walks.append(a[0]), real_walk(*a, **kw))[1])
    assert list(logic.find_bundles([str(root)])) == ["a"]
    assert list(logic.find_bundles([str(root)])) == ["a"]
    assert len(walks) == 1  # the second call only checked mtimes
    make_bundle(str(root / "nested" / "b"))
    assert list(logic.find_bundles([str(root)])) == ["a", "nested/b"]
    os.remove(root / "a" / "vae" / "config.json")  # a bundle broken inside one of its parts
    assert list(logic.find_bundles([str(root)])) == ["nested/b"]


# the model config ---------------------------------------------------------------------------

def test_model_load_params_and_loras():
    m = MossModel("/m/sft", dit_file="/d.safetensors", tier="high")
    assert m.load_params(ALL, "1.1.0", "/cache") == {
        "model": "/m/sft", "mode": "balanced", "dit_file": "/d.safetensors", "tier": "high",
        "attention": "default", "cache_dir": "/cache"}
    chained = m.with_lora("/a.safetensors", 0.5).with_lora("/b.safetensors", 1)
    assert chained.loras == (("/a.safetensors", 0.5), ("/b.safetensors", 1.0))
    assert m.loras == ()  # immutable: the first node's output is unchanged
    assert chained.load_params(ALL, "x") == m.load_params(ALL, "x")  # LoRAs are per generation


def test_model_load_params_gating():
    with pytest.raises(MossInferError, match="'model_files' capability; this one \\(0.1\\)"):
        MossModel("/m", vae_file="/v").load_params({"tiers": True}, "0.1")
    with pytest.raises(MossInferError, match="'tiers'"):
        MossModel("/m", tier="high").load_params({}, "0.1")
    assert MossModel("/m").load_params({}, "0.1") == {"model": "/m", "mode": "balanced"}


def test_control_chain():
    c = MossControl("/adapter", 0.8).with_image(ControlImage("depth", None, 1.0, 0.0, 0.5))
    c = c.with_image(ControlImage("color", None, 2.0))
    assert c.to_param(["/d.png", "/c.png"]) == {
        "adapter": "/adapter", "lora_strength": 0.8,
        "images": [{"kind": "depth", "path": "/d.png", "strength": 1.0, "start": 0.0, "end": 0.5},
                   {"kind": "color", "path": "/c.png", "strength": 2.0, "start": 0.0, "end": 1.0}]}
    with pytest.raises(ValueError, match="given twice"):
        c.with_image(ControlImage("depth", None))


# generate params ----------------------------------------------------------------------------

def test_generate_params_full_build():
    p = logic.build_generate_params(values(sampler="heun", scheduler="karras", step_cache="fbcache",
                                           step_cache_threshold=0.03, hires_width=512, hires_height=512),
                                    ALL, "1.1.0", output="/o.png", preview_output="/p.png",
                                    init_image="/i.png", mask="/m.png", loras=[("/l.safetensors", 0.7)],
                                    control={"adapter": "/a"})
    assert p["output"] == "/o.png" and p["save_metadata"] is False
    assert p["sampler"] == "heun" and p["scheduler"] == "karras" and p["step_cache"] == "fbcache"
    assert p["step_cache_threshold"] == 0.03
    assert p["hires"] == {"width": 512, "height": 512, "strength": 0.45, "steps": 0}
    assert p["init_image"] == "/i.png" and p["denoise_strength"] == 0.75 and p["resize"] == "crop" and p["mask"] == "/m.png"
    assert p["loras"] == [{"path": "/l.safetensors", "strength": 0.7}]
    assert p["control"] == {"adapter": "/a"}
    assert p["preview_output"] == "/p.png" and p["preview_every"] == 1


def test_generate_params_old_server_omits_gated_defaults():
    p = logic.build_generate_params(values(), OLD, "0.2.0", output="/o.png", preview_output="/p.png")
    for name in logic.GATED_DEFAULTS:
        assert name not in p
    for name in ("init_image", "denoise_strength", "resize", "hires", "loras", "control", "step_cache_threshold"):
        assert name not in p
    assert set(logic.BASE_PARAMS) <= set(p)


@pytest.mark.parametrize("change,cap", [
    ({"sampler": "heun"}, "samplers"),
    ({"scheduler": "karras"}, "samplers"),
    ({"eta": 0.5}, "samplers"),
    ({"step_cache": "fbcache"}, "accel"),
    ({"cfg_truncation": 0.5}, "accel"),
    ({"step_cache_threshold": 0.05}, "accel"),
    ({"hires_width": 512, "hires_height": 512}, "hires"),
])
def test_generate_params_refuses_what_the_server_lacks(change, cap):
    with pytest.raises(MossInferError) as e:
        logic.build_generate_params(values(**change), OLD, "0.2.0", output="/o.png")
    assert e.value.code == "not_supported" and f"'{cap}'" in str(e.value) and "0.2.0" in str(e.value)


def test_generate_params_refuses_files_the_server_lacks():
    for kw, cap in (({"init_image": "/i"}, "image_input"), ({"loras": [("/l", 1.0)]}, "lora"),
                    ({"control": {"adapter": "/a"}}, "control")):
        with pytest.raises(MossInferError, match=f"'{cap}'"):
            logic.build_generate_params(values(), {}, "0.1", output="/o.png", **kw)


def test_generate_params_hires_needs_both_sides():
    with pytest.raises(MossInferError, match="both hires_width and hires_height"):
        logic.build_generate_params(values(hires_width=512), ALL, "1", output="/o.png")


def test_generate_params_preview_only_with_capability():
    p = logic.build_generate_params(values(width=1024, height=1024), ALL, "1", output="/o", preview_output="/p")
    assert p["preview_every"] == 5
    p = logic.build_generate_params(values(), {}, "1", output="/o", preview_output="/p")
    assert "preview_output" not in p and "preview_every" not in p


def test_after_run_options():
    assert logic.after_run_options({}) == ["keep", "unload"]
    assert logic.after_run_options({"release": True}) == ["keep", "release", "unload"]


def test_old_server_accepts_what_is_built_for_it(tmp_path, fake_server, fake_env):
    """The fake rejects params its capabilities lack, as an old moss-server would."""
    env = dict(fake_env, FAKE_MOSS_CAPS="preview,model_files,lora,tiers", FAKE_MOSS_VERSION="0.2.0")
    m = ServerManager(log_dir=str(tmp_path / "l"), env=env)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    try:
        with m.session(fake_server) as s:
            s.ensure_loaded(MossModel(str(bundle)).load_params(s.capabilities, s.version))
            p = logic.build_generate_params(values(), s.capabilities, s.version, output=str(tmp_path / "o.png"),
                                            preview_output=str(tmp_path / "p.png"))
            assert s.run_job("generate", p)["status"] == "ok"
            # sending a gated param anyway is what the gating prevents
            with pytest.raises(MossInferError, match='unknown parameter "scheduler"'):
                s.run_job("generate", dict(p, scheduler="moss"))
    finally:
        m.shutdown()
    assert any(meth == "generate" for meth, _ in recorded(env))


# progress -------------------------------------------------------------------------------------

def test_progress_steps():
    m = ProgressMapper(False, 4)
    assert m.total == 4
    assert m.update({"event": "progress", "phase": "text_encoder", "step": 0, "total_steps": 4}) is None
    assert m.update({"event": "progress", "phase": "denoise", "step": 2, "total_steps": 3}) == (2, 3)
    assert m.preview_position({"event": "preview", "step": 3, "total_steps": 3}) == (3, 3)
    # a fraction from a server is ignored without the capability
    assert m.update({"phase": "decode", "step": 0, "total_steps": 3, "fraction": 0.9}) is None


def test_progress_fraction():
    m = ProgressMapper(True, 4)
    assert m.total == logic.FRACTION_UNITS
    assert m.update({"phase": "text_encoder", "step": 0, "total_steps": 4, "fraction": 0.0}) == (0, 1000)
    assert m.update({"phase": "denoise", "step": 2, "total_steps": 4, "fraction": 0.5}) == (500, 1000)
    assert m.preview_position({"step": 3, "total_steps": 4}) == (500, 1000)
    assert m.update({"phase": "decode", "fraction": 1.2}) == (1000, 1000)


def test_progress_text():
    assert logic.progress_text({"phase": "dit_load", "step": 0, "total_steps": 4}) == "loading DiT"
    assert logic.progress_text({"phase": "denoise", "step": 2, "total_steps": 4, "eta_seconds": 12.4}) == "denoise 2/4 · ETA 12 s"
    assert logic.progress_text({"phase": "denoise", "step": 2, "total_steps": 4, "eta_seconds": None}) == "denoise 2/4"

"""Node logic without ComfyUI: folders, request building, progress, inspect results."""

import os
from dataclasses import replace

import pytest

from helpers import make_bundle, recorded
from owoinfer_comfy import logic
from owoinfer_comfy.logic import ControlImage, MossControl, MossModel, MossReferences, ProgressMapper
from owoinfer_comfy.server import MossInferError, ServerManager

ALL = {c: True for c in ("preview", "model_files", "lora", "tiers", "samplers", "image_input", "hires",
                         "distill", "control", "accel", "release", "preload", "progress_fraction", "references")}
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
    # a bundle at the depth limit that is still downloading: its last config lands in an existing folder
    deep = root / "x" / "y" / "z"
    make_bundle(str(deep))
    os.remove(deep / "vae" / "config.json")
    assert "x/y/z" not in logic.find_bundles([str(root)])
    (deep / "vae" / "config.json").write_text("{}")
    assert "x/y/z" in logic.find_bundles([str(root)])


# the model config ---------------------------------------------------------------------------

def test_model_load_params_and_loras():
    m = MossModel("/m/sft", dit_file="/d.safetensors", tier="high")
    assert m.load_params(ALL, "2.0.0", "/cache") == {
        "model": "/m/sft", "mode": "balanced", "dit_file": "/d.safetensors", "tier": "high",
        "attention": "default", "cache_dir": "/cache"}
    chained = m.with_lora("/a.safetensors", 0.5).with_lora("/b.safetensors", 1)
    assert chained.loras == (("/a.safetensors", 0.5), ("/b.safetensors", 1.0))
    assert m.loras == ()  # immutable: the first node's output is unchanged
    assert chained.load_params(ALL, "x") == m.load_params(ALL, "x")  # LoRAs are per generation


def test_model_load_params_gating():
    with pytest.raises(MossInferError, match="'model_files' capability; this one \\(0.1\\)"):
        MossModel("/m", vae_file="/v").load_params({"tiers": True}, "0.1")
    with pytest.raises(MossInferError, match="'tiers'"):  # the tier is required; a server without it can't take one
        MossModel("/m").load_params({}, "0.1")
    assert MossModel("/m").load_params({"tiers": True}, "2.0.0") == {
        "model": "/m", "mode": "balanced", "tier": "high", "attention": "default"}


def test_control_chain():
    c = MossControl("/adapter", 0.8).with_image(ControlImage("depth", None, 1.0, 0.0, 0.5))
    c = c.with_image(ControlImage("color", None, 2.0))
    assert c.to_param(["/d.png", "/c.png"]) == {
        "adapter": "/adapter", "lora_strength": 0.8,
        "images": [{"kind": "depth", "path": "/d.png", "strength": 1.0, "start": 0.0, "end": 0.5},
                   {"kind": "color", "path": "/c.png", "strength": 2.0, "start": 0.0, "end": 1.0}]}
    with pytest.raises(ValueError, match="given twice"):
        c.with_image(ControlImage("depth", None))


def test_references_chain():
    empty = MossReferences(256)
    first = empty.with_image("source")
    r = first.with_image("other")
    assert r.images == ("source", "other") and r.max_edge == 256
    # A node's output is not changed by the next node: one output can feed several chains.
    assert empty.images == () and first.images == ("source",)
    assert r.to_param(["/s.png", "/o.png"]) == {"images": ["/s.png", "/o.png"], "max_edge": 256}
    assert MossReferences().max_edge == 512
    four = r.with_image("c").with_image("d")
    with pytest.raises(ValueError, match="at most 4"):
        four.with_image("e")


# generate params ----------------------------------------------------------------------------

def test_generate_params_full_build():
    p = logic.build_generate_params(values(sampler="heun", scheduler="karras", step_cache="fbcache",
                                           step_cache_threshold=0.03, hires_width=512, hires_height=512),
                                    ALL, "2.0.0", output="/o.png", preview_output="/p.png",
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


def test_generate_params_references():
    refs = {"images": ["/s.png"], "max_edge": 512}
    p = logic.build_generate_params(values(), ALL, "2.1.1", output="/o.png", references=refs)
    assert p["references"] == refs and p["lora_placement"] == "default"
    assert "references" not in logic.build_generate_params(values(), ALL, "2.1.1", output="/o.png")
    with pytest.raises(MossInferError) as e:
        logic.build_generate_params(values(), OLD, "2.0.0", output="/o.png", references=refs)
    assert e.value.code == "not_supported" and "'references'" in str(e.value) and "2.0.0" in str(e.value)


def test_generate_params_lora_placement():
    assert logic.build_generate_params(values(lora_placement="merge"), ALL, "2.1.1", output="/o.png")["lora_placement"] == "merge"
    assert "lora_placement" not in logic.build_generate_params(values(), OLD, "2.0.0", output="/o.png")
    with pytest.raises(MossInferError) as e:
        logic.build_generate_params(values(lora_placement="merge"), OLD, "2.0.0", output="/o.png")
    assert e.value.code == "not_supported" and "lora_placement 'merge'" in str(e.value) and "'references'" in str(e.value)


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


def test_references_reach_the_server(tmp_path, fake_server, fake_env):
    from PIL import Image
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 16)).save(source)
    m = ServerManager(log_dir=str(tmp_path / "l"), env=fake_env)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    try:
        with m.session(fake_server) as s:
            assert s.has("references")
            s.ensure_loaded(MossModel(str(bundle)).load_params(s.capabilities, s.version))
            refs = MossReferences(256).with_image(None).to_param([str(source)])
            p = logic.build_generate_params(values(lora_placement="side_branch"), s.capabilities, s.version,
                                            output=str(tmp_path / "o.png"), references=refs)
            assert s.run_job("generate", p)["status"] == "ok"
            with pytest.raises(MossInferError, match="not with slg_scale"):  # the server's rule, not the pack's
                s.run_job("generate", dict(p, slg_scale=1.0))
    finally:
        m.shutdown()
    sent = [p for meth, p in recorded(fake_env) if meth == "generate"][0]
    assert sent["references"] == refs and sent["lora_placement"] == "side_branch"
    assert [p for meth, p in recorded(fake_env) if meth == "references_seen"][0]["sizes"] == [[32, 16]]


def test_gguf_conversion_progress(tmp_path, fake_server, fake_env):
    """The fake sends the conversion's progress on the first load only, before `plan`."""
    env = dict(fake_env, FAKE_MOSS_CAPS="preview,model_files,lora,tiers,samplers,progress_fraction")
    m = ServerManager(log_dir=str(tmp_path / "l"), cache_dir=str(tmp_path / "cache"), env=env)
    bundle, gguf = tmp_path / "bundle", tmp_path / "q8.gguf"
    bundle.mkdir()
    gguf.write_bytes(b"GGUF")
    label = "converting GGUF DiT (first load)"
    try:
        with m.session(fake_server) as s:
            s.ensure_loaded(MossModel(str(bundle), dit_file=str(gguf)).load_params(s.capabilities, s.version, m.cache_dir))
            p = logic.build_generate_params(values(), s.capabilities, s.version, output=str(tmp_path / "o.png"))
            runs = [[], []]
            for run in runs:
                def on_event(e, run=run):
                    run.append(logic.progress_text(e) if e["event"] == "progress" else e["event"])
                assert s.run_job("generate", p, on_event=on_event)["status"] == "ok"
    finally:
        m.shutdown()
    assert runs[0][:3] == [label, label, "plan"]
    assert "loading DiT" in " ".join(runs[0][3:]) and label not in runs[0][3:]
    assert label not in runs[1]


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


def test_progress_text_gguf_conversion():
    label = "converting GGUF DiT (first load)"
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": 0, "eta_seconds": None}) == label
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": 0.0, "eta_seconds": None}) == label
    # a library without progress fields: null, not 0
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": None, "eta_seconds": None}) == "loading DiT"
    # an ordinary DiT load
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": 0.4, "eta_seconds": None}) == "loading DiT"
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": 0.4, "eta_seconds": 3.0}) == "loading DiT · ETA 3 s"
    assert logic.progress_text({"phase": "dit_load", "step": 0, "fraction": 0, "eta_seconds": 3.0}) == "loading DiT · ETA 3 s"
    # another phase at the job's start
    assert logic.progress_text({"phase": "text_encoder", "step": 0, "fraction": 0, "eta_seconds": None}) == "text encoder"


# inspect results and hello values -----------------------------------------------------

def test_inspected_problems():
    from helpers import CONTROL, INT4_DIT, INT8_DIT, LORA
    from owoinfer_comfy.server import MossInferError as E
    m = MossModel("/m", dit_file="/d/int8.safetensors", text_encoder_file="/t/te.safetensors",
                  vae_file="/v/vae.safetensors", tier="reference").with_lora("/l/a.safetensors", 1)
    good = {"/d/int8.safetensors": INT8_DIT, "/t/te.safetensors": {"kind": "text_encoder", "loadable": True},
            "/v/vae.safetensors": {"kind": "vae", "loadable": True}, "/l/a.safetensors": LORA,
            "/c/control.safetensors": CONTROL}
    # a tier not in `tiers`: the server's reason
    assert logic.inspected_problems(m, None, good) == [
        "dit_file int8.safetensors: " + INT8_DIT["tier_reasons"]["reference"]]
    assert logic.inspected_problems(replace(m, tier="high"), None, good) == []
    int4 = dict(good, **{"/d/int8.safetensors": INT4_DIT})
    assert logic.inspected_problems(replace(m, tier="high"), None, int4) == [
        "dit_file int8.safetensors: " + INT4_DIT["tier_reasons"]["high"]]
    assert logic.inspected_problems(replace(m, tier="turbo"), None, int4) == []
    # wrong kinds in a slot
    wrong = dict(good, **{"/d/int8.safetensors": LORA, "/v/vae.safetensors": {"kind": "text_encoder", "loadable": True}})
    assert logic.inspected_problems(replace(m, tier="high"), None, wrong) == [
        "dit_file int8.safetensors is a LoRA, not a DiT", "vae_file vae.safetensors is a text encoder, not a VAE"]
    # LoRAs: not loadable, or not a LoRA
    bad = dict(good, **{"/l/a.safetensors": dict(LORA, loadable=False, note="a LoRA for SDXL")})
    assert logic.inspected_problems(replace(m, tier="high"), None, bad) == ["LoRA a.safetensors can't be used: a LoRA for SDXL"]
    unknown = dict(good, **{"/l/a.safetensors": {"kind": "unknown", "loadable": False, "note": "a GGUF file"}})
    assert logic.inspected_problems(replace(m, tier="high"), None, unknown) == [
        "LoRA a.safetensors is not a file the engine can use (a GGUF file), not a LoRA"]
    # errors from inspect itself
    assert logic.inspected_problems(replace(m, tier="high"), None,
                                    dict(good, **{"/t/te.safetensors": E("io", "/t/te.safetensors: no such file")})) == [
        "text_encoder_file te.safetensors: /t/te.safetensors: no such file"]
    # control kinds
    control = MossControl("/c/control.safetensors").with_image(ControlImage("depth", None)).with_image(ControlImage("normal", None))
    assert logic.inspected_problems(replace(m, tier="high"), control, good) == [
        "control kind 'normal': control.safetensors has the kinds color, depth"]
    assert logic.inspected_problems(replace(m, tier="high"), MossControl("/l/a.safetensors"), good) == [
        "control adapter a.safetensors is a LoRA, not a control adapter"]
    assert logic.inspect_paths(m, control) == ["/d/int8.safetensors", "/t/te.safetensors", "/v/vae.safetensors",
                                               "/l/a.safetensors", "/c/control.safetensors"]


def test_value_problems():
    values = {"sampler": ["euler", "ab2"], "attention": ["default", "exact"], "mode": ["balanced"]}
    load = {"model": "/m", "mode": "balanced", "tier": "high", "attention": "q8_fp8"}
    gen = {"sampler": "heun", "scheduler": "karras", "prompt": "p"}
    assert logic.value_problems(values, load, gen, "1.0.9") == [
        "attention 'q8_fp8' is not supported by this moss-server (1.0.9); it accepts: default, exact",
        "sampler 'heun' is not supported by this moss-server (1.0.9); it accepts: euler, ab2"]
    assert logic.value_problems({}, load, gen, "1.0.0") == []  # no `values`: nothing to check
    assert logic.value_problems({"lora_placement": ["default"]}, load, dict(gen, lora_placement="merge"), "2.1.1") == [
        "lora_placement 'merge' is not supported by this moss-server (2.1.1); it accepts: default"]
    assert logic.value_problems({"lora_placement": logic.LORA_PLACEMENTS}, load, dict(gen, lora_placement="merge"), "2.1.1") == []

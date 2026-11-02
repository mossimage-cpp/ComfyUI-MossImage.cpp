"""GPU tests of the server manager with the real moss-server (no ComfyUI).

Opt-in with MOSSIMAGE_CPP_GPU_TESTS=1, under the GPU lock; 256x256, 2 to 4 steps.
"""

import os
import shutil
import signal

import numpy as np
import pytest
from PIL import Image

from helpers import DIT_INT8, EDIT_ADAPTER, GGUF_Q8, GPU_SERVER, SFT, TE_INT8, gpu_skip_reason
from owoinfer_comfy import logic
from owoinfer_comfy.imaging import load_preview
from owoinfer_comfy.logic import MossModel, MossReferences
from owoinfer_comfy.server import ServerCrashed, ServerManager

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    reason = gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    tmp = tmp_path_factory.mktemp("gpu-server")
    m = ServerManager(log_dir=str(tmp / "logs"), cache_dir=str(tmp / "cache"))
    m.tmp = tmp
    yield m
    m.shutdown()


def params(tmp, **kw):
    v = {"prompt": "a lighthouse on a cliff at sunset", "negative_prompt": "", "width": 256, "height": 256,
         "steps": 3, "cfg": 3.0, "shift": 5.0, "seed": 11, "uncond_skip_layers": "22", "slg_layers": "",
         "slg_scale": 0.0, "slg_start": 0.0, "slg_stop": 1.0, "max_sequence_length": 4096}
    v.update(kw)
    return v


def model():
    return MossModel(SFT, dit_file=DIT_INT8, text_encoder_file=TE_INT8)


def test_configs_only_folder_runs(manager):
    """A folder of configs and tokenizer only, with every component from a file, makes the same image as the full folder."""
    folder = manager.tmp / "configs-only" / "MOSS-Image-SFT"
    shutil.copytree(os.path.join(SFT, "tokenizer"), folder / "tokenizer")
    for c in ("transformer", "text_encoder", "vae"):
        (folder / c).mkdir(parents=True)
        shutil.copy(os.path.join(SFT, c, "config.json"), folder / c / "config.json")
    shutil.copy(os.path.join(SFT, "model_index.json"), folder / "model_index.json")
    vae = os.path.join(os.path.dirname(DIT_INT8), "flux2-vae.safetensors")
    images = []
    for i, root in enumerate((str(folder), SFT)):
        m = MossModel(root, dit_file=DIT_INT8, text_encoder_file=TE_INT8, vae_file=vae)
        with manager.session(GPU_SERVER) as s:
            s.ensure_loaded(m.load_params(s.capabilities, s.version, manager.cache_dir))
            out = manager.tmp / f"c{i}.png"
            p = logic.build_generate_params(params(manager.tmp, steps=2), s.capabilities, s.version, output=str(out))
            done = s.run_job("generate", p)
            s.unload()
        assert done["status"] == "ok", done
        with Image.open(out) as img:
            images.append(img.tobytes())
    assert images[0] == images[1]


def test_previews_and_progress(manager):
    events = []
    with manager.session(GPU_SERVER) as s:
        s.ensure_loaded(model().load_params(s.capabilities, s.version, manager.cache_dir))
        p = logic.build_generate_params(params(manager.tmp), s.capabilities, s.version,
                                        output=str(manager.tmp / "p.png"), preview_output=str(manager.tmp / "prev.png"))
        sizes = []

        def on_event(e):
            events.append(e)
            if e["event"] == "preview":
                sizes.append(load_preview(e["image"]).size)

        done = s.run_job("generate", p, on_event=on_event)
    assert done["status"] == "ok", done
    assert sizes == [(256, 256)] * 3
    phases = [e["phase"] for e in events if e["event"] == "progress"]
    assert phases[0] == "text_encoder" and phases[-1] == "decode" and phases.count("denoise") == 4
    mapper = logic.ProgressMapper(s.has("progress_fraction"), 3)
    positions = [x for x in (mapper.update(e) for e in events if e["event"] == "progress") if x]
    if s.has("progress_fraction"):  # the whole job's share: never decreasing, below 1 until done
        values = [v for v, _ in positions]
        assert values == sorted(values) and values[-1] < mapper.total and values[-1] > mapper.total // 2
        assert all(e["eta_seconds"] is None or e["eta_seconds"] >= 0 for e in events if e["event"] == "progress")
    else:
        assert positions[-1][0] == mapper.total


def test_killed_server_is_restarted(manager):
    """Crash isolation: a server that dies mid-run fails the job; the next one starts a new process."""
    with manager.session(GPU_SERVER) as s:
        s.ensure_loaded(model().load_params(s.capabilities, s.version, manager.cache_dir))
        pid = manager.running.proc.pid
        p = logic.build_generate_params(params(manager.tmp, steps=200), s.capabilities, s.version,
                                        output=str(manager.tmp / "k.png"))

        def on_event(e):
            if e["event"] == "progress" and e["phase"] == "denoise" and e["step"] == 2:
                os.kill(pid, signal.SIGKILL)

        with pytest.raises(ServerCrashed):
            s.run_job("generate", p, on_event=on_event)
    assert manager.running is None
    with manager.session(GPU_SERVER) as s:
        assert manager.running.proc.pid != pid
        s.ensure_loaded(model().load_params(s.capabilities, s.version, manager.cache_dir))
        p = logic.build_generate_params(params(manager.tmp, steps=2), s.capabilities, s.version,
                                        output=str(manager.tmp / "k2.png"))
        assert s.run_job("generate", p)["status"] == "ok"


def test_release_then_generate(manager):
    with manager.session(GPU_SERVER) as s:
        s.ensure_loaded(model().load_params(s.capabilities, s.version, manager.cache_dir))
        p = logic.build_generate_params(params(manager.tmp, steps=2), s.capabilities, s.version,
                                        output=str(manager.tmp / "r.png"))
        assert s.run_job("generate", p)["status"] == "ok"
        r = s.release(keep_host_copy=True)
        assert r["cancelled"] is False and r["vram_released_bytes"] > 0
        assert s.ensure_loaded(model().load_params(s.capabilities, s.version, manager.cache_dir)) is False
        done = s.run_job("generate", dict(p, output=str(manager.tmp / "r2.png")))
    assert done["status"] == "ok"
    if r["dit_host_copy"]:
        assert done["report"]["dit_load"]["source"] == "host_copy"


def test_edit_adapter_with_references(manager):
    if not os.path.isfile(EDIT_ADAPTER):
        pytest.skip(f"model file not on this machine: {EDIT_ADAPTER}")
    paths = [str(manager.tmp / "ref0.png"), str(manager.tmp / "ref1.png")]
    Image.effect_noise((128, 96), 64).convert("RGB").save(paths[0])
    Image.linear_gradient("L").resize((96, 128)).convert("RGB").save(paths[1])
    refs = MossReferences(128).with_image(None).with_image(None).to_param(paths)
    m = model().with_lora(EDIT_ADAPTER, 1.0)
    out = manager.tmp / "edit.png"
    with manager.session(GPU_SERVER) as s:
        if not s.has("references"):
            pytest.skip(f"moss-server {s.version} is older than 2.1: no reference images")
        s.ensure_loaded(m.load_params(s.capabilities, s.version, manager.cache_dir))
        p = logic.build_generate_params(params(manager.tmp, steps=2, lora_placement="side_branch"), s.capabilities, s.version,
                                        output=str(out), loras=m.loras, references=refs)
        done = s.run_job("generate", p)
        s.unload()
    assert done["status"] == "ok", done
    report = done["report"]["references"]
    assert report["count"] == 2 and report["max_edge"] == 128, report
    assert not done["warnings"], done["warnings"]
    with Image.open(out) as img:
        assert img.size == (256, 256)


def test_gguf_dit(manager):
    """A GGUF DiT: its first load converts into <cache>/gguf/ (5.4 GB), later loads find the file."""
    cache = os.environ.get("MOSS_TEST_CACHE", "")
    if not cache:
        pytest.skip("converting a GGUF writes 5.4 GB and /tmp may be tmpfs: set MOSS_TEST_CACHE to a folder with room")
    if not os.path.isfile(GGUF_Q8):
        pytest.skip(f"model file not on this machine: {GGUF_Q8}")
    folder = os.path.join(cache, "gguf")
    fresh = not (os.path.isdir(folder) and os.listdir(folder))
    m = MossModel(SFT, dit_file=GGUF_Q8, text_encoder_file=TE_INT8)
    label = "converting GGUF DiT (first load)"
    texts: list[list[str]] = []
    with manager.session(GPU_SERVER) as s:
        if not s.has("references"):
            pytest.skip(f"moss-server {s.version} is older than 2.1: no GGUF DiTs")
        for i in range(2):
            s.ensure_loaded(m.load_params(s.capabilities, s.version, cache))
            run: list[str] = []
            texts.append(run)
            out = manager.tmp / f"gguf{i}.png"
            p = logic.build_generate_params(params(manager.tmp, steps=2), s.capabilities, s.version, output=str(out))

            def on_event(e, run=run):
                run.append(logic.progress_text(e) if e["event"] == "progress" else e["event"])

            done = s.run_job("generate", p, on_event=on_event)
            assert done["status"] == "ok", done
            s.unload()
            with Image.open(out) as img:
                assert np.asarray(img.convert("RGB"), dtype=np.float32).std() > 0.05 * 255, "a flat image"
    assert os.listdir(folder)
    if fresh:
        assert label in texts[0] and texts[0].index(label) < texts[0].index("plan")
    assert label not in texts[1]

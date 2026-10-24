#!/usr/bin/env python3
"""A fake moss-server for tests: protocol 1 over stdin/stdout, no GPU.

Modelled on MossStudio's tools/fake-moss-server.mjs, extended to what the node
pack uses: capabilities from the environment, previews, progress fractions,
preload/release, and failure modes chosen by the prompt.

Environment:
  FAKE_MOSS_CAPS        comma-separated capabilities that are true (default: a 1.0.0 build)
  FAKE_MOSS_VERSION     the version in hello (default 1.0.0)
  FAKE_MOSS_PROTOCOL    the protocol_version in hello (default 1)
  FAKE_MOSS_ABI         the abi_version in hello (default 1)
  FAKE_MOSS_STEP_DELAY  seconds per phase or step (default 0.02)
  FAKE_MOSS_RECORD      a file that gets every request line, and "argv ..." at start
  FAKE_MOSS_RELEASE_DELAY  seconds a release takes (default 0.02)
  FAKE_MOSS_VALUES      "1": hello carries `values` (the name tables)
  FAKE_MOSS_VALUES_DROP names left out of them, e.g. "sampler:heun,attention:q8_fp8"
  FAKE_MOSS_INSPECT_MAX inspects that run at once before `busy` (default 2)
  FAKE_MOSS_INSPECT_DELAY seconds an inspect takes (default 0.02; ten times that for a
                        file whose name contains "slow")

`inspect` (with the capability) answers from the file's safetensors metadata
"fake_inspect" (a JSON object merged into the result); any other file is "unknown".

Prompts with special words:
  CRASH          the process exits with status 134 during denoise
  CUDA_ERROR     done with status error, code cuda
  IGNORE_CANCEL  cancel is acknowledged but the job runs to the end
  HANG_ON_CANCEL after a cancel the server hangs: no events, no answers, ever
  SLOW_CANCEL    after a cancel the job sends nothing for FAKE_MOSS_SLOW_CANCEL seconds
                 (default 2), while requests are still answered, then ends cancelled
  BAD_LINES      malformed protocol lines before the first step
  WARN           done carries two warnings
"""

import json
import os
import struct
import sys
import threading
import time
import zlib

DEFAULT_CAPS = "preview,model_files,lora,tiers,samplers,image_input,hires,distill,system_cuda,control,accel"
ALL_CAPS = [
    "preview", "comfyui_models", "model_files", "lora", "tiers", "samplers", "image_input", "hires",
    "distill", "system_cuda", "control", "accel", "preload", "release", "certificate_dirs",
    "te_host_pin", "progress_fraction", "inspect",
]
CAPS = {c: False for c in ALL_CAPS}
for _c in os.environ.get("FAKE_MOSS_CAPS", DEFAULT_CAPS).split(","):
    if _c.strip():
        CAPS[_c.strip()] = True
VERSION = os.environ.get("FAKE_MOSS_VERSION", "1.0.0")
PROTOCOL = int(os.environ.get("FAKE_MOSS_PROTOCOL", "1"))
ABI = int(os.environ.get("FAKE_MOSS_ABI", "1"))
SLOW_CANCEL = float(os.environ.get("FAKE_MOSS_SLOW_CANCEL", "2"))
INSPECT_MAX = int(os.environ.get("FAKE_MOSS_INSPECT_MAX", "2"))
INSPECT_DELAY = float(os.environ.get("FAKE_MOSS_INSPECT_DELAY", "0.02"))
VALUES = {
    "mode": ["min_memory", "balanced", "max_speed"], "block_path": ["default", "op_by_op", "fused", "fused_graphs"],
    "tier": ["auto", "reference", "high", "turbo"],
    "attention": ["default", "exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc"],
    "dit_direct_io": ["auto", "buffered", "direct"], "te_host_pin": ["auto", "on", "off"],
    "sampler": ["euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep", "dataspace"],
    "scheduler": ["moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"],
    "noise_layout": ["official", "batch_stream"], "noise_source": ["philox", "philox_device", "brownian_tree"],
    "resize": ["crop", "fit", "stretch"], "mask_encoding": ["u8", "binary01"], "step_cache": ["off", "fbcache"],
}
for _drop in filter(None, os.environ.get("FAKE_MOSS_VALUES_DROP", "").split(",")):
    _k, _v = _drop.split(":")
    VALUES[_k].remove(_v)
inspects_running = 0
DELAY = float(os.environ.get("FAKE_MOSS_STEP_DELAY", "0.02"))
RECORD = os.environ.get("FAKE_MOSS_RECORD", "")
RELEASE_DELAY = float(os.environ.get("FAKE_MOSS_RELEASE_DELAY", "0.02"))

LOAD_PARAMS = {
    None: {"model", "mode", "device", "vram_budget_bytes", "vram_reserve_bytes", "host_ram_budget_bytes",
           "cache_dir", "embedding_cache", "direct_io", "block_path"},
    "model_files": {"dit_file", "text_encoder_file", "vae_file"},
    "tiers": {"tier", "attention"},
    "lora": {"lora_strict"},
    "distill": {"time_embedding_file", "distill"},
    "certificate_dirs": {"certificate_dirs"},
    "te_host_pin": {"te_host_pin"},
}
GEN_PARAMS = {
    None: {"output", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed",
           "uncond_skip_layers", "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length",
           "sampler", "save_metadata", "include_prompts"},
    "samplers": {"scheduler", "noise_layout", "noise_source", "flow_multiplier", "eta", "s_noise", "dataspace"},
    "lora": {"loras"},
    "preview": {"preview_output", "preview_every"},
    "image_input": {"init_image", "mask", "mask_encoding", "denoise_strength", "resize"},
    "hires": {"hires"},
    "distill": {"distill"},
    "control": {"control"},
    "accel": {"cfg_truncation", "step_cache", "step_cache_threshold"},
}
MODES = {"min_memory", "balanced", "max_speed"}
TIERS = {"auto", "reference", "high", "turbo"}
SAMPLERS = {"euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep", "dataspace"}
SCHEDULERS = {"moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"}

out_lock = threading.Lock()
state_lock = threading.Lock()
engine = None  # the load params
job = None  # {"id", "kind", "cancel": Event, "step", "total"}


def send(msg):
    with out_lock:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def record(line):
    if RECORD:
        with open(RECORD, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")


class ParamError(Exception):
    pass


def allowed(table):
    names = set()
    for cap, members in table.items():
        if cap is None or CAPS.get(cap):
            names |= members
    return names


def check_known(params, table):
    for name in params:
        if name not in allowed(table):
            raise ParamError(f'unknown parameter "{name}"')


def png_bytes(width, height, rgb):
    raw = bytearray()
    row = bytes(rgb) * width
    for _ in range(height):
        raw += b"\x00" + row

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")


def write_png(path, width, height, rgb):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(png_bytes(width, height, rgb))
    os.replace(tmp, path)


def fake_inspect(path):
    """The file's "fake_inspect" metadata as an inspect result; None when the file is missing."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    result = {"path": path, "kind": "unknown", "loadable": False, "file_bytes": size, "tensors": 0, "note": ""}
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        result["tensors"] = len([k for k in header if k != "__metadata__"])
        meta = (header.get("__metadata__") or {}).get("fake_inspect")
        if meta:
            result.update(json.loads(meta))
        else:
            result["note"] = f"{path} is not a MOSS-Image file (fake)"
    except (ValueError, struct.error, OSError):
        result["note"] = f"{path} is not a safetensors file"
    return result


def validate_generate(p, need_output=True):
    check_known(p, GEN_PARAMS)
    if need_output and not p.get("output"):
        raise ParamError('"output" is required')
    if need_output and not isinstance(p.get("prompt"), str):
        raise ParamError('"prompt" is required')
    for k in ("width", "height"):
        v = p.get(k, 1024)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ParamError(f'"{k}" must be an integer in [16, 4096]')
        if v <= 0 or v % 16:
            raise ParamError("Height and width must be positive multiples of 16.")
    if os.environ.get("FAKE_MOSS_VALUES") == "1" and p.get("sampler", "euler") not in VALUES["sampler"]:
        raise ParamError('"sampler" must be one of ' + ", ".join(VALUES["sampler"]))
    if p.get("sampler", "euler") not in SAMPLERS:
        raise ParamError('"sampler" must be one of ' + ", ".join(sorted(SAMPLERS)))
    if p.get("scheduler", "moss") not in SCHEDULERS:
        raise ParamError('"scheduler" must be one of ' + ", ".join(sorted(SCHEDULERS)))
    steps = p.get("steps", 50)
    if not isinstance(steps, int) or steps < 1:
        raise ParamError('"steps" must be an integer >= 1')
    if "mask" in p and "init_image" not in p:
        raise ParamError("mask needs init_image")
    for k in ("init_image", "mask"):
        if k in p and not os.path.isfile(p[k]):
            raise ParamError(f"{k}: file not found: {p[k]}")
    if "denoise_strength" in p:
        d = p["denoise_strength"]
        if not 0 <= d <= 1:
            raise ParamError('"denoise_strength" must be in [0, 1]')
        if d < 1 and "init_image" not in p:
            raise ParamError("denoise_strength below 1 needs init_image")
    for lora in p.get("loras", []):
        if set(lora) - {"path", "strength"} or not os.path.isfile(lora.get("path", "")):
            raise ParamError(f"loras: bad entry or missing file: {lora}")
    c = p.get("control")
    if c is not None:
        if not os.path.isfile(c.get("adapter", "")):
            raise ParamError(f"control.adapter: file not found: {c.get('adapter')}")
        for i, img in enumerate(c.get("images", [])):
            if not os.path.isfile(img.get("path", "")):
                raise ParamError(f"control.images[{i}].path: file not found: {img.get('path')}")
    if "preview_every" in p and "preview_output" not in p:
        raise ParamError("preview_every needs preview_output")


def progress(job_id, phase, step, total, elapsed, fraction):
    ev = {"event": "progress", "id": job_id, "phase": phase, "step": step, "total_steps": total,
          "step_seconds": DELAY if phase == "denoise" and step else 0, "elapsed_seconds": elapsed}
    if CAPS["progress_fraction"]:
        ev["fraction"] = fraction
        ev["eta_seconds"] = None if fraction == 0 else round(elapsed / fraction - elapsed, 3)
    send(ev)


def finish(j, msg):
    """The server is idle again by the time `done` is written (server-protocol.md)."""
    global job
    with state_lock:
        if job is j:
            job = None
    send(msg)


def run_generate(j, p):
    prompt = p.get("prompt", "")
    total = p.get("steps", 50)
    if "init_image" in p:
        total = max(1, round(total * p.get("denoise_strength", 0.75)))
    j["total"] = total
    t0 = time.monotonic()
    cancel = j["cancel"]
    honour = "IGNORE_CANCEL" not in prompt

    def elapsed():
        return round(time.monotonic() - t0, 4)

    def cancelled():
        if "HANG_ON_CANCEL" in prompt and cancel.is_set():
            while True:
                time.sleep(1)
        if "SLOW_CANCEL" in prompt and cancel.is_set():
            time.sleep(SLOW_CANCEL)  # e.g. a DiT load that ends before the cancel is seen
        return honour and cancel.is_set()

    def finish_cancelled(steps_run):
        finish(j, {"event": "done", "id": j["id"], "status": "cancelled", "steps_run": steps_run,
              "report": {"cancelled": True, "steps_run": steps_run}, "certified": True, "warnings": []})

    send({"event": "plan", "id": j["id"], "explanation": "fake plan", "plan": {"mode": engine.get("mode", "balanced")}})
    units = total + 3
    for i, phase in enumerate(("text_encoder", "dit_load")):
        j["phase"] = phase
        progress(j["id"], phase, 0, total, elapsed(), i / units)
        time.sleep(DELAY)
        if cancelled():
            return finish_cancelled(0)
    j["phase"] = "denoise"
    if "BAD_LINES" in prompt:
        with out_lock:
            sys.stdout.write('not json\n[1, 2]\n{"id": "x", "error": "a string"}\n{"event": "progress"}\n')
            sys.stdout.flush()
    progress(j["id"], "denoise", 0, total, elapsed(), 2 / units)
    for step in range(1, total + 1):
        time.sleep(DELAY)
        if "CRASH" in prompt and step == max(1, total // 2):
            sys.stderr.write("fake: simulated fault\n")
            sys.stderr.flush()
            os._exit(134)
        if cancelled():
            return finish_cancelled(step - 1)
        j["step"] = step
        if "preview_output" in p and (step % p.get("preview_every", 1) == 0 or step == total):
            write_png(p["preview_output"], p.get("width", 1024), p.get("height", 1024), (step * 10 % 256, 0, 0))
            send({"event": "preview", "id": j["id"], "step": step, "total_steps": total,
                  "image": p["preview_output"], "width": p.get("width", 1024), "height": p.get("height", 1024)})
        progress(j["id"], "denoise", step, total, elapsed(), (2 + step) / units)
    j["phase"] = "decode"
    progress(j["id"], "decode", 0, total, elapsed(), (2 + total) / units)
    time.sleep(DELAY)
    if "CUDA_ERROR" in prompt:
        finish(j, {"event": "done", "id": j["id"], "status": "error",
              "error": {"code": "cuda", "message": "CUDA error: an illegal memory access was encountered"}, "warnings": []})
        return
    seed = p.get("seed", 42)
    seed = 7 if seed in (-1, None) else seed
    rgb = (seed % 256, (seed // 256) % 256, 200 if "init_image" in p else 100)
    write_png(p["output"], p.get("width", 1024), p.get("height", 1024), rgb)
    warnings = ["LoRA layers.0.proj.q: requantized", "prompt truncated to 1 of 2 tokens"] if "WARN" in prompt else []
    report = {"request": p, "seed_used": seed, "steps_run": total, "cancelled": False, "image": p["output"],
              "quality": {"tier": engine.get("tier", "auto")}, "models": {"dit": {"certified": not p.get("loras")}}}
    finish(j, {"event": "done", "id": j["id"], "status": "ok", "image": p["output"], "report": report,
          "certified": not p.get("loras") and "control" not in p, "warnings": warnings})


def run_preload(j, p):
    send({"event": "plan", "id": j["id"], "explanation": "fake plan", "plan": {}})
    for phase in ("dit_load", "decode"):
        progress(j["id"], phase, 0, 0, 0.0, 0.0)
        time.sleep(DELAY)
        if j["cancel"].is_set():
            finish(j, {"event": "done", "id": j["id"], "status": "cancelled", "steps_run": 0, "report": {},
                  "certified": True, "warnings": []})
            return
    finish(j, {"event": "done", "id": j["id"], "status": "ok", "report": {}, "preload_note": "",
          "dit_load": {"source": "files", "io": "direct", "bytes": 1, "seconds": 0.1, "note": ""},
          "certified": True, "warnings": []})


def worker(j, fn, p):
    global job
    try:
        fn(j, p)
    finally:
        with state_lock:
            if job is j:
                job = None


def handle(msg):
    global engine, job
    req_id = msg.get("id")
    method = msg.get("method")
    p = msg.get("params", {})

    def ok(result):
        send({"id": req_id, "result": result})

    def err(code, message):
        send({"id": req_id, "error": {"code": code, "message": message}})

    try:
        if method == "hello":
            if p.get("protocol_version", PROTOCOL) != PROTOCOL:
                return err("unsupported_protocol", "unsupported protocol version")
            hello = {"protocol_version": PROTOCOL, "server": "moss-server", "version": VERSION, "abi_version": ABI,
                       "capabilities": {c: v for c, v in CAPS.items() if v or c in DEFAULT_CAPS.split(",")},
                       "probe": {"message": "fake probe", "device_count": 1, "supported_device_count": 1,
                                 "build_flavour": "cu13"}}
            if os.environ.get("FAKE_MOSS_VALUES") == "1":
                hello["values"] = VALUES
            return ok(hello)
        if method == "inspect":
            if not CAPS["inspect"]:
                return err("method_not_found", "unknown method: inspect")
            check_known(p, {None: {"path"}})
            path = p.get("path")
            if not isinstance(path, str) or not path:
                raise ParamError('"path" must be a non-empty string')
            if os.path.exists(path) and not os.path.isfile(path):
                raise ParamError(f"{path} is not a regular file")
            global inspects_running
            with state_lock:
                if inspects_running >= INSPECT_MAX:
                    return err("busy", f"{INSPECT_MAX} inspect requests are running; send it again after one answers")
                inspects_running += 1

            def run_inspect():
                global inspects_running
                time.sleep(INSPECT_DELAY * (10 if "slow" in os.path.basename(path) else 1))
                answer = fake_inspect(path)
                with state_lock:
                    inspects_running -= 1  # free before answering: the client may send the next one at once
                if answer is None:
                    return err("io", f"{path}: no such file")
                ok(answer)

            threading.Thread(target=run_inspect, daemon=True).start()
            return
        if method == "status":
            with state_lock:
                r = {"state": "unloaded" if engine is None else "idle"}
                if engine is not None:
                    r.update(model=engine["model"], mode=engine.get("mode", "balanced"))
                if job is not None:
                    r.update(state={"generate": "generating", "preload": "preloading", "release": "releasing"}[job["kind"]],
                             generating_id=job["id"], cancelling=job["cancel"].is_set(),
                             phase=job.get("phase", "text_encoder"), step=job["step"], total_steps=job["total"],
                             elapsed_seconds=0.1)
            return ok(r)
        if method == "cancel":
            with state_lock:
                if job is None:
                    return ok({"cancelling": False})
                if "id" in p and p["id"] != job["id"]:
                    return ok({"cancelling": False, "generating_id": job["id"]})
                job["cancel"].set()
                ok({"cancelling": True, "generating_id": job["id"]})
                hang = "HANG_ON_CANCEL" in job.get("prompt", "")
            while hang:  # the whole server stops: nothing reads stdin any more
                time.sleep(1)
            return
        if method == "shutdown":
            with state_lock:
                j = job
            if j is not None:
                j["cancel"].set()
                while True:
                    with state_lock:
                        if job is None:
                            break
                    time.sleep(0.01)
            ok({})
            sys.stdout.flush()
            os._exit(0)
        if method == "load":
            check_known(p, LOAD_PARAMS)
            if p.get("mode", "balanced") not in MODES:
                raise ParamError('"mode" must be one of balanced, max_speed, min_memory')
            if p.get("tier", "auto") not in TIERS:
                raise ParamError('"tier" must be one of auto, high, reference, turbo')
            with state_lock:
                if job is not None:
                    return err("busy", "a job is running")
            model = p.get("model", "")
            if os.path.isfile(model):
                return err("not_supported", f"{model}: a file as the model path is not supported")
            if not os.path.isdir(model):
                engine = None
                return err("invalid_argument", f"model: no such directory: {model}")
            for k in ("dit_file", "text_encoder_file", "vae_file"):
                if k in p and not os.path.isfile(p[k]):
                    engine = None
                    return err("invalid_argument", f"{k}: no such file: {p[k]}")
            engine = dict(p)
            send({"event": "log", "id": req_id, "level": "info", "message": f"engine created for {model} (fake)"})
            return ok({"model": model, "mode": p.get("mode", "balanced"), "device": p.get("device", 0)})
        if method == "unload":
            with state_lock:
                if job is not None:
                    return err("busy", "a job is running")
            had, engine = engine is not None, None
            return ok({"unloaded": had})
        if method == "release":
            if not CAPS["release"]:
                return err("method_not_found", "unknown method: release")
            if not isinstance(p.get("keep_host_copy", True), bool):
                raise ParamError('"keep_host_copy" must be a boolean')
            with state_lock:
                if job is not None:
                    return err("busy", "a job is running")
                if engine is None:
                    return err("no_engine", "no engine is loaded")
                j = {"id": req_id, "kind": "release", "cancel": threading.Event(), "step": 0, "total": 0}
                job = j
            keep = p.get("keep_host_copy", True)

            def run_release():
                global job
                cancelled = j["cancel"].wait(RELEASE_DELAY)
                with state_lock:
                    job = None
                if cancelled:
                    return ok({"cancelled": True, "seconds": 0.0, "vram_released_bytes": 0, "host_copy_bytes": 0,
                               "host_copy_seconds": 0.0, "dit_host_copy": False, "te_host_copy": False,
                               "note": "cancelled before it started"})
                ok({"cancelled": False, "seconds": RELEASE_DELAY, "vram_released_bytes": 1 << 30,
                    "host_copy_bytes": (1 << 29) if keep else 0, "host_copy_seconds": RELEASE_DELAY,
                    "dit_host_copy": keep, "te_host_copy": False, "note": ""})

            threading.Thread(target=run_release, daemon=True).start()
            return
        if method in ("generate", "preload"):
            if method == "preload" and not CAPS["preload"]:
                return err("method_not_found", "unknown method: preload")
            if method == "preload":
                for k in ("output", "preview_output", "preview_every", "save_metadata", "include_prompts"):
                    if k in p:
                        raise ParamError(f'unknown parameter "{k}"')
            validate_generate(p, need_output=(method == "generate"))
            with state_lock:
                if job is not None:
                    return err("busy", "a job is running")
                if engine is None:
                    return err("no_engine", "no engine is loaded")
                j = {"id": req_id, "kind": method, "cancel": threading.Event(), "step": 0, "total": p.get("steps", 50),
                     "prompt": p.get("prompt", "")}
                job = j
            ok({"accepted": True})
            fn = run_generate if method == "generate" else run_preload
            threading.Thread(target=worker, args=(j, fn, p), daemon=True).start()
            return
        return err("method_not_found", f"unknown method: {method}")
    except ParamError as e:
        return err("invalid_params", str(e))


def main():
    if "--version" in sys.argv:
        print(f"moss-server {VERSION} (protocol {PROTOCOL}, ABI 1)")
        return
    record("argv " + json.dumps(sys.argv[1:]))
    sys.stderr.write(f"fake moss-server {VERSION} starting\n")
    sys.stderr.flush()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        record(line)
        try:
            msg = json.loads(line)
        except ValueError as e:
            send({"id": None, "error": {"code": "parse_error", "message": str(e)}})
            continue
        handle(msg)
    os._exit(0)


if __name__ == "__main__":
    main()

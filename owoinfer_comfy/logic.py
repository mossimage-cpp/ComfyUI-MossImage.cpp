"""Node logic without ComfyUI: protocol values, model config, request building.

Values and ranges are those of MossInfer's docs/server-protocol.md (protocol 2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .server import MossInferError

FROM_FOLDER = "(from folder)"
NO_BUNDLE = "(no MOSS-Image folder found)"

MODES = ["balanced", "min_memory", "max_speed"]
TIERS = ["reference", "high", "turbo"]
ATTENTIONS = ["default", "exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc"]
# `dataspace` is left out: it needs a sampler parameters file (a distill bundle's), see README "Planned".
SAMPLERS = ["euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep"]
BASE_SAMPLERS = {"euler", "ab2"}  # the samplers after these need the `samplers` capability
SCHEDULERS = ["moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"]
NOISE_LAYOUTS = ["official", "batch_stream"]
NOISE_SOURCES = ["philox", "philox_device", "brownian_tree"]
RESIZE_MODES = ["crop", "fit", "stretch"]
STEP_CACHES = ["off", "fbcache"]
AFTER_RUN = ["keep", "release", "unload"]

MAX_SIDE = 4096
MAX_STEPS = 10000
MAX_SEED = 2**53 - 1  # the server's browser-safe range
MAX_SEQUENCE_LENGTH = 4096

# Generate params that only a server with the capability knows, with the protocol's
# default: at the default they are left out, so older builds still accept the request.
GATED_DEFAULTS: dict[str, tuple[str, Any]] = {
    "scheduler": ("samplers", "moss"),
    "noise_layout": ("samplers", "official"),
    "noise_source": ("samplers", "philox"),
    "flow_multiplier": ("samplers", 1.0),
    "eta": ("samplers", 1.0),
    "s_noise": ("samplers", 1.0),
    "cfg_truncation": ("accel", 1.0),
    "step_cache": ("accel", "off"),
}
BASE_PARAMS = ["prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed",
               "uncond_skip_layers", "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length"]


def missing_capability(what: str, capability: str, version: str) -> MossInferError:
    return MossInferError("not_supported",
                          f"{what} needs a moss-server with the '{capability}' capability; this one ({version}) does not have it")


# model folders -------------------------------------------------------------------------

def is_bundle_dir(path: str) -> bool:
    """A MOSS-Image folder: tokenizer/ plus the three component configs."""
    return (os.path.isdir(os.path.join(path, "tokenizer"))
            and all(os.path.isfile(os.path.join(path, c, "config.json")) for c in ("transformer", "text_encoder", "vae")))


def find_bundles(roots: Iterable[str], max_depth: int = 3) -> dict[str, str]:
    """Bundle folders under `roots`, named by their path relative to the root ("/" separated).

    A bundle's own subfolders are not searched; the first root wins on a name clash.
    The result is cached until a folder that the search looked at changes (its mtime),
    because ComfyUI asks for the node schemas often.
    """
    roots = list(roots)
    cached = _bundle_cache.get((tuple(roots), max_depth))
    if cached is not None and all(_mtime(d) == t for d, t in cached[1].items()):
        return dict(cached[0])
    found: dict[str, str] = {}
    seen: dict[str, int | None] = {}
    for root in roots:
        seen[root] = _mtime(root)
        if not os.path.isdir(root):
            continue
        if is_bundle_dir(root):  # a path that names one bundle directly
            found.setdefault(os.path.basename(os.path.normpath(root)), root)
            continue
        for dirpath, dirnames, _ in os.walk(root, followlinks=True):
            seen[dirpath] = _mtime(dirpath)
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth and is_bundle_dir(dirpath):
                found.setdefault(rel.replace(os.sep, "/"), dirpath)
                for c in BUNDLE_PARTS:  # a part removed or re-made changes these
                    seen[os.path.join(dirpath, c)] = _mtime(os.path.join(dirpath, c))
                dirnames[:] = []
                continue
            if depth >= max_depth:
                dirnames[:] = []
                for c in BUNDLE_PARTS:  # a bundle still downloading here completes in these
                    seen[os.path.join(dirpath, c)] = _mtime(os.path.join(dirpath, c))
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    found = dict(sorted(found.items()))
    _bundle_cache[(tuple(roots), max_depth)] = (found, seen)
    return dict(found)


BUNDLE_PARTS = ("tokenizer", "transformer", "text_encoder", "vae")
_bundle_cache: dict[tuple, tuple[dict[str, str], dict[str, int | None]]] = {}


def _mtime(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def file_stamp(path: str) -> tuple[str, int, int] | None:
    """(path, mtime in ns, size): changes when the file is replaced. None when missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return path, st.st_mtime_ns, st.st_size


def bundle_stamp(model_dir: str) -> list:
    """file_stamp() of every file in the bundle's tokenizer/ and component folders."""
    out = []
    for part in BUNDLE_PARTS:
        folder = os.path.join(model_dir, part)
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        out += [file_stamp(os.path.join(folder, n)) for n in names]
    return out


# the model config ----------------------------------------------------------------------

@dataclass(frozen=True)
class MossModel:
    """What MOSS-Image Model Loader hands on (ComfyUI type MOSS_MODEL)."""

    model_dir: str
    dit_file: str = ""
    text_encoder_file: str = ""
    vae_file: str = ""
    mode: str = "balanced"
    tier: str = "high"
    attention: str = "default"
    loras: tuple[tuple[str, float], ...] = ()

    def with_lora(self, path: str, strength: float) -> "MossModel":
        return replace(self, loras=self.loras + ((path, float(strength)),))

    def stamp(self) -> list:
        """The files' sizes and mtimes, so a replaced file makes the engine load again."""
        return bundle_stamp(self.model_dir) + [file_stamp(f) for f in
                                               (self.dit_file, self.text_encoder_file, self.vae_file) if f]

    def load_params(self, caps: dict[str, bool], version: str, cache_dir: str | None = None) -> dict[str, Any]:
        p: dict[str, Any] = {"model": self.model_dir, "mode": self.mode}
        files = {"dit_file": self.dit_file, "text_encoder_file": self.text_encoder_file, "vae_file": self.vae_file}
        if any(files.values()) and not caps.get("model_files"):
            raise missing_capability("Component model files", "model_files", version)
        p.update({k: v for k, v in files.items() if v})
        if not caps.get("tiers"):
            raise missing_capability("Choosing the tier", "tiers", version)
        p["tier"] = self.tier
        p["attention"] = self.attention
        if cache_dir:
            p["cache_dir"] = cache_dir
        return p


@dataclass
class ControlImage:
    kind: str
    image: Any = field(compare=False)  # a ComfyUI IMAGE tensor
    strength: float = 1.0
    start: float = 0.0
    end: float = 1.0


@dataclass
class MossControl:
    """What MOSS-Image Control hands on (ComfyUI type MOSS_CONTROL)."""

    adapter: str
    lora_strength: float = 1.0
    images: tuple[ControlImage, ...] = ()

    def with_image(self, image: ControlImage) -> "MossControl":
        if any(i.kind == image.kind for i in self.images):
            raise ValueError(f"control kind {image.kind!r} is given twice; the adapter takes one image per kind")
        return MossControl(self.adapter, self.lora_strength, self.images + (image,))

    def to_param(self, image_paths: list[str]) -> dict[str, Any]:
        return {"adapter": self.adapter, "lora_strength": self.lora_strength,
                "images": [{"kind": i.kind, "path": p, "strength": i.strength, "start": i.start, "end": i.end}
                           for i, p in zip(self.images, image_paths)]}


# generate --------------------------------------------------------------------------------

def preview_every_for(width: int, height: int) -> int:
    """Every step for small images, every 5th from 1 MP on (server-protocol.md, `preview`)."""
    return 1 if width * height < 1024 * 1024 else 5


def build_generate_params(values: dict[str, Any], caps: dict[str, bool], version: str, *,
                          output: str, preview_output: str | None = None,
                          init_image: str | None = None, mask: str | None = None,
                          loras: Iterable[tuple[str, float]] = (),
                          control: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `generate` params for the node's widget values and the files written for it.

    Params a server without the capability does not know are left out at their
    defaults, and refused with a clear message otherwise. The server validates the
    values itself; its `invalid_params` message is what the user sees.
    """
    p: dict[str, Any] = {"output": output}
    for name in BASE_PARAMS:
        p[name] = values[name]
    p["save_metadata"] = False  # the PNG is read back and deleted; ComfyUI saves its own metadata

    sampler = values.get("sampler", "euler")
    if sampler not in BASE_SAMPLERS and not caps.get("samplers"):
        raise missing_capability(f"The {sampler} sampler", "samplers", version)
    p["sampler"] = sampler

    for name, (cap, default) in GATED_DEFAULTS.items():
        value = values.get(name, default)
        if caps.get(cap):
            p[name] = value
        elif value != default:
            raise missing_capability(f"{name} {value!r}", cap, version)

    threshold = values.get("step_cache_threshold", 0.0)
    if threshold:  # 0 = the engine's default for the model
        if not caps.get("accel"):
            raise missing_capability("step_cache_threshold", "accel", version)
        p["step_cache_threshold"] = threshold

    if init_image is not None or mask is not None:
        if not caps.get("image_input"):
            raise missing_capability("An init image", "image_input", version)
        if init_image is not None:
            p["init_image"] = init_image
            p["denoise_strength"] = values.get("denoise_strength", 0.75)
            p["resize"] = values.get("resize", "crop")
        if mask is not None:
            p["mask"] = mask  # without init_image the server answers with its own message

    hw, hh = values.get("hires_width", 0), values.get("hires_height", 0)
    if hw or hh:
        if not caps.get("hires"):
            raise missing_capability("The hires fix", "hires", version)
        if not (hw and hh):
            raise MossInferError("invalid_params", "hires fix: set both hires_width and hires_height (0 and 0 = off)")
        p["hires"] = {"width": hw, "height": hh, "strength": values.get("hires_strength", 0.45),
                      "steps": values.get("hires_steps", 0)}

    loras = [{"path": path, "strength": strength} for path, strength in loras]
    if loras:
        if not caps.get("lora"):
            raise missing_capability("LoRAs", "lora", version)
        p["loras"] = loras

    if control is not None:
        if not caps.get("control"):
            raise missing_capability("A control adapter", "control", version)
        p["control"] = control

    if preview_output and caps.get("preview"):
        p["preview_output"] = preview_output
        p["preview_every"] = preview_every_for(values["width"], values["height"])
    return p


def after_run_options(caps: dict[str, bool]) -> list[str]:
    """`release` (soft unload, weights kept in RAM) only where the server has it."""
    return [o for o in AFTER_RUN if o != "release" or caps.get("release")]


# progress -----------------------------------------------------------------------------------

PHASE_LABELS = {"text_encoder": "text encoder", "dit_load": "loading DiT", "denoise": "denoise", "decode": "decode"}
FRACTION_UNITS = 1000


class ProgressMapper:
    """`progress` events to ProgressBar (value, total).

    With the `progress_fraction` capability the bar follows `fraction` (the whole
    job, never decreasing); otherwise `step / total_steps` during `denoise`.
    """

    def __init__(self, use_fraction: bool, steps: int):
        self.use_fraction = use_fraction
        self.value, self.total = 0, (FRACTION_UNITS if use_fraction else max(1, steps))

    def update(self, event: dict[str, Any]) -> tuple[int, int] | None:
        fraction = event.get("fraction")
        if self.use_fraction and isinstance(fraction, (int, float)):
            self.value, self.total = round(min(1.0, max(0.0, fraction)) * FRACTION_UNITS), FRACTION_UNITS
            return self.value, self.total
        if event.get("phase") == "denoise" and event.get("total_steps", 0) > 0 and not self.use_fraction:
            self.value, self.total = int(event["step"]), int(event["total_steps"])
            return self.value, self.total
        return None

    def preview_position(self, event: dict[str, Any]) -> tuple[int, int]:
        """Where to show a `preview` event's image: it comes right before its step's progress."""
        if not self.use_fraction and event.get("total_steps", 0) > 0:
            return int(event["step"]), int(event["total_steps"])
        return self.value, self.total


def progress_text(event: dict[str, Any]) -> str:
    phase = str(event.get("phase", ""))
    text = PHASE_LABELS.get(phase, phase)
    if phase == "denoise" and event.get("total_steps"):
        text += f" {event.get('step', 0)}/{event['total_steps']}"
    eta = event.get("eta_seconds")
    if isinstance(eta, (int, float)) and eta >= 0:
        text += f" · ETA {eta:.0f} s"
    return text


# what the server says about files (`inspect`) and names (hello's `values`) ----------------

KIND_NAMES = {"dit": "a DiT", "text_encoder": "a text encoder", "vae": "a VAE", "lora": "a LoRA",
              "control_adapter": "a control adapter", "unknown": "not a file the engine can use"}


def _what(result: dict[str, Any]) -> str:
    kind = str(result.get("kind", "unknown"))
    text = KIND_NAMES.get(kind, kind)
    note = str(result.get("note") or "")
    return f"{text} ({note})" if note else text


def inspected_problems(model: "MossModel", control: "MossControl | None",
                       results: dict[str, Any]) -> list[str]:
    """Why the files of a run can't work, from the server's `inspect` results ([] when they can).

    `results` maps each path to its result dict, or to the MossInferError its inspect got.
    The tier must be in the DiT's `tiers`, the engine's own decision for that DiT with the bundle's
    text encoder and the tier's default attention. The other combinations a tier refuses (e.g.
    `reference` with a quantized text encoder file, or with an 8-bit `attention`), and any tier
    with the folder's own DiT, are refused by the engine at the first `generate`, with its reason.
    """
    problems: list[str] = []

    def check(slot: str, path: str, kind: str) -> dict[str, Any] | None:
        r = results.get(path)
        name = os.path.basename(path)
        if isinstance(r, Exception):
            problems.append(f"{slot} {name}: {r}")
            return None
        if r is None:
            return None
        if r.get("kind") != kind:
            problems.append(f"{slot} {name} is {_what(r)}, not {KIND_NAMES[kind]}")
            return None
        if not r.get("loadable", False):
            problems.append(f"{slot} {name} can't be used: {r.get('note') or 'the engine refuses it'}")
            return None
        return r

    if model.dit_file:
        dit = check("dit_file", model.dit_file, "dit")
        if dit is not None and model.tier not in (dit.get("tiers") or []):
            reason = (dit.get("tier_reasons") or {}).get(model.tier) or f"tier {model.tier} does not run this DiT"
            problems.append(f"dit_file {os.path.basename(model.dit_file)}: {reason}")
    if model.text_encoder_file:
        check("text_encoder_file", model.text_encoder_file, "text_encoder")
    if model.vae_file:
        check("vae_file", model.vae_file, "vae")
    for path, _ in model.loras:
        check("LoRA", path, "lora")
    if control is not None:
        adapter = check("control adapter", control.adapter, "control_adapter")
        if adapter is not None:
            kinds = list(adapter.get("control_kinds") or [])
            for image in control.images:
                if image.kind not in kinds:
                    problems.append(f"control kind {image.kind!r}: {os.path.basename(control.adapter)} has the kinds "
                                    f"{', '.join(kinds) or '(none)'}")
    return problems


def inspect_paths(model: "MossModel", control: "MossControl | None") -> list[str]:
    paths = [p for p in (model.dit_file, model.text_encoder_file, model.vae_file) if p]
    paths += [p for p, _ in model.loras]
    if control is not None:
        paths.append(control.adapter)
    return list(dict.fromkeys(paths))


# load / generate params checked against hello's `values`, under the same names
VALUE_CHECKS = {"load": ("mode", "tier", "attention"),
                "generate": ("sampler", "scheduler", "resize", "step_cache", "noise_layout", "noise_source")}


def value_problems(values: dict[str, list[str]], load_params: dict[str, Any],
                   generate_params: dict[str, Any], version: str) -> list[str]:
    """Names this server does not accept, from hello's `values` ([] when it has no `values`).

    The node's lists are fixed when ComfyUI defines it; an older server may lack a name.
    """
    problems = []
    for params, names in ((load_params, VALUE_CHECKS["load"]), (generate_params, VALUE_CHECKS["generate"])):
        for name in names:
            accepted = values.get(name)
            if accepted is not None and name in params and params[name] not in accepted:
                problems.append(f"{name} {params[name]!r} is not supported by this moss-server ({version}); "
                                f"it accepts: {', '.join(accepted)}")
    return problems

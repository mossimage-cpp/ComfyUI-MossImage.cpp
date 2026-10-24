"""The ComfyUI nodes (V3 API). The logic lives in logic.py, server.py and imaging.py."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy.cli_args import LatentPreviewMethod, args
from comfy_api.latest import ComfyExtension, io

from . import logic
from .imaging import load_image_tensor, load_preview, save_image_png, save_mask_png
from .logic import FROM_FOLDER, ControlImage, MossControl, MossModel, ProgressMapper
from .server import MossInferError, ServerManager, locate_server

log = logging.getLogger("owoinfer")

PACK = "ComfyUI-MossImage-owoinfer"
CATEGORY = "MOSS-Image (owoinfer)"
MODEL_FOLDER = "moss_image"

MossModelType = io.Custom("MOSS_MODEL")
MossControlType = io.Custom("MOSS_CONTROL")


def register_model_folder() -> None:
    """models/moss_image/ for bundle folders; extra_model_paths.yaml can add more (`moss_image: <dir>`)."""
    folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER), is_default=True)


_manager: ServerManager | None = None


def get_manager() -> ServerManager:
    """The one moss-server of this ComfyUI process, started on the first run and shut down at exit."""
    global _manager
    if _manager is None:
        base = os.path.join(folder_paths.get_user_directory(), PACK)
        _manager = ServerManager(log_dir=base, cache_dir=os.path.join(base, "cache"))
    return _manager


def _bundles() -> dict[str, str]:
    return logic.find_bundles(folder_paths.get_folder_paths(MODEL_FOLDER))


def _file_choice(folder: str, name: str) -> str:
    return "" if name == FROM_FOLDER else folder_paths.get_full_path_or_raise(folder, name)


def _stamp_of(folder: str, name: str):
    path = None if name == FROM_FOLDER else folder_paths.get_full_path(folder, name)
    return logic.file_stamp(path) if path else None


def _progress_text(node_id: str | None, text: str) -> None:
    """Text on the node (ComfyUI's progress text), when a front end is connected."""
    if not node_id:
        return
    try:
        from server import PromptServer  # ComfyUI's
        PromptServer.instance.send_progress_text(text, node_id)
    except (ImportError, AttributeError):
        pass


class ModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        bundles = list(_bundles()) or [logic.NO_BUNDLE]
        return io.Schema(
            node_id="OwoInferMossImageModelLoader",
            display_name="MOSS-Image Model Loader",
            category=CATEGORY,
            description="Picks a MOSS-Image model for owoinfer (MossInfer). The engine runs in its own process "
                        "(moss-server), shared by every node; it loads the model on the first run.",
            inputs=[
                io.Combo.Input("model_folder", options=bundles,
                               tooltip=f"A MOSS-Image bundle folder (Base or SFT, from Hugging Face) under models/{MODEL_FOLDER}/. "
                                       "It supplies the tokenizer and configs, and the components not replaced below."),
                io.Combo.Input("dit_file", options=cls._dit_options(), default=FROM_FOLDER,
                               tooltip="A single-file DiT (ComfyUI / community INT8 ConvRot, INT4 ConvRot, bf16) "
                                       "from diffusion_models."),
                io.Combo.Input("text_encoder_file", options=[FROM_FOLDER] + folder_paths.get_filename_list("text_encoders"),
                               default=FROM_FOLDER,
                               tooltip="A single-file Qwen3.5-9B text encoder (INT8 ConvRot or W4A8) from text_encoders."),
                io.Combo.Input("vae_file", options=[FROM_FOLDER] + folder_paths.get_filename_list("vae"),
                               default=FROM_FOLDER, tooltip="A single-file FLUX.2 VAE from vae."),
                io.Combo.Input("mode", options=logic.MODES, default="balanced",
                               tooltip="Memory mode. balanced adapts to free VRAM; min_memory shares the GPU with "
                                       "other apps; max_speed uses the RAM it can."),
                io.Combo.Input("tier", options=logic.TIERS, default="auto",
                               tooltip="Quality tier. auto picks the fastest certified configuration. "
                                       "turbo (lossy, uncertified) is for INT4 DiT files only, and they need it."),
                io.Combo.Input("attention", options=logic.ATTENTIONS, default="default", advanced=True,
                               tooltip="Overrides the tier's attention: exact or an 8-bit mode."),
            ],
            outputs=[MossModelType.Output(display_name="moss_model")],
        )

    @staticmethod
    def _dit_options() -> list[str]:
        return [FROM_FOLDER] + folder_paths.get_filename_list("diffusion_models")

    @classmethod
    def validate_inputs(cls, dit_file=None, tier=None) -> bool | str:
        # ComfyUI skips its own checks for the inputs named here, so they are repeated.
        # A linked input is None here (its value comes at run time): execute() checks it then.
        if dit_file is None or tier is None:
            return True
        return cls._check(dit_file, tier) or True

    @classmethod
    def _check(cls, dit_file: str, tier: str) -> str:
        if dit_file not in cls._dit_options():
            return f"dit_file not found in diffusion_models: {dit_file}"
        if tier not in logic.TIERS:
            return f"tier must be one of {', '.join(logic.TIERS)}"
        dit = _file_choice("diffusion_models", dit_file)
        return logic.tier_problem(tier, logic.dit_quant_kind(dit) if dit else None,
                                  has_certificate=bool(dit) and os.path.isfile(dit + logic.CERT_SIDECAR))

    @classmethod
    def fingerprint_inputs(cls, model_folder, dit_file, text_encoder_file, vae_file, **kwargs):
        path = _bundles().get(model_folder)
        return (tuple(logic.bundle_stamp(path)) if path else None,
                _stamp_of("diffusion_models", dit_file), _stamp_of("text_encoders", text_encoder_file),
                _stamp_of("vae", vae_file))

    @classmethod
    def execute(cls, model_folder, dit_file, text_encoder_file, vae_file, mode, tier,
                attention="default") -> io.NodeOutput:
        problem = cls._check(dit_file, tier)  # again: linked inputs skipped validate_inputs
        if problem:
            raise ValueError(problem)
        bundles = _bundles()
        if model_folder not in bundles:
            raise ValueError(f"MOSS-Image folder {model_folder!r} not found in "
                             f"{', '.join(folder_paths.get_folder_paths(MODEL_FOLDER))}")
        return io.NodeOutput(MossModel(
            model_dir=bundles[model_folder], dit_file=_file_choice("diffusion_models", dit_file),
            text_encoder_file=_file_choice("text_encoders", text_encoder_file),
            vae_file=_file_choice("vae", vae_file), mode=mode, tier=tier, attention=attention))


class LoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OwoInferMossImageLoRA",
            display_name="MOSS-Image LoRA",
            category=CATEGORY,
            description="Adds a LoRA (kohya, PEFT / diffusers, LyCORIS LoHa / LoKr, full deltas) to a MOSS-Image model. "
                        "Chain several.",
            inputs=[
                MossModelType.Input("moss_model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[MossModelType.Output(display_name="moss_model")],
        )

    @classmethod
    def fingerprint_inputs(cls, lora_name, **kwargs):
        return _stamp_of("loras", lora_name)

    @classmethod
    def execute(cls, moss_model: MossModel, lora_name: str, strength: float) -> io.NodeOutput:
        if strength == 0:
            return io.NodeOutput(moss_model)
        return io.NodeOutput(moss_model.with_lora(folder_paths.get_full_path_or_raise("loras", lora_name), strength))


class Control(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OwoInferMossImageControl",
            display_name="MOSS-Image Control",
            category=CATEGORY,
            description="A control adapter (e.g. MOSS-Image color / depth control) with one conditioning image. "
                        "Chain one node per kind. Lossy: runs with it are uncertified.",
            inputs=[
                io.Combo.Input("adapter", options=folder_paths.get_filename_list("loras"),
                               tooltip="The control adapter file, from loras."),
                io.String.Input("kind", default="depth", tooltip="One of the adapter's kinds, e.g. depth or color."),
                io.Image.Input("image", tooltip="The conditioning image (the first of a batch); resized to the "
                                                "output size like an init image."),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of the steps where the control starts."),
                io.Float.Input("end", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of the steps where the control ends."),
                io.Float.Input("lora_strength", default=1.0, min=-10.0, max=10.0, step=0.01,
                               tooltip="Strength of the adapter's LoRA part; the same on every node of a chain."),
                MossControlType.Input("control", optional=True, tooltip="Another kind's control, for the same adapter."),
            ],
            outputs=[MossControlType.Output(display_name="control")],
        )

    @classmethod
    def fingerprint_inputs(cls, adapter, **kwargs):
        return _stamp_of("loras", adapter)

    @classmethod
    def execute(cls, adapter, kind, image, strength, start, end, lora_strength, control=None) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", adapter)
        if control is None:
            control = MossControl(path, lora_strength)
        elif control.adapter != path:
            raise ValueError("chained MOSS-Image Control nodes must use the same adapter file")
        elif control.lora_strength != lora_strength:
            raise ValueError(f"chained MOSS-Image Control nodes must use the same lora_strength "
                             f"({control.lora_strength} and {lora_strength}): the adapter has one LoRA part")
        if image.shape[0] > 1:
            log.warning("MOSS-Image Control: image is a batch of %d; using the first", image.shape[0])
        return io.NodeOutput(control.with_image(ControlImage(kind.strip(), image, strength, start, end)))


class Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OwoInferMossImageGenerate",
            display_name="MOSS-Image Generate",
            category=CATEGORY,
            description="Generates an image with owoinfer (moss-server). Optional init image and mask for img2img "
                        "and inpainting.",
            inputs=[
                MossModelType.Input("moss_model"),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("width", default=1024, min=16, max=logic.MAX_SIDE, step=16),
                io.Int.Input("height", default=1024, min=16, max=logic.MAX_SIDE, step=16),
                io.Int.Input("steps", default=50, min=1, max=logic.MAX_STEPS),
                io.Float.Input("cfg", default=3.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Int.Input("seed", default=42, min=0, max=logic.MAX_SEED, control_after_generate=True),
                io.Combo.Input("sampler", options=logic.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=logic.SCHEDULERS, default="moss",
                               tooltip="moss with euler is the official pipeline."),
                io.Float.Input("shift", default=5.0, min=0.01, max=100.0, step=0.01),
                io.Image.Input("init_image", optional=True, tooltip="img2img: the starting image (the first of a batch)."),
                io.Mask.Input("mask", optional=True,
                              tooltip="Inpainting: 1 repaints, 0 keeps (the first of a batch). The init image's size."),
                MossControlType.Input("control", optional=True),
                io.Float.Input("denoise_strength", default=0.75, min=0.0, max=1.0, step=0.01,
                               tooltip="With an init image: the share of the schedule run from it."),
                io.Combo.Input("resize", options=logic.RESIZE_MODES, default="crop",
                               tooltip="How the init image is brought to width x height."),
                io.String.Input("uncond_skip_layers", default="22", advanced=True,
                                tooltip='DiT blocks the negative branch skips, e.g. "22, 24"; empty for none.'),
                io.String.Input("slg_layers", default="", advanced=True, tooltip="Skip-layer guidance blocks; empty for none."),
                io.Float.Input("slg_scale", default=0.0, min=0.0, max=100.0, step=0.01, advanced=True),
                io.Float.Input("slg_start", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("slg_stop", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("max_sequence_length", default=4096, min=1, max=logic.MAX_SEQUENCE_LENGTH, advanced=True),
                io.Float.Input("flow_multiplier", default=1.0, min=0.01, max=100.0, step=0.01, advanced=True,
                               tooltip="Ignored by the moss scheduler."),
                io.Combo.Input("noise_layout", options=logic.NOISE_LAYOUTS, default="official", advanced=True),
                io.Combo.Input("noise_source", options=logic.NOISE_SOURCES, default="philox", advanced=True),
                io.Float.Input("eta", default=1.0, min=0.0, max=100.0, step=0.01, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, advanced=True),
                io.Int.Input("hires_width", default=0, min=0, max=logic.MAX_SIDE, step=16, advanced=True,
                             tooltip="Hires fix: second-pass width (0 = off)."),
                io.Int.Input("hires_height", default=0, min=0, max=logic.MAX_SIDE, step=16, advanced=True,
                             tooltip="Hires fix: second-pass height (0 = off)."),
                io.Float.Input("hires_strength", default=0.45, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("hires_steps", default=0, min=0, max=logic.MAX_STEPS, advanced=True, tooltip="0 = steps."),
                io.Float.Input("cfg_truncation", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Lossy accelerator: CFG only for this share of the steps (1 = off)."),
                io.Combo.Input("step_cache", options=logic.STEP_CACHES, default="off", advanced=True,
                               tooltip="Lossy accelerator: the First-Block Cache."),
                io.Float.Input("step_cache_threshold", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True,
                               tooltip="0 = the engine's default for the model."),
                io.Boolean.Input("preview", default=True, advanced=True,
                                 tooltip="Latent previews on the node while sampling, unless ComfyUI's preview "
                                         "method is none."),
                io.Combo.Input("after_run", options=logic.AFTER_RUN, default="keep", advanced=True,
                               tooltip="keep: the model stays in VRAM, which ComfyUI's memory manager cannot see. "
                                       "release (moss-server with the release capability): VRAM freed, weights kept "
                                       "in RAM for a fast reload. unload: everything freed."),
                io.Boolean.Input("free_comfy_vram", default=True, advanced=True,
                                 tooltip="Unload ComfyUI's own models first, so they and the engine do not both hold VRAM."),
            ],
            outputs=[io.Image.Output(display_name="image")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, moss_model: MossModel, init_image=None, mask=None, control: MossControl | None = None,
                **values: Any) -> io.NodeOutput:
        server_path = locate_server()
        if values.get("free_comfy_vram", True):
            mm.unload_all_models()
            mm.soft_empty_cache()
        tmp_dir = os.path.join(folder_paths.get_temp_directory(), PACK)
        os.makedirs(tmp_dir, exist_ok=True)
        tag = uuid.uuid4().hex[:12]
        files: list[str] = []
        node_id = cls.hidden.unique_id if cls.hidden is not None else None

        def tmp(name: str) -> str:
            files.append(os.path.join(tmp_dir, f"{tag}-{name}.png"))
            return files[-1]

        try:
            init_path = mask_path = None
            if init_image is not None:
                if init_image.shape[0] > 1:
                    log.warning("MOSS-Image Generate: init_image is a batch of %d; using the first", init_image.shape[0])
                init_path = tmp("init")
                save_image_png(init_image[0], init_path)
            if mask is not None:
                if mask.dim() == 3 and mask.shape[0] > 1:
                    log.warning("MOSS-Image Generate: mask is a batch of %d; using the first", mask.shape[0])
                mask_path = tmp("mask")
                save_mask_png(mask, mask_path)
            control_param = None
            if control is not None:
                paths = [tmp(f"control{i}") for i in range(len(control.images))]
                for image, path in zip(control.images, paths):
                    save_image_png(image.image, path)
                control_param = control.to_param(paths)
            previews = values.get("preview", True) and args.preview_method != LatentPreviewMethod.NoPreviews

            manager = get_manager()
            with manager.session(server_path) as s:
                caps = s.capabilities
                after_run = values.get("after_run", "keep")
                if after_run == "release" and not caps.get("release"):
                    raise logic.missing_capability("after_run release", "release", s.version)
                params = logic.build_generate_params(
                    values, caps, s.version, output=tmp("out"), preview_output=tmp("preview") if previews else None,
                    init_image=init_path, mask=mask_path, loras=moss_model.loras, control=control_param)
                adapters = [path for path, _ in moss_model.loras] + ([control.adapter] if control is not None else [])
                s.ensure_loaded(moss_model.load_params(caps, s.version, manager.cache_dir), moss_model.stamp(), adapters)
                try:
                    done = s.run_job("generate", params, on_event=cls._event_handler(caps, values["steps"], node_id),
                                     should_cancel=mm.processing_interrupted)
                except MossInferError as e:
                    if e.code == "cancel_timeout":
                        log.warning("MOSS-Image: %s", e)
                        mm.throw_exception_if_processing_interrupted()
                    raise
                finally:
                    cls._after_run(s, after_run)
            return io.NodeOutput(cls._result(done, params["output"], node_id))
        finally:
            for path in files:
                try:
                    os.remove(path)
                except OSError:
                    pass

    @classmethod
    def _event_handler(cls, caps: dict[str, bool], steps: int, node_id: str | None):
        mapper = ProgressMapper(bool(caps.get("progress_fraction")), steps)
        pbar = comfy.utils.ProgressBar(mapper.total)

        def on_event(event: dict[str, Any]) -> None:
            kind = event.get("event")
            if kind == "progress":
                position = mapper.update(event)
                if position is not None:
                    pbar.update_absolute(*position)
                _progress_text(node_id, logic.progress_text(event))
            elif kind == "preview":
                try:
                    image = load_preview(event["image"])
                except (OSError, KeyError):
                    return
                value, total = mapper.preview_position(event)
                pbar.update_absolute(value, total, ("JPEG", image, args.preview_size))
            elif kind == "plan":
                log.debug("MOSS-Image plan: %s", event.get("explanation", ""))

        return on_event

    @staticmethod
    def _after_run(session, after_run: str) -> None:
        if after_run == "keep" or get_manager().running is None:
            return
        try:
            if after_run == "release":
                # After an interrupted run the flag is usually still set: only a new interrupt cancels.
                was = mm.processing_interrupted()
                r = session.release(keep_host_copy=True,
                                    should_cancel=lambda: not was and mm.processing_interrupted())
                if r.get("cancelled"):
                    log.info("MOSS-Image: release cancelled; the engine keeps its VRAM")
                else:
                    log.info("MOSS-Image: released %.1f GiB of VRAM%s", r.get("vram_released_bytes", 0) / 2**30,
                             ", DiT kept in RAM" if r.get("dit_host_copy") else (f" ({r['note']})" if r.get("note") else ""))
            elif after_run == "unload":
                session.unload()
        except MossInferError as e:
            log.warning("MOSS-Image: after_run %s failed: %s", after_run, e)

    @staticmethod
    def _result(done: dict[str, Any], output: str, node_id: str | None):
        warnings = [str(w) for w in done.get("warnings") or []]
        for warning in warnings:
            log.warning("MOSS-Image: %s", warning)
        status = done.get("status")
        if status == "cancelled":
            mm.throw_exception_if_processing_interrupted()
            raise mm.InterruptProcessingException()
        if status != "ok":
            err = done.get("error") or {}
            raise MossInferError(str(err.get("code", "internal")), str(err.get("message", "generation failed")))
        report = done.get("report") or {}
        quality = report.get("quality") or {}
        summary = (f"seed {report.get('seed_used')}, tier {quality.get('tier', '?')}, "
                   f"{'certified' if done.get('certified') else 'uncertified'}")
        log.info("MOSS-Image: %s", summary)
        _progress_text(node_id, "\n".join([summary] + [f"warning: {w}" for w in warnings]))
        return load_image_tensor(done.get("image") or output)


class OwoInferExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ModelLoader, LoRA, Control, Generate]

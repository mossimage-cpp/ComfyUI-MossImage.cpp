"""ComfyUI-MossImage-owoinfer: MOSS-Image in ComfyUI through owoinfer (MossInfer)'s moss-server."""

from .owoinfer_comfy.nodes import OwoInferExtension, register_model_folder

register_model_folder()


async def comfy_entrypoint() -> OwoInferExtension:
    return OwoInferExtension()

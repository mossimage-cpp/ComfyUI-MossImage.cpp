"""ComfyUI-MossImage.cpp: MOSS-Image in ComfyUI through mossimage.cpp's moss-server."""

from .owoinfer_comfy.nodes import OwoInferExtension, register_model_folder

register_model_folder()


async def comfy_entrypoint() -> OwoInferExtension:
    return OwoInferExtension()

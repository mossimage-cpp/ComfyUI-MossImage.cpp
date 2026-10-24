"""ComfyUI tensors to PNG files for moss-server, and its PNGs back (torch, numpy, PIL)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _to_uint8(array: np.ndarray) -> np.ndarray:
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_image_png(image: torch.Tensor, path: str) -> tuple[int, int]:
    """Write one ComfyUI image ([H, W, C] or [1, H, W, C], floats in 0..1) as an RGB PNG.

    Returns (width, height). An alpha channel is dropped.
    """
    if image.dim() == 4:
        image = image[0]
    array = image.detach().to("cpu", torch.float32).numpy()
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an image tensor [H, W, 1|3|4], got {list(image.shape)}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    Image.fromarray(_to_uint8(array[:, :, :3])).save(path, compress_level=1)
    return array.shape[1], array.shape[0]


def save_mask_png(mask: torch.Tensor, path: str) -> tuple[int, int]:
    """Write a ComfyUI MASK ([H, W] or [1, H, W]; 1 = repaint) as a grayscale PNG.

    moss-server regenerates where the mask is white (>= 128) and keeps black.
    """
    if mask.dim() == 3:
        mask = mask[0]
    array = mask.detach().to("cpu", torch.float32).numpy()
    if array.ndim != 2:
        raise ValueError(f"expected a mask tensor [H, W], got {list(mask.shape)}")
    Image.fromarray(_to_uint8(array)).save(path, compress_level=1)
    return array.shape[1], array.shape[0]


def load_image_tensor(path: str) -> torch.Tensor:
    """A PNG as a ComfyUI IMAGE: [1, H, W, 3] float32 in 0..1."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())[None]


def load_preview(path: str) -> Image.Image:
    """The server's preview PNG, read and closed at once (the server renames the next one over it)."""
    with Image.open(path) as img:
        return img.convert("RGB")

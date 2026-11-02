"""Tensor <-> PNG conversion (needs torch, numpy and PIL, as ComfyUI has)."""

import os

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed in this Python; run the tests with ComfyUI's venv")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from owoinfer_comfy.imaging import load_image_tensor, load_preview, save_image_png, save_mask_png  # noqa: E402


def test_image_round_trip(tmp_path):
    image = torch.rand(1, 24, 40, 3)
    path = str(tmp_path / "i.png")
    assert save_image_png(image, path) == (40, 24)
    back = load_image_tensor(path)
    assert back.shape == (1, 24, 40, 3) and back.dtype == torch.float32
    assert torch.max(torch.abs(back - image)) <= 0.5 / 255 + 1e-6


def test_image_batch_saves_its_first_frame(tmp_path):
    """Reference and Control hand on the whole batch: the first frame is the one sent."""
    batch = torch.zeros(2, 4, 4, 3)
    batch[0, ..., 0] = 1.0  # red
    batch[1, ..., 2] = 1.0  # blue
    save_image_png(batch, str(tmp_path / "b.png"))
    with Image.open(tmp_path / "b.png") as img:
        assert img.getpixel((0, 0)) == (255, 0, 0)


def test_image_alpha_and_gray(tmp_path):
    rgba = torch.zeros(8, 8, 4)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 0.2
    save_image_png(rgba, str(tmp_path / "a.png"))
    with Image.open(tmp_path / "a.png") as img:
        assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 0, 0)
    save_image_png(torch.full((1, 4, 4, 1), 0.5), str(tmp_path / "g.png"))
    with Image.open(tmp_path / "g.png") as img:
        assert img.getpixel((1, 1)) == (128, 128, 128)
    with pytest.raises(ValueError):
        save_image_png(torch.zeros(4, 4), str(tmp_path / "bad.png"))


def test_mask(tmp_path):
    mask = torch.zeros(1, 6, 10)
    mask[0, :, 5:] = 1.0
    assert save_mask_png(mask, str(tmp_path / "m.png")) == (10, 6)
    with Image.open(tmp_path / "m.png") as img:
        assert img.mode == "L" and img.getpixel((0, 0)) == 0 and img.getpixel((9, 5)) == 255


def test_preview_is_read_in_full(tmp_path):
    """The server renames the next preview over the file: what was read must not change with it."""
    path = tmp_path / "p.png"
    Image.new("RGB", (4, 2), (1, 2, 3)).save(path)
    img = load_preview(str(path))
    Image.new("RGB", (8, 8), (9, 9, 9)).save(tmp_path / "next.png")
    os.replace(tmp_path / "next.png", path)
    assert img.size == (4, 2) and img.getpixel((0, 0)) == (1, 2, 3)


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc/self/fd (Linux) to count open files")
def test_preview_file_is_closed(tmp_path):
    """No handle stays open on the preview (on Windows one would make the server's rename fail)."""
    path = tmp_path / "p.png"
    Image.new("RGB", (4, 2), (1, 2, 3)).save(path)
    before = set(os.listdir("/proc/self/fd"))
    img = load_preview(str(path))
    after = set(os.listdir("/proc/self/fd"))
    opened = [os.readlink(f"/proc/self/fd/{fd}") for fd in after - before if os.path.exists(f"/proc/self/fd/{fd}")]
    assert str(path) not in opened and img.size == (4, 2)

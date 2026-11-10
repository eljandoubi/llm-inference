import pytest
import torch
from PIL import Image

from vlmserve.multimodal.preprocess import ImageTransform


def test_output_is_channels_first_float32() -> None:
    out = ImageTransform(height=12, width=16)(Image.new("RGB", (40, 30), "red"))
    assert out.shape == (3, 12, 16)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_default_normalization_maps_black_and_white_to_minus_one_and_one() -> None:
    transform = ImageTransform(4, 4)
    torch.testing.assert_close(
        transform(Image.new("RGB", (4, 4), "white")), torch.ones(3, 4, 4)
    )
    torch.testing.assert_close(
        transform(Image.new("RGB", (4, 4), "black")), -torch.ones(3, 4, 4)
    )


def test_mean_and_std_are_applied_per_channel() -> None:
    transform = ImageTransform(2, 2, mean=(0.0, 0.5, 1.0), std=(1.0, 0.5, 2.0))
    out = transform(Image.new("RGB", (2, 2), (255, 255, 255)))
    assert out[:, 0, 0].tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_channels_keep_their_order() -> None:
    out = ImageTransform(2, 2, mean=(0, 0, 0), std=(1, 1, 1))(
        Image.new("RGB", (2, 2), (255, 0, 51))
    )
    assert out[:, 0, 0].tolist() == pytest.approx([1.0, 0.0, 0.2])


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "LA"])
def test_any_pil_mode_is_accepted(mode: str) -> None:
    assert ImageTransform(4, 4)(Image.new(mode, (9, 7))).shape == (3, 4, 4)


def test_width_and_height_are_not_swapped() -> None:
    out = ImageTransform(height=5, width=9)(Image.new("RGB", (100, 100)))
    assert out.shape[1:] == (5, 9)

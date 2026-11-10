from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Resize and normalize an image into the [3, H, W] tensor a vision tower takes."""

    height: int
    width: int
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (self.width, self.height), Image.Resampling.BICUBIC
        )
        pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32))  # [H, W, 3]
        mean = torch.tensor(self.mean).view(1, 1, 3)
        std = torch.tensor(self.std).view(1, 1, 3)
        return ((pixels / 255 - mean) / std).permute(2, 0, 1).contiguous()

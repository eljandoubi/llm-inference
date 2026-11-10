"""What the Hugging Face tests share: conversations, the transformers reference and
checkpoints rewritten under other tensor layouts. The random checkpoints themselves
come from randomhf."""

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file

from randomhf.checkpoints import IMAGE_SIZE, PATCH, TinyLlava
from vlmserve.frontend import ImagePart, TextPart, Turn

# clip drops its CLS token and siglip has none, so both give one token per patch
TOKENS_PER_IMAGE = (IMAGE_SIZE // PATCH) ** 2


def picture(seed: int, size: tuple[int, int] = (40, 30)) -> Image.Image:
    """A noisy image that is not square, so resizing and cropping have to happen."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def rewrite_weights(
    source: Path, target: Path, rename: Callable[[str], str | None]
) -> Path:
    """Copy a checkpoint folder under other tensor names (None drops a tensor)."""
    shutil.copytree(source, target)
    file = target / "model.safetensors"
    tensors = {
        new: tensor.contiguous()
        for name, tensor in load_file(file).items()
        if (new := rename(name)) is not None
    }
    save_file(tensors, file, metadata={"format": "pt"})
    return target


def classic_layout(name: str) -> str:
    """The llava-hf repos: the vision tower is still wrapped in a vision_model."""
    if name.startswith("vision_tower."):
        return name.replace("vision_tower.", "vision_tower.vision_model.", 1)
    return name


def nested_layout(name: str) -> str:
    """Files of transformers 4.52 to 4.57: everything under model., the head apart."""
    name = classic_layout(name)
    if name.startswith("language_model.lm_head."):
        return name.removeprefix("language_model.")
    if name.startswith("language_model.model."):
        return "model.language_model." + name.removeprefix("language_model.model.")
    return "model." + name


LAYOUTS = {"classic": classic_layout, "nested": nested_layout}


def image(seed: int) -> ImagePart:
    return ImagePart(picture(seed), f"key-{seed}")


SINGLE = [Turn("user", [image(1), TextPart("what is this?")])]
MULTI = [
    Turn("system", [TextPart("be brief")]),
    Turn("user", [TextPart("look: "), image(1), TextPart("and "), image(2)]),
    Turn("assistant", [TextPart("they differ")]),
    Turn("user", [image(3), TextPart("and this?")]),
]
TEXT_ONLY = [Turn("user", [TextPart("hello there")])]


def reference(tiny: TinyLlava, turns: Sequence[Turn]) -> Any:
    """What transformers' own processor makes of the same conversation."""
    conversation = [
        {
            "role": turn.role,
            "content": [
                {"type": "image"}
                if isinstance(p, ImagePart)
                else {"type": "text", "text": p.text}
                for p in turn.parts
            ],
        }
        for turn in turns
    ]
    text = tiny.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    pictures = [p.image for t in turns for p in t.parts if isinstance(p, ImagePart)]
    return tiny.processor(images=pictures or None, text=text, return_tensors="pt")


def hf_continuation(
    tiny: TinyLlava, turns: Sequence[Turn], steps: int = 8
) -> list[int]:
    """The tokens transformers generates greedily after the conversation."""
    inputs = reference(tiny, turns)
    out = tiny.model.generate(**inputs, do_sample=False, max_new_tokens=steps)
    return list(out[0, inputs["input_ids"].shape[1] :].tolist())

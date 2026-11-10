"""The vision side of llava-style models: an image tower plus a projector into the
embedding space of the text model."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

from vlmserve.hf.frontend import ImageSupport
from vlmserve.model.checkpoint import Checkpoint

log = logging.getLogger(__name__)

SUPPORTED_TOWERS = ("clip_vision_model", "siglip_vision_model")
# only the pooled output of a tower uses these, llava reads the hidden states
_UNUSED_BY_LLAVA = ("post_layernorm.", "head.")


def tokens_per_image(config: LlavaConfig) -> int:
    """How many placeholder tokens one image takes in the prompt."""
    vision = config.vision_config
    side = vision.image_size // vision.patch_size
    has_cls = vision.model_type == "clip_vision_model"
    dropped = config.vision_feature_select_strategy == "default"
    return side * side + int(has_cls) - int(dropped)


def llava_images(processor: Any, config: LlavaConfig) -> ImageSupport:
    """Image handling for the frontend, from a llava processor and the model config."""
    token = processor.image_token
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if token_id != config.image_token_index:
        raise ValueError(
            f"the tokenizer has {token!r} as id {token_id} but the model config "
            f"says images are id {config.image_token_index}"
        )
    size = config.vision_config.image_size
    probe = processor.image_processor(
        images=[Image.new("RGB", (9, 7))], return_tensors="pt"
    )
    if tuple(probe["pixel_values"].shape[-2:]) != (size, size):
        raise ValueError(
            f"the image processor makes {tuple(probe['pixel_values'].shape[-2:])} "
            f"pixels but the vision tower takes {size}x{size}"
        )
    return ImageSupport(
        processor.image_processor, token, token_id, tokens_per_image(config), size
    )


class LlavaVision:
    """Runs the tower, picks its feature layer(s), projects: the engine's
    VisionEncoder for llava models."""

    def __init__(
        self,
        tower: nn.Module,
        projector: nn.Module,
        layers: Sequence[int],
        drop_first_token: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.tower = tower
        self.projector = projector
        self.layers = tuple(layers)
        self.drop_first_token = drop_first_token
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def encode(
        self, inputs: Sequence[Mapping[str, torch.Tensor]]
    ) -> list[torch.Tensor]:
        pixels = torch.stack([item["pixel_values"] for item in inputs])
        outputs = self.tower(
            pixel_values=pixels.to(self.device, self.dtype),
            output_hidden_states=True,
            return_dict=True,
        )
        chosen = [outputs.hidden_states[i] for i in self.layers]
        if self.drop_first_token:  # llava's "default" selection leaves out the CLS
            chosen = [states[:, 1:] for states in chosen]
        features = self.projector(torch.cat(chosen, dim=-1))
        # a copy each, so a cached embedding doesn't keep its whole batch alive
        return [one.clone() for one in features]


def load_llava_vision(
    config: LlavaConfig,
    checkpoint: Checkpoint,
    device: torch.device,
    dtype: torch.dtype,
) -> LlavaVision:
    kind = config.vision_config.model_type
    if kind not in SUPPORTED_TOWERS:
        raise ValueError(
            f"vision tower {kind!r} is not supported, "
            f"only {', '.join(SUPPORTED_TOWERS)}"
        )
    with torch.device(device):
        tower = AutoModel.from_config(
            config.vision_config, attn_implementation="sdpa", dtype=dtype
        )
        projector = LlavaMultiModalProjector(config).to(dtype)
    _fill(tower, checkpoint.submodule("vision_tower"), "vision tower", _UNUSED_BY_LLAVA)
    _fill(projector, checkpoint.submodule("multi_modal_projector"), "projector")
    for module in (tower, projector):
        module.eval().requires_grad_(False)

    layer = config.vision_feature_layer
    return LlavaVision(
        tower,
        projector,
        [layer] if isinstance(layer, int) else layer,
        drop_first_token=config.vision_feature_select_strategy == "default",
        device=device,
        dtype=dtype,
    )


def _fill(
    module: nn.Module,
    tensors: Iterable[tuple[str, torch.Tensor]],
    what: str,
    optional: tuple[str, ...] = (),
) -> None:
    """Copy checkpoint tensors into a module, whatever layout the file was saved in."""
    state = module.state_dict()  # shares storage with the parameters
    filled: set[str] = set()
    with torch.no_grad():
        for name, tensor in tensors:
            key = _find(name, state)
            if key is None:
                log.debug("%s: nothing for %s", what, name)
                continue
            if state[key].shape != tensor.shape:
                raise ValueError(
                    f"{what} {name}: checkpoint has {tuple(tensor.shape)}, "
                    f"the model expects {tuple(state[key].shape)}"
                )
            state[key].copy_(tensor)
            filled.add(key)
    missing = [k for k in state if k not in filled and not k.startswith(optional)]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(f"{what}: {len(missing)} weights were never loaded: {shown}")


def _find(name: str, state: Mapping[str, torch.Tensor]) -> str | None:
    """Older files wrap the tower in a "vision_model." that newer code leaves out."""
    for candidate in (
        name,
        name.removeprefix("vision_model."),
        f"vision_model.{name}",
    ):
        if candidate in state:
            return candidate
    return None

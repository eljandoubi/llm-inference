"""Turn a model directory (or a Hub id) into the pieces the engine runs on."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.utils import logging as transformers_logging

from vlmserve.hf.frontend import HFFrontend
from vlmserve.hf.hub import resolve_model_dir
from vlmserve.hf.llava import llava_images, load_llava_vision
from vlmserve.model.checkpoint import Checkpoint
from vlmserve.model.config import SUPPORTED_TEXT_MODELS, DecoderConfig
from vlmserve.model.decoder import Decoder
from vlmserve.model.weights import load_decoder_weights
from vlmserve.runner import VisionEncoder

log = logging.getLogger(__name__)

MULTIMODAL_MODELS = ("llava",)


@dataclass(frozen=True, slots=True)
class ModelSource:
    """A model directory and what its config says, read before the weights are."""

    directory: Path
    config: Any  # the transformers config: llava, or a plain text model
    decoder: DecoderConfig

    @property
    def multimodal(self) -> bool:
        return bool(self.config.model_type in MULTIMODAL_MODELS)

    @property
    def text_config(self) -> Any:
        return self.config.text_config if self.multimodal else self.config

    @property
    def context_length(self) -> int:
        """The longest sequence the model was trained for."""
        return self.decoder.max_position_embeddings


@dataclass(frozen=True, slots=True)
class LoadedModel:
    source: ModelSource
    decoder: Decoder
    vision: VisionEncoder | None
    frontend: HFFrontend


def open_model(ref: str, revision: str | None = None) -> ModelSource:
    """Find the model (downloading its files if it is a Hub id) and read its config."""
    directory = resolve_model_dir(ref, revision)
    config = AutoConfig.from_pretrained(directory)
    kind = config.model_type
    if kind in MULTIMODAL_MODELS:
        text = config.text_config
    elif kind in SUPPORTED_TEXT_MODELS:
        text = config
    else:
        supported = ", ".join((*MULTIMODAL_MODELS, *SUPPORTED_TEXT_MODELS))
        raise ValueError(f"model type {kind!r} is not supported, only {supported}")
    return ModelSource(directory, config, DecoderConfig.from_hf(text.to_dict()))


def load_model(
    source: ModelSource,
    device: torch.device,
    dtype: torch.dtype,
    max_model_len: int,
) -> LoadedModel:
    """Read the weights onto the device. The frontend comes first: a model without a
    usable tokenizer or processor should fail before gigabytes are read."""
    if max_model_len > source.context_length:
        raise ValueError(
            f"max_model_len is {max_model_len} but the model was trained for "
            f"{source.context_length} tokens"
        )
    directory, config = source.directory, source.config
    with _quiet():
        if source.multimodal:
            processor = AutoProcessor.from_pretrained(directory)
            tokenizer = processor.tokenizer
            frontend = HFFrontend(
                tokenizer,
                processor,
                eos_token_ids(directory, tokenizer, source.text_config),
                llava_images(processor, config),
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(directory)
            frontend = HFFrontend(
                tokenizer,
                tokenizer,
                eos_token_ids(directory, tokenizer, source.text_config),
            )

    checkpoint = Checkpoint(directory)
    decoder = Decoder(source.decoder, max_model_len, device, dtype)
    unused = load_decoder_weights(decoder, checkpoint.text_weights())
    if unused:
        log.debug("%d text tensors have no place in the decoder", len(unused))
    vision = None
    if source.multimodal:
        vision = load_llava_vision(config, checkpoint, device, dtype)
    return LoadedModel(source, decoder, vision, frontend)


def eos_token_ids(directory: Path, tokenizer: Any, text_config: Any) -> frozenset[int]:
    """Every id that ends an answer, as the model's files and tokenizer name them."""
    found = [getattr(text_config, "eos_token_id", None), tokenizer.eos_token_id]
    generation = directory / "generation_config.json"
    if generation.is_file():
        found.append(json.loads(generation.read_text()).get("eos_token_id"))
    ids: set[int] = set()
    for value in found:
        ids.update(value if isinstance(value, list) else [value])
    return frozenset(i for i in ids if isinstance(i, int))


@contextmanager
def _quiet() -> Iterator[None]:
    """transformers warns about its optional torchvision backend, which we don't use."""
    previous = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(previous)

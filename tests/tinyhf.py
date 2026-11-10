"""Tiny random llava checkpoints on disk, to test the Hugging Face integration offline.

They have the file layout, config keys, tokenizer and processor classes of the real
models (llava-1.5 is clip + llama, llava-interleave is siglip + qwen2) with a few
thousand parameters instead of billions. The transformers model that wrote them is
kept around as the reference the server is compared with.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import (
    CLIPImageProcessorPil,
    CLIPVisionConfig,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipImageProcessorPil,
    SiglipVisionConfig,
)

IMAGE_SIZE = 28
PATCH = 14
TOKENS_PER_IMAGE = (
    IMAGE_SIZE // PATCH
) ** 2  # clip drops its CLS token, siglip has none

PLAIN_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)

LLAMA_TEMPLATE = (
    "{% for m in messages %}"
    "{{ m['role'].upper() + ': ' }}"
    "{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}{{ '<image>\n' }}"
    "{% else %}{{ c['text'] + ' ' }}{% endif %}"
    "{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)

CHATML_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' }}"
    "{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}{{ '<image>\n' }}"
    "{% else %}{{ c['text'] }}{% endif %}"
    "{% endfor %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    text: str  # "llama" or "qwen2"
    vision: str  # "clip" or "siglip"

    @property
    def llama(self) -> bool:
        return self.text == "llama"


CLIP_LLAMA = Variant("clip-llama", "llama", "clip")  # llava-1.5
SIGLIP_QWEN = Variant("siglip-qwen", "qwen2", "siglip")  # llava-interleave
VARIANTS = (CLIP_LLAMA, SIGLIP_QWEN)


@dataclass(frozen=True, slots=True)
class TinyLlava:
    variant: Variant
    path: Path
    model: LlavaForConditionalGeneration  # transformers' own, the reference
    processor: LlavaProcessor


def picture(seed: int, size: tuple[int, int] = (40, 30)) -> Image.Image:
    """A noisy image that is not square, so resizing and cropping have to happen."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def build_tokenizer(variant: Variant) -> PreTrainedTokenizerFast:
    """One token per byte plus the specials, which is all a test needs."""
    specials = ["<unk>", "<image>"]
    specials += ["<s>", "</s>"] if variant.llama else ["<|im_start|>", "<|im_end|>"]
    vocab = {
        tok: i for i, tok in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    for special in specials:
        vocab[special] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    if variant.llama:  # like llama tokenizers, qwen ones add nothing in front
        tok.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", vocab["<s>"])]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>" if variant.llama else None,
        eos_token="</s>" if variant.llama else "<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=[s for s in specials if s != "<unk>"],
    )


def text_config(variant: Variant, tokenizer: PreTrainedTokenizerFast) -> Any:
    """The text model of a variant: llama with its own head, qwen2 with a tied one."""
    args = {
        "vocab_size": len(tokenizer),
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "initializer_range": 0.3,  # wide logits, so greedy decoding has no near ties
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    if variant.llama:
        return LlamaConfig(**args)
    return Qwen2Config(**args, tie_word_embeddings=True)


def build_tiny_llava(path: Path, variant: Variant, seed: int = 0) -> TinyLlava:
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(variant)
    clip = variant.vision == "clip"
    vision_args = {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH,
    }
    vision = (
        CLIPVisionConfig(projection_dim=16, **vision_args)
        if clip
        else SiglipVisionConfig(vision_use_head=False, **vision_args)
    )
    text = text_config(variant, tokenizer)
    config = LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
        vision_feature_layer=-2 if clip else -1,
        vision_feature_select_strategy="default" if clip else "full",
    )
    model = LlavaForConditionalGeneration(config).eval()
    model.generation_config = GenerationConfig(
        eos_token_id=[tokenizer.eos_token_id], pad_token_id=tokenizer.unk_token_id
    )
    model.save_pretrained(path)

    size = (
        {"shortest_edge": IMAGE_SIZE}
        if clip
        else {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    )
    if clip:
        images = CLIPImageProcessorPil(
            size=size, crop_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        )
    else:
        images = SiglipImageProcessorPil(size=size)
    processor = LlavaProcessor(
        image_processor=images,
        tokenizer=tokenizer,
        patch_size=PATCH,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        num_additional_image_tokens=1 if clip else 0,
        chat_template=LLAMA_TEMPLATE if variant.llama else CHATML_TEMPLATE,
    )
    processor.save_pretrained(path)
    return TinyLlava(variant, path, model, processor)


@dataclass(frozen=True, slots=True)
class TinyText:
    variant: Variant
    path: Path
    model: Any  # transformers' own LlamaForCausalLM or Qwen2ForCausalLM
    tokenizer: PreTrainedTokenizerFast


def build_tiny_text(path: Path, variant: Variant, seed: int = 0) -> TinyText:
    """A plain text model, the way llama and qwen2 checkpoints are laid out."""
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(variant)
    tokenizer.chat_template = PLAIN_TEMPLATE
    config = text_config(variant, tokenizer)
    model = (LlamaForCausalLM if variant.llama else Qwen2ForCausalLM)(config).eval()
    model.generation_config = GenerationConfig(
        eos_token_id=[tokenizer.eos_token_id], pad_token_id=tokenizer.unk_token_id
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return TinyText(variant, path, model, tokenizer)


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

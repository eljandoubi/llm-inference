"""Random llava and plain text checkpoints written to disk, in the real file layouts.

They carry the config keys, tokenizer and processor classes of real models (llava-1.5
is clip + llama, llava-interleave is siglip + qwen2), with a few thousand parameters or
the size of a small real model. The transformers model that wrote a checkpoint is
returned with it, as the reference to compare the server with.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
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


@dataclass(frozen=True, slots=True)
class Sizes:
    """How big a random model is. The defaults are tiny, tests run in milliseconds."""

    hidden: int = 32
    intermediate: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    max_positions: int = 128
    vocab_size: int = 0  # at least this many embedding rows, the tokenizer needs fewer
    init_range: float = 0.3  # wide logits, so greedy decoding has no near ties
    vision_hidden: int = 32
    vision_intermediate: int = 64
    vision_layers: int = 3
    vision_heads: int = 4
    image_size: int = IMAGE_SIZE
    patch: int = PATCH


TINY = Sizes()

# about SmolVLM-256M: a 135M llama (SmolLM2) and a ViT-B tower, 196 tokens an image
SMALL = Sizes(
    hidden=576,
    intermediate=1536,
    layers=30,
    heads=9,
    kv_heads=3,
    max_positions=2048,
    vocab_size=49152,
    init_range=0.02,
    vision_hidden=768,
    vision_intermediate=3072,
    vision_layers=12,
    vision_heads=12,
    image_size=224,
    patch=16,
)
PRESETS = {"tiny": TINY, "small": SMALL}

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


def build_tokenizer(variant: Variant, vocab_size: int = 0) -> PreTrainedTokenizerFast:
    """One token per byte plus the specials, which is all a test needs.

    A bigger vocab_size adds plain tokens (t300, t301, ...) so that a model with a
    real-sized embedding table still only picks ids that decode to text.
    """
    specials = ["<unk>", "<image>"]
    specials += ["<s>", "</s>"] if variant.llama else ["<|im_start|>", "<|im_end|>"]
    vocab = {
        tok: i for i, tok in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    for special in specials:
        vocab[special] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"t{len(vocab)}"] = len(vocab)
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


def text_config(
    variant: Variant, tokenizer: PreTrainedTokenizerFast, sizes: Sizes = TINY
) -> Any:
    """The text model of a variant: llama with its own head, qwen2 with a tied one."""
    args = {
        "vocab_size": max(len(tokenizer), sizes.vocab_size),
        "hidden_size": sizes.hidden,
        "intermediate_size": sizes.intermediate,
        "num_hidden_layers": sizes.layers,
        "num_attention_heads": sizes.heads,
        "num_key_value_heads": sizes.kv_heads,
        "max_position_embeddings": sizes.max_positions,
        "initializer_range": sizes.init_range,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    if variant.llama:
        return LlamaConfig(**args)
    return Qwen2Config(**args, tie_word_embeddings=True)


def build_tiny_llava(
    path: Path,
    variant: Variant,
    seed: int = 0,
    sizes: Sizes = TINY,
    dtype: torch.dtype = torch.float32,
) -> TinyLlava:
    """Write a random llava checkpoint of the given size (small enough, by default,
    for a unit test) and keep the transformers model that wrote it."""
    torch.manual_seed(seed)
    tokenizer = build_tokenizer(variant, sizes.vocab_size)
    clip = variant.vision == "clip"
    vision_args = {
        "hidden_size": sizes.vision_hidden,
        "intermediate_size": sizes.vision_intermediate,
        "num_hidden_layers": sizes.vision_layers,
        "num_attention_heads": sizes.vision_heads,
        "image_size": sizes.image_size,
        "patch_size": sizes.patch,
    }
    vision = (
        CLIPVisionConfig(projection_dim=16, **vision_args)
        if clip
        else SiglipVisionConfig(vision_use_head=False, **vision_args)
    )
    text = text_config(variant, tokenizer, sizes)
    config = LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
        vision_feature_layer=-2 if clip else -1,
        vision_feature_select_strategy="default" if clip else "full",
    )
    model = LlavaForConditionalGeneration(config).eval().to(dtype)
    model.generation_config = GenerationConfig(
        eos_token_id=[tokenizer.eos_token_id], pad_token_id=tokenizer.unk_token_id
    )
    model.save_pretrained(path)

    side = sizes.image_size
    if clip:
        images = CLIPImageProcessorPil(
            size={"shortest_edge": side}, crop_size={"height": side, "width": side}
        )
    else:
        images = SiglipImageProcessorPil(size={"height": side, "width": side})
    processor = LlavaProcessor(
        image_processor=images,
        tokenizer=tokenizer,
        patch_size=sizes.patch,
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

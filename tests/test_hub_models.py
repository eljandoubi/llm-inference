"""Checks against the real files of real models, with no weights downloaded.

Needs the network, so they only run on request:  pytest -m hub
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor

from vlmserve.frontend import ImagePart, TextPart, Turn
from vlmserve.hf.frontend import HFFrontend
from vlmserve.hf.llava import llava_images
from vlmserve.hf.loader import eos_token_ids, open_model
from vlmserve.model.checkpoint import find_text_prefix
from vlmserve.model.decoder import Decoder

pytestmark = pytest.mark.hub

# small files only: configs, tokenizer, processor, chat template, the weight index
SMALL_FILES = ["*.json", "*.jinja", "merges.txt"]


@dataclass(frozen=True, slots=True)
class RealModel:
    repo: str
    # hidden size, layers, heads, kv heads, vocab, qkv bias, tied embeddings
    decoder: tuple[int, int, int, int, int, bool, bool]
    tokens_per_image: int
    image_token_id: int


MODELS = [
    RealModel(
        "llava-hf/llava-1.5-7b-hf", (4096, 32, 32, 32, 32064, False, False), 576, 32000
    ),
    RealModel(
        "llava-hf/llava-interleave-qwen-0.5b-hf",
        (1024, 24, 16, 16, 152000, True, True),
        729,
        151646,
    ),
]


@pytest.fixture(scope="module", params=MODELS, ids=lambda m: m.repo.split("/")[1])
def real(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[RealModel, Path]:
    model: RealModel = request.param
    folder = tmp_path_factory.mktemp(model.repo.split("/")[1])
    snapshot_download(model.repo, allow_patterns=SMALL_FILES, local_dir=folder)
    return model, Path(folder)


def test_the_config_is_read_into_the_decoder_we_expect(
    real: tuple[RealModel, Path],
) -> None:
    model, folder = real
    source = open_model(str(folder))
    d = source.decoder
    got = (
        d.hidden_size,
        d.num_layers,
        d.num_heads,
        d.num_kv_heads,
        d.vocab_size,
        d.qkv_bias,
        d.tie_word_embeddings,
    )
    assert got == model.decoder


def test_a_prompt_is_what_the_real_processor_makes(
    real: tuple[RealModel, Path],
) -> None:
    model, folder = real
    source = open_model(str(folder))
    processor = AutoProcessor.from_pretrained(folder)
    support = llava_images(processor, source.config)
    assert support.token_id == model.image_token_id
    assert support.tokens_per_image == model.tokens_per_image

    frontend = HFFrontend(
        processor.tokenizer,
        processor,
        eos_token_ids(folder, processor.tokenizer, source.text_config),
        support,
    )
    photo = Image.new("RGB", (640, 480), (30, 90, 200))
    got = frontend.build_prompt(
        [Turn("user", [ImagePart(photo, "k"), TextPart("What is in this picture?")])]
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this picture?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    want = processor(images=[photo], text=text, return_tensors="pt")
    assert got.token_ids == want["input_ids"][0].tolist()
    (segment,) = got.segments
    assert segment.length == model.tokens_per_image
    assert (segment.inputs["pixel_values"] == want["pixel_values"][0]).all()


def test_every_decoder_weight_has_a_tensor_in_the_weight_index(
    real: tuple[RealModel, Path],
) -> None:
    _, folder = real
    index = folder / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip("one file of weights, no index to read the names from")
    names = list(json.loads(index.read_text())["weight_map"])
    prefix = find_text_prefix(names)
    text_names = {n.removeprefix(prefix) for n in names if n.startswith(prefix)}
    if any(n.endswith("lm_head.weight") for n in names):
        text_names.add("lm_head.weight")

    with torch.device("meta"):
        decoder = Decoder(
            open_model(str(folder)).decoder, 8, torch.device("meta"), torch.float16
        )
    missing = []
    for param in decoder.state_dict():
        if "qkv_proj" in param:
            wanted = [re.sub("qkv_proj", f"{p}_proj", param) for p in "qkv"]
        elif "gate_up_proj" in param:
            wanted = [
                re.sub("gate_up_proj", f"{p}_proj", param) for p in ("gate", "up")
            ]
        else:
            wanted = [param]
        missing += [w for w in wanted if w not in text_names]
    assert missing == []

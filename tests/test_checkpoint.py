import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.dense import random_weights
from tests.models import tiny_config
from vlmserve.model.checkpoint import Checkpoint, find_text_prefix
from vlmserve.model.decoder import Decoder
from vlmserve.model.weights import load_decoder_weights


def save(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))


def prefixed(
    weights: dict[str, torch.Tensor], text: str, head: str
) -> dict[str, torch.Tensor]:
    out = {text + k: v for k, v in weights.items() if k != "lm_head.weight"}
    out[head] = weights["lm_head.weight"]
    return out


def test_plain_llama_layout(tmp_path: Path) -> None:
    weights = random_weights(tiny_config())
    save(tmp_path / "model.safetensors", prefixed(weights, "model.", "lm_head.weight"))
    got = dict(Checkpoint(tmp_path).text_weights())
    assert got.keys() == weights.keys()
    torch.testing.assert_close(
        got["layers.1.mlp.up_proj.weight"], weights["layers.1.mlp.up_proj.weight"]
    )


@pytest.mark.parametrize(
    ("text", "head"),
    [
        ("language_model.model.", "language_model.lm_head.weight"),  # older llava
        ("model.language_model.", "lm_head.weight"),  # newer llava
        ("model.text_model.", "lm_head.weight"),  # idefics3
    ],
)
def test_multimodal_layouts_only_yield_the_text_model(
    tmp_path: Path, text: str, head: str
) -> None:
    weights = random_weights(tiny_config())
    tensors = prefixed(weights, text, head)
    tensors["model.vision_tower.encoder.weight"] = torch.zeros(2, 2)
    tensors["multi_modal_projector.linear_1.weight"] = torch.zeros(2, 2)
    save(tmp_path / "model.safetensors", tensors)
    got = dict(Checkpoint(tmp_path).text_weights())
    assert got.keys() == weights.keys()


def test_the_streamed_text_weights_load_into_a_decoder(tmp_path: Path) -> None:
    config = tiny_config()
    weights = random_weights(config)
    save(tmp_path / "model.safetensors", prefixed(weights, "model.", "lm_head.weight"))
    decoder = Decoder(config, 32, torch.device("cpu"), torch.float32)
    assert load_decoder_weights(decoder, Checkpoint(tmp_path).text_weights()) == []
    torch.testing.assert_close(decoder.norm.weight, weights["norm.weight"])


def test_shards_follow_the_index(tmp_path: Path) -> None:
    weights = prefixed(random_weights(tiny_config()), "model.", "lm_head.weight")
    names = sorted(weights)
    half = len(names) // 2
    save(
        tmp_path / "model-00001-of-00002.safetensors",
        {n: weights[n] for n in names[:half]},
    )
    save(
        tmp_path / "model-00002-of-00002.safetensors",
        {n: weights[n] for n in names[half:]},
    )
    weight_map = {
        n: f"model-0000{1 if i < half else 2}-of-00002.safetensors"
        for i, n in enumerate(names)
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    # a stray file that is not part of the model must not be read
    save(tmp_path / "consolidated.safetensors", {"junk": torch.zeros(1)})
    checkpoint = Checkpoint(tmp_path)
    assert sorted(checkpoint.names) == names
    assert len(dict(checkpoint.text_weights())) == len(names)


def test_items_give_back_any_named_tensor(tmp_path: Path) -> None:
    save(tmp_path / "model.safetensors", {"a": torch.ones(2), "b": torch.zeros(3)})
    got = dict(Checkpoint(tmp_path).items(["b"]))
    assert list(got) == ["b"]
    assert got["b"].shape == (3,)


@pytest.mark.parametrize("prefix", ["", "model.", "a.b."])
def test_a_submodule_is_found_wherever_it_sits(tmp_path: Path, prefix: str) -> None:
    save(
        tmp_path / "model.safetensors",
        {
            f"{prefix}vision_tower.layers.0.weight": torch.ones(2),
            f"{prefix}vision_tower.norm.bias": torch.zeros(3),
            f"{prefix}multi_modal_projector.linear_1.weight": torch.ones(4),
            "language_model.layers.0.weight": torch.ones(5),
        },
    )
    got = dict(Checkpoint(tmp_path).submodule("vision_tower"))
    assert sorted(got) == ["layers.0.weight", "norm.bias"]
    assert got["norm.bias"].shape == (3,)


def test_a_submodule_name_has_to_match_a_whole_word(tmp_path: Path) -> None:
    save(
        tmp_path / "model.safetensors",
        {
            "my_vision_tower.weight": torch.ones(1),
            "vision_tower_extra.weight": torch.ones(1),
        },
    )
    assert dict(Checkpoint(tmp_path).submodule("vision_tower")) == {}


def test_two_embedding_tables_are_ambiguous() -> None:
    with pytest.raises(ValueError, match="can't tell"):
        find_text_prefix(["a.embed_tokens.weight", "b.embed_tokens.weight"])
    with pytest.raises(ValueError, match="can't tell"):
        find_text_prefix(["layers.0.weight"])


def test_a_directory_without_weights_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no \.safetensors"):
        Checkpoint(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        Checkpoint(tmp_path / "nope")

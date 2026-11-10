import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig, LlavaConfig

from tests.tinyhf import (
    LAYOUTS,
    TOKENS_PER_IMAGE,
    TinyLlava,
    picture,
    rewrite_weights,
)
from vlmserve.hf.llava import LlavaVision, load_llava_vision, tokens_per_image
from vlmserve.model.checkpoint import Checkpoint


def llava_config(path: Path) -> LlavaConfig:
    config = AutoConfig.from_pretrained(path)
    assert isinstance(config, LlavaConfig)
    return config


def pixel_inputs(tiny: TinyLlava, seeds: list[int]) -> list[dict[str, torch.Tensor]]:
    images = [picture(seed) for seed in seeds]
    pixels = tiny.processor.image_processor(images=images, return_tensors="pt")
    return [{"pixel_values": p} for p in pixels["pixel_values"]]


def reference(
    tiny: TinyLlava, inputs: list[dict[str, torch.Tensor]]
) -> list[torch.Tensor]:
    config = tiny.model.config
    pixels = torch.stack([i["pixel_values"] for i in inputs])
    with torch.no_grad():
        out = tiny.model.model.get_image_features(
            pixels,
            vision_feature_layer=config.vision_feature_layer,
            vision_feature_select_strategy=config.vision_feature_select_strategy,
        )
    return list(out.pooler_output)


def encoder(
    tiny: TinyLlava, device: torch.device, path: Path | None = None
) -> LlavaVision:
    folder = path or tiny.path
    return load_llava_vision(
        llava_config(folder), Checkpoint(folder), device, torch.float32
    )


def test_it_counts_the_placeholders_one_image_takes(tiny_llava: TinyLlava) -> None:
    assert tokens_per_image(llava_config(tiny_llava.path)) == TOKENS_PER_IMAGE


@pytest.mark.parametrize(
    ("tower", "size", "strategy", "expected"),
    [
        ("clip_vision_model", 336, "default", 576),  # llava-1.5
        ("clip_vision_model", 336, "full", 577),
        ("siglip_vision_model", 384, "full", 729),  # llava-interleave
    ],
)
def test_the_count_for_real_sized_towers(
    tower: str, size: int, strategy: str, expected: int
) -> None:
    config = LlavaConfig(
        vision_config={"model_type": tower, "image_size": size, "patch_size": 14},
        text_config={"model_type": "llama"},
        vision_feature_select_strategy=strategy,
    )
    assert tokens_per_image(config) == expected


def test_the_embeddings_match_transformers(
    tiny_llava: TinyLlava, device: torch.device
) -> None:
    inputs = pixel_inputs(tiny_llava, [1, 2, 3])
    ours = encoder(tiny_llava, device).encode(inputs)
    theirs = reference(tiny_llava, inputs)
    assert len(ours) == len(theirs) == 3
    for got, want in zip(ours, theirs, strict=True):
        assert got.device.type == device.type
        assert got.shape == (TOKENS_PER_IMAGE, want.shape[-1])
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-4)


def test_an_image_gives_the_same_embeddings_alone_or_in_a_batch(
    tiny_llava: TinyLlava,
) -> None:
    vision = encoder(tiny_llava, torch.device("cpu"))
    inputs = pixel_inputs(tiny_llava, [4, 5])
    together = vision.encode(inputs)
    for item, batched in zip(inputs, together, strict=True):
        (alone,) = vision.encode([item])
        torch.testing.assert_close(alone, batched, rtol=1e-5, atol=1e-5)


def test_cached_embeddings_do_not_share_one_buffer(tiny_llava: TinyLlava) -> None:
    first, second = encoder(tiny_llava, torch.device("cpu")).encode(
        pixel_inputs(tiny_llava, [1, 2])
    )
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()


@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=LAYOUTS.keys())
def test_files_saved_by_other_transformers_versions_load_the_same(
    tiny_llava: TinyLlava, tmp_path: Path, layout: Callable[[str], str]
) -> None:
    other = rewrite_weights(tiny_llava.path, tmp_path / "other", layout)
    with safe_open(other / "model.safetensors", framework="pt") as f:
        names = list(f.keys())
    assert any("vision_tower." in n for n in names)
    inputs = pixel_inputs(tiny_llava, [1, 2])
    cpu = torch.device("cpu")
    for got, want in zip(
        encoder(tiny_llava, cpu, other).encode(inputs),
        encoder(tiny_llava, cpu).encode(inputs),
        strict=True,
    ):
        torch.testing.assert_close(got, want)


def test_a_tower_weight_that_is_missing_is_an_error(
    tiny_llava: TinyLlava, tmp_path: Path
) -> None:
    broken = rewrite_weights(
        tiny_llava.path,
        tmp_path / "broken",
        lambda n: None if n.endswith("layers.1.mlp.fc1.weight") else n,
    )
    with pytest.raises(ValueError, match=r"vision tower: 1 weights were never loaded"):
        encoder(tiny_llava, torch.device("cpu"), broken)


def test_the_unused_pooling_weights_may_be_missing(
    tiny_llava: TinyLlava, tmp_path: Path
) -> None:
    lean = rewrite_weights(
        tiny_llava.path,
        tmp_path / "lean",
        lambda n: None if "post_layernorm" in n or ".head." in n else n,
    )
    inputs = pixel_inputs(tiny_llava, [1])
    cpu = torch.device("cpu")
    (got,) = encoder(tiny_llava, cpu, lean).encode(inputs)
    (want,) = encoder(tiny_llava, cpu).encode(inputs)
    torch.testing.assert_close(got, want)


def test_a_projector_weight_that_is_missing_is_an_error(
    tiny_llava: TinyLlava, tmp_path: Path
) -> None:
    broken = rewrite_weights(
        tiny_llava.path,
        tmp_path / "broken",
        lambda n: None if "multi_modal_projector.linear_2.bias" in n else n,
    )
    with pytest.raises(ValueError, match="projector: 1 weights were never loaded"):
        encoder(tiny_llava, torch.device("cpu"), broken)


def test_a_tensor_of_the_wrong_shape_is_an_error(
    tiny_llava: TinyLlava, tmp_path: Path
) -> None:
    shapes = rewrite_weights(tiny_llava.path, tmp_path / "shapes", lambda n: n)
    config = llava_config(shapes)
    config.vision_config.hidden_size = 48  # the files still say 32
    with pytest.raises(ValueError, match="the model expects"):
        load_llava_vision(
            config, Checkpoint(shapes), torch.device("cpu"), torch.float32
        )


def test_an_unsupported_tower_is_refused(tiny_llava: TinyLlava) -> None:
    config = copy.deepcopy(llava_config(tiny_llava.path))
    config.vision_config.model_type = "pixtral"
    with pytest.raises(ValueError, match=r"pixtral.*not supported"):
        load_llava_vision(
            config, Checkpoint(tiny_llava.path), torch.device("cpu"), torch.float32
        )

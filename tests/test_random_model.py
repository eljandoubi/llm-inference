from pathlib import Path

import pytest

from benchmarks import random_model
from tests.tinyhf import TINY
from vlmserve.hf.loader import open_model


def test_a_random_model_can_be_written_and_opened(tmp_path: Path) -> None:
    assert random_model.main([str(tmp_path / "m"), "--preset", "tiny"]) == 0
    source = open_model(str(tmp_path / "m"))
    assert source.multimodal
    assert source.decoder.hidden_size == TINY.hidden


def test_the_small_preset_is_the_size_of_a_small_real_model() -> None:
    small = random_model.PRESETS["small"]
    text = (
        small.layers
        * (
            2 * small.hidden * small.hidden  # q and o
            + 2 * small.hidden * (small.hidden // small.heads * small.kv_heads)  # k, v
            + 3 * small.hidden * small.intermediate
        )
        + 2 * small.vocab_size * small.hidden
    )  # embeddings and head
    assert 130e6 < text < 180e6
    assert (small.image_size // small.patch) ** 2 == 196


def test_an_unknown_preset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        random_model.main([str(tmp_path), "--preset", "huge"])

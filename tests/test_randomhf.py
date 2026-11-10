from pathlib import Path

import pytest

from randomhf.__main__ import main
from randomhf.checkpoints import CLIP_LLAMA, PRESETS, TINY, build_tokenizer
from vlmserve.hf.loader import open_model


def test_a_random_model_can_be_written_and_opened(tmp_path: Path) -> None:
    assert main([str(tmp_path / "m"), "--preset", "tiny"]) == 0
    source = open_model(str(tmp_path / "m"))
    assert source.multimodal
    assert source.decoder.hidden_size == TINY.hidden


def test_the_small_preset_is_the_size_of_a_small_real_model() -> None:
    small = PRESETS["small"]
    kv_width = small.hidden // small.heads * small.kv_heads
    per_layer = (
        2 * small.hidden * small.hidden  # q and o
        + 2 * small.hidden * kv_width  # k and v
        + 3 * small.hidden * small.intermediate  # gate, up, down
    )
    embeddings_and_head = 2 * small.vocab_size * small.hidden
    text = small.layers * per_layer + embeddings_and_head
    assert 130e6 < text < 180e6
    assert (small.image_size // small.patch) ** 2 == 196


def test_an_unknown_preset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--preset", "huge"])


def test_a_padded_vocab_still_decodes_every_id_to_text() -> None:
    tokenizer = build_tokenizer(CLIP_LLAMA, vocab_size=300)
    assert len(tokenizer) == 300
    assert tokenizer.decode([299]) == "t299"
    assert all(tokenizer.decode([i]) for i in range(260, 300))

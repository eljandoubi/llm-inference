from typing import Any

import pytest

from vlmserve.config import EngineConfig


def test_defaults() -> None:
    config = EngineConfig(model="some/model")
    assert config.max_blocks_per_seq == 256
    assert config.enable_prefix_caching


def test_blocks_per_seq_rounds_up() -> None:
    assert EngineConfig("m", max_model_len=100, block_size=16).max_blocks_per_seq == 7
    assert EngineConfig("m", max_model_len=96, block_size=16).max_blocks_per_seq == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_model_len": 1},
        {"max_num_seqs": 0},
        {"max_num_seqs": 64, "max_num_batched_tokens": 63},
        {"block_size": 0},
        {"num_kv_blocks": 1},
        {"kv_memory_fraction": 0.0},
        {"kv_memory_fraction": 1.5},
        {"max_encoder_tokens_per_step": 0},
        {"encoder_cache_tokens": -1},
    ],
)
def test_bad_values_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        EngineConfig("m", **kwargs)


def test_explicit_block_count_wins() -> None:
    config = EngineConfig("m", num_kv_blocks=10)
    assert config.plan_num_blocks(bytes_per_block=1, free_bytes=10**12) == 10


def test_blocks_follow_the_free_memory() -> None:
    config = EngineConfig("m", kv_memory_fraction=0.5)
    # half of 1000 bytes, 100 bytes a block
    assert config.plan_num_blocks(bytes_per_block=100, free_bytes=1000) == 5


def test_blocks_are_capped_at_what_the_sequences_can_use() -> None:
    config = EngineConfig("m", max_model_len=64, block_size=16, max_num_seqs=2)
    # 2 sequences * 4 blocks + the null block
    assert config.plan_num_blocks(bytes_per_block=1, free_bytes=10**9) == 9


def test_cache_must_hold_one_full_sequence() -> None:
    config = EngineConfig("m", max_model_len=64, block_size=16)
    config.check_cache_size(5)  # 4 usable blocks = 64 tokens
    with pytest.raises(ValueError, match="holds 48 tokens"):
        config.check_cache_size(4)

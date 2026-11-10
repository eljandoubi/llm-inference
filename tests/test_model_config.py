from typing import Any

import pytest
import torch

from vlmserve.model.config import DecoderConfig

LLAMA: dict[str, Any] = {
    "model_type": "llama",
    "vocab_size": 49152,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "max_position_embeddings": 8192,
    "tie_word_embeddings": True,
}


def test_reads_a_llama_config() -> None:
    cfg = DecoderConfig.from_hf(LLAMA)
    assert cfg.num_layers == 30
    assert cfg.num_heads == 9
    assert cfg.num_kv_heads == 3
    assert cfg.head_dim == 64  # 576 / 9
    assert cfg.rope_theta == 100000.0
    assert cfg.tie_word_embeddings
    assert not cfg.qkv_bias


def test_qwen2_always_has_qkv_biases() -> None:
    cfg = DecoderConfig.from_hf({**LLAMA, "model_type": "qwen2"})
    assert cfg.qkv_bias


def test_missing_kv_heads_means_plain_multi_head_attention() -> None:
    raw = {k: v for k, v in LLAMA.items() if k != "num_key_value_heads"}
    assert DecoderConfig.from_hf(raw).num_kv_heads == 9


def test_explicit_head_dim_wins() -> None:
    assert DecoderConfig.from_hf({**LLAMA, "head_dim": 128}).head_dim == 128


def test_rope_theta_can_sit_in_the_rope_block() -> None:
    raw = {k: v for k, v in LLAMA.items() if k != "rope_theta"}
    raw["rope_parameters"] = {"rope_type": "default", "rope_theta": 5e5}
    assert DecoderConfig.from_hf(raw).rope_theta == 5e5


def test_scaled_rope_is_refused_loudly() -> None:
    raw = {**LLAMA, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}
    with pytest.raises(NotImplementedError, match="llama3"):
        DecoderConfig.from_hf(raw)


def test_unknown_architectures_are_refused() -> None:
    with pytest.raises(ValueError, match="gemma"):
        DecoderConfig.from_hf({**LLAMA, "model_type": "gemma"})


def test_heads_must_divide_into_kv_heads() -> None:
    with pytest.raises(ValueError, match="can't share"):
        DecoderConfig.from_hf({**LLAMA, "num_key_value_heads": 4})


def test_kv_memory_per_block() -> None:
    cfg = DecoderConfig.from_hf(LLAMA)
    # K and V, 30 layers, 3 kv heads, 64 dims, 16 tokens, 2 bytes each
    assert cfg.kv_block_bytes(16, torch.float16) == 2 * 30 * 3 * 64 * 16 * 2

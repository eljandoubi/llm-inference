from dataclasses import replace
from typing import Any

from vlmserve.model.config import DecoderConfig
from vlmserve.model.decoder import Attention, Decoder, DecoderLayer

TINY = DecoderConfig(
    vocab_size=50,
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
)


def tiny_config(**overrides: Any) -> DecoderConfig:
    return replace(TINY, **overrides)


def attention_of(decoder: Decoder, index: int) -> Attention:
    layer = decoder.layers[index]
    assert isinstance(layer, DecoderLayer)
    return layer.self_attn

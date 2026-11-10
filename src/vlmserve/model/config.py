from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

SUPPORTED_TEXT_MODELS = ("llama", "qwen2")


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    qkv_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"{self.num_heads} heads can't share {self.num_kv_heads} kv heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def kv_elements_per_token(self) -> int:
        """K and V of every layer, for one token."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim

    def kv_block_bytes(self, block_size: int, dtype: torch.dtype) -> int:
        return self.kv_elements_per_token * block_size * dtype.itemsize

    @classmethod
    def from_hf(cls, cfg: Mapping[str, Any]) -> "DecoderConfig":
        """Read the text part of a Hugging Face config.json (already unnested)."""
        model_type = cfg.get("model_type", "llama")
        if model_type not in SUPPORTED_TEXT_MODELS:
            raise ValueError(
                f"text model {model_type!r} is not supported, "
                f"only {', '.join(SUPPORTED_TEXT_MODELS)}"
            )
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        rope_type = rope.get("rope_type") or rope.get("type") or "default"
        if rope_type != "default":
            raise NotImplementedError(f"rope scaling {rope_type!r} is not implemented")
        heads = cfg["num_attention_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=heads,
            num_kv_heads=cfg.get("num_key_value_heads") or heads,
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta") or rope.get("rope_theta") or 10000),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            # qwen2 always has biases on q, k and v even though its config is silent
            qkv_bias=model_type == "qwen2" or bool(cfg.get("attention_bias", False)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )

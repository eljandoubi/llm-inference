from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vlmserve.model.attention import AttentionMetadata, paged_attention, write_kv
from vlmserve.model.config import DecoderConfig
from vlmserve.model.layers import MLP, RMSNorm, Rotary, apply_rope


class Attention(nn.Module):
    def __init__(
        self, config: DecoderConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_size = config.num_heads * config.head_dim
        self.kv_size = config.num_kv_heads * config.head_dim
        self.scale = config.head_dim**-0.5
        # q, k and v come out of one matmul
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=config.qkv_bias,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.q_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )
        self.kv_cache: torch.Tensor | None = None  # this layer's slice of the cache

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("allocate_kv_cache() has to run before the first step")
        n = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], -1)
        q = apply_rope(q.view(n, self.num_heads, self.head_dim), cos, sin)
        k = apply_rope(k.view(n, self.num_kv_heads, self.head_dim), cos, sin)
        v = v.reshape(n, self.num_kv_heads, self.head_dim)
        write_kv(self.kv_cache, meta.slot_mapping, k, v)
        return self.o_proj(paged_attention(q, self.kv_cache, meta, self.scale))


class DecoderLayer(nn.Module):
    def __init__(
        self, config: DecoderConfig, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        eps, size = config.rms_norm_eps, config.hidden_size
        self.input_layernorm = RMSNorm(size, eps, device, dtype)
        self.self_attn = Attention(config, device, dtype)
        self.post_attention_layernorm = RMSNorm(size, eps, device, dtype)
        self.mlp = MLP(size, config.intermediate_size, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    """A llama-style decoder over a packed batch of tokens, with a paged KV cache.

    Tokens of all sequences are laid out one after another along dim 0, the
    attention metadata says which rows belong to which sequence.
    """

    def __init__(
        self,
        config: DecoderConfig,
        max_positions: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, device, dtype) for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.lm_head: nn.Linear | None = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
        self.rotary = Rotary(config.head_dim, config.rope_theta, max_positions, device)
        self.kv_cache: torch.Tensor | None = None
        self.requires_grad_(False)  # inference only

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor, meta: AttentionMetadata
    ) -> torch.Tensor:
        """hidden [T, hidden_size] are input embeddings, positions [T] are absolute."""
        cos, sin = self.rotary(positions)
        for layer in self.layers:
            hidden = cast(DecoderLayer, layer)(hidden, cos, sin, meta)
        return self.norm(hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, head)

    def allocate_kv_cache(self, num_blocks: int, block_size: int) -> torch.Tensor:
        cfg = self.config
        cache = torch.zeros(
            cfg.num_layers,
            2,
            cfg.num_kv_heads,
            num_blocks,
            block_size,
            cfg.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        # cache[i], not iterating over the tensor: the views unbind() hands out can't
        # be written in place while autograd is on
        for i, layer in enumerate(self.layers):
            cast(DecoderLayer, layer).self_attn.kv_cache = cache[i]
        self.kv_cache = cache
        return cache

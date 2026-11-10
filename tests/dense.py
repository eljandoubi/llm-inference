import math

import torch

from vlmserve.model.config import DecoderConfig


def random_weights(config: DecoderConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random weights under the names a hf llama/qwen2 checkpoint uses."""
    gen = torch.Generator().manual_seed(seed)
    hidden, inter = config.hidden_size, config.intermediate_size
    q_size = config.num_heads * config.head_dim
    kv_size = config.num_kv_heads * config.head_dim

    def mat(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * 0.15

    def norm() -> torch.Tensor:
        return 1 + 0.1 * torch.randn(hidden, generator=gen)

    weights = {"embed_tokens.weight": mat(config.vocab_size, hidden)}
    for i in range(config.num_layers):
        p = f"layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = norm()
        weights[f"{p}.post_attention_layernorm.weight"] = norm()
        for name, size in (("q", q_size), ("k", kv_size), ("v", kv_size)):
            weights[f"{p}.self_attn.{name}_proj.weight"] = mat(size, hidden)
            if config.qkv_bias:
                weights[f"{p}.self_attn.{name}_proj.bias"] = mat(size)
        weights[f"{p}.self_attn.o_proj.weight"] = mat(hidden, q_size)
        weights[f"{p}.mlp.gate_proj.weight"] = mat(inter, hidden)
        weights[f"{p}.mlp.up_proj.weight"] = mat(inter, hidden)
        weights[f"{p}.mlp.down_proj.weight"] = mat(hidden, inter)
    weights["norm.weight"] = norm()
    if not config.tie_word_embeddings:
        weights["lm_head.weight"] = mat(config.vocab_size, hidden)
    return weights


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    length, _, dim = x.shape
    inv_freq = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    angles = torch.arange(length, dtype=torch.float64)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos().float()[:, None, :], angles.sin().float()[:, None, :]
    first, second = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


def dense_greedy(
    config: DecoderConfig,
    w: dict[str, torch.Tensor],
    prompt: list[int],
    steps: int,
    image_rows: tuple[int, torch.Tensor] | None = None,
) -> list[int]:
    """Greedy continuation by recomputing everything at every step.

    image_rows is (first position, embeddings) of an image inside the prompt.
    """
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(steps):
        embeds = w["embed_tokens.weight"][torch.tensor(tokens)].clone()
        if image_rows is not None:
            start, rows = image_rows
            embeds[start : start + len(rows)] = rows
        logits = dense_logits(config, w, tokens, embeds)[-1]
        first, second = logits.topk(2).values.tolist()
        assert first - second > 1e-3, "a near tie could flip, try another seed"
        out.append(int(logits.argmax()))
        tokens.append(out[-1])
    return out


def dense_logits(
    config: DecoderConfig,
    w: dict[str, torch.Tensor],
    token_ids: list[int],
    embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """The slow, obvious forward pass over a whole sequence: no fusing, no cache.

    embeds, if given, replaces the embedding lookup (that is how images get in).
    Returns the logits of every position, [len(token_ids), vocab], in float32.
    """
    heads, kv_heads, dim = config.num_heads, config.num_kv_heads, config.head_dim
    length = len(token_ids)
    x = w["embed_tokens.weight"][torch.tensor(token_ids)] if embeds is None else embeds
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    for i in range(config.num_layers):
        p = f"layers.{i}"
        h = _rms(x, w[f"{p}.input_layernorm.weight"], config.rms_norm_eps)

        def project(name: str, h: torch.Tensor = h, p: str = p) -> torch.Tensor:
            out = h @ w[f"{p}.self_attn.{name}_proj.weight"].T
            bias = w.get(f"{p}.self_attn.{name}_proj.bias")
            return out if bias is None else out + bias

        q = _rope(project("q").view(length, heads, dim), config.rope_theta)
        k = _rope(project("k").view(length, kv_heads, dim), config.rope_theta)
        v = project("v").view(length, kv_heads, dim)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dim)
        scores = scores.masked_fill(~causal, float("-inf"))
        attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        x = (
            x
            + attended.reshape(length, heads * dim)
            @ w[f"{p}.self_attn.o_proj.weight"].T
        )

        h = _rms(x, w[f"{p}.post_attention_layernorm.weight"], config.rms_norm_eps)
        gate = torch.nn.functional.silu(h @ w[f"{p}.mlp.gate_proj.weight"].T)
        x = (
            x
            + (gate * (h @ w[f"{p}.mlp.up_proj.weight"].T))
            @ w[f"{p}.mlp.down_proj.weight"].T
        )
    x = _rms(x, w["norm.weight"], config.rms_norm_eps)
    return x @ w.get("lm_head.weight", w["embed_tokens.weight"]).T

import random
from dataclasses import replace
from typing import Any

import torch

from vlmserve.model.attention import SeqPlan, build_metadata
from vlmserve.model.config import DecoderConfig
from vlmserve.model.decoder import Attention, Decoder, DecoderLayer
from vlmserve.model.weights import load_decoder_weights

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


def layer_of(decoder: Decoder, index: int) -> DecoderLayer:
    layer = decoder.layers[index]
    assert isinstance(layer, DecoderLayer)
    return layer


def attention_of(decoder: Decoder, index: int) -> Attention:
    return layer_of(decoder, index).self_attn


def build_decoder(
    config: DecoderConfig,
    weights: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Decoder:
    decoder = Decoder(config, 64, device, dtype)
    load_decoder_weights(decoder, weights.items())
    return decoder


class PagedRun:
    """Drives a decoder like the engine will, with scattered blocks."""

    def __init__(
        self,
        decoder: Decoder,
        block_size: int = 4,
        num_blocks: int = 60,
        seed: int = 0,
    ) -> None:
        self.decoder = decoder
        self.block_size = block_size
        decoder.allocate_kv_cache(num_blocks, block_size)
        self.free = list(range(1, num_blocks))
        random.Random(seed).shuffle(self.free)
        self.blocks: dict[str, list[int]] = {}
        self.cached: dict[str, int] = {}
        self.tokens: dict[str, list[int]] = {}  # everything a sequence has seen so far

    def step(self, work: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        """Run new tokens for some sequences, return their logits (float32, cpu)."""
        cfg, device = self.decoder.config, self.decoder.device
        plans: list[SeqPlan] = []
        ids: list[int] = []
        positions: list[int] = []
        for name, new in work.items():
            start = self.cached.get(name, 0)
            blocks = self.blocks.setdefault(name, [])
            while len(blocks) * self.block_size < start + len(new):
                blocks.append(self.free.pop())
            plans.append(SeqPlan(start, len(new), blocks))
            ids += new
            positions += range(start, start + len(new))
        meta = build_metadata(
            plans, self.block_size, cfg.num_heads // cfg.num_kv_heads, device
        )
        hidden = self.decoder.embed(torch.tensor(ids, device=device))
        hidden = self.decoder(hidden, torch.tensor(positions, device=device), meta)
        logits = self.decoder.logits(hidden).float().cpu()
        out: dict[str, torch.Tensor] = {}
        row = 0
        for name, new in work.items():
            out[name] = logits[row : row + len(new)]
            row += len(new)
            self.cached[name] = self.cached.get(name, 0) + len(new)
            self.tokens.setdefault(name, []).extend(new)
        return out

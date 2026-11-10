import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model: str
    device: str = "auto"
    dtype: str = "auto"
    max_model_len: int = 4096
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    block_size: int = 16
    num_kv_blocks: int | None = None  # None: size the cache from the free memory
    kv_memory_fraction: float = 0.8  # share of the memory left after the weights
    enable_prefix_caching: bool = True
    max_encoder_tokens_per_step: int = 4096  # image tokens we may embed in one step
    encoder_cache_tokens: int = 16384  # embeddings kept for reuse, 0 turns it off
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_model_len < 2:
            raise ValueError(f"max_model_len must be >= 2, got {self.max_model_len}")
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")
        if self.max_num_batched_tokens < self.max_num_seqs:
            # every running request needs a token per step or the tail starves
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be "
                f"at least max_num_seqs ({self.max_num_seqs})"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.num_kv_blocks is not None and self.num_kv_blocks < 2:
            raise ValueError("num_kv_blocks must be >= 2, block 0 is reserved")
        if not 0 < self.kv_memory_fraction <= 1:
            raise ValueError(
                f"kv_memory_fraction must be in (0, 1], got {self.kv_memory_fraction}"
            )
        if self.max_encoder_tokens_per_step < 1:
            raise ValueError("max_encoder_tokens_per_step must be >= 1")
        if self.encoder_cache_tokens < 0:
            raise ValueError("encoder_cache_tokens must be >= 0")

    @property
    def max_blocks_per_seq(self) -> int:
        return math.ceil(self.max_model_len / self.block_size)

    def plan_num_blocks(self, bytes_per_block: int, free_bytes: int) -> int:
        if self.num_kv_blocks is not None:
            return self.num_kv_blocks
        by_memory = int(free_bytes * self.kv_memory_fraction) // bytes_per_block
        # more than "every slot full to max_model_len" (+ the null block) is wasted
        return min(by_memory, self.max_num_seqs * self.max_blocks_per_seq + 1)

    def check_cache_size(self, num_blocks: int) -> None:
        capacity = (num_blocks - 1) * self.block_size  # block 0 is never handed out
        if capacity < self.max_model_len:
            raise ValueError(
                f"the KV cache holds {capacity} tokens but max_model_len is "
                f"{self.max_model_len}: lower it or give the cache more memory"
            )

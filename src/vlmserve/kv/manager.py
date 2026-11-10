import math
from collections.abc import Sequence

from vlmserve.kv.hashing import extend_block_hashes
from vlmserve.kv.pool import BlockPool
from vlmserve.request import Request


class KVCacheManager:
    """Decides which blocks hold whose KV, and which of them can be shared."""

    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True
    ) -> None:
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.pool = BlockPool(num_blocks)

    @property
    def usage(self) -> float:
        usable = self.pool.num_blocks - 1
        return (usable - self.pool.num_free) / usable

    def match_prefix(self, req: Request) -> list[int]:
        """Blocks that already hold the longest prefix of this request, in order."""
        if not self.enable_prefix_caching:
            return []
        extend_block_hashes(
            req.block_hashes, req.token_ids, self.block_size, req.segments
        )
        # the last token has to run through the model to give us logits, so it
        # can't come out of the cache
        limit = (req.num_tokens - 1) // self.block_size
        hits: list[int] = []
        for block_hash in req.block_hashes[:limit]:
            block_id = self.pool.lookup(block_hash)
            if block_id is None:
                break
            hits.append(block_id)
        return hits

    def allocate(
        self, req: Request, num_tokens: int, cached_blocks: Sequence[int] = ()
    ) -> bool:
        """Make the block table cover num_tokens tokens.

        cached_blocks (from match_prefix) become the start of a fresh table. Returns
        False and changes nothing if the pool can't spare that many blocks.
        """
        if cached_blocks and req.block_ids:
            raise ValueError("cached blocks can only start a new block table")
        total = math.ceil(num_tokens / self.block_size)
        fresh = max(total - len(req.block_ids) - len(cached_blocks), 0)
        # a cached block that is sitting in the free list leaves it when we take it
        revived = sum(1 for b in cached_blocks if self.pool.ref_count(b) == 0)
        if fresh + revived > self.pool.num_free:
            return False
        for block_id in cached_blocks:
            self.pool.acquire(block_id)
            req.block_ids.append(block_id)
        req.num_committed_blocks = len(cached_blocks) or req.num_committed_blocks
        req.block_ids.extend(self.pool.allocate() for _ in range(fresh))
        return True

    def commit(self, req: Request) -> None:
        """Offer the blocks that are now completely written to other requests."""
        if not self.enable_prefix_caching:
            return
        full = req.num_computed_tokens // self.block_size
        if full <= req.num_committed_blocks:
            return
        extend_block_hashes(
            req.block_hashes, req.token_ids, self.block_size, req.segments
        )
        for i in range(req.num_committed_blocks, full):
            self.pool.register(req.block_ids[i], req.block_hashes[i])
        req.num_committed_blocks = full

    def free(self, req: Request) -> None:
        # tail first, so the prefix is the last thing to be evicted
        for block_id in reversed(req.block_ids):
            self.pool.release(block_id)
        req.block_ids = []
        req.num_committed_blocks = 0

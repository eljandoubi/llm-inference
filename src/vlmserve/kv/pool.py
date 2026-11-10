from collections import OrderedDict


class BlockPool:
    """The physical KV blocks: reference counts, a free list, and the prefix lookup.

    Block 0 is the null block. Padding in a block table points at it and it is
    never handed out.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 2:
            raise ValueError("a pool needs the null block plus at least one more")
        self.num_blocks = num_blocks
        self._ref_counts = [0] * num_blocks
        self._hash_of: list[bytes | None] = [None] * num_blocks
        self._block_of: dict[bytes, int] = {}
        # the front is whatever was freed longest ago: that is evicted first, and
        # blocks that never held anything cached sit in front of all the others
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(1, num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._block_of)

    def ref_count(self, block_id: int) -> int:
        return self._ref_counts[block_id]

    def lookup(self, block_hash: bytes) -> int | None:
        return self._block_of.get(block_hash)

    def allocate(self) -> int:
        """Hand out the least recently freed block, dropping what it used to cache."""
        if not self._free:
            raise RuntimeError("out of KV blocks")
        block_id, _ = self._free.popitem(last=False)
        stale = self._hash_of[block_id]
        if stale is not None:
            del self._block_of[stale]
            self._hash_of[block_id] = None
        self._ref_counts[block_id] = 1
        return block_id

    def acquire(self, block_id: int) -> None:
        """Take another reference on a block, reviving it if it sat in the free list."""
        if self._ref_counts[block_id] == 0:
            del self._free[block_id]
        self._ref_counts[block_id] += 1

    def release(self, block_id: int) -> None:
        if self._ref_counts[block_id] <= 0:
            raise RuntimeError(f"block {block_id} released more often than acquired")
        self._ref_counts[block_id] -= 1
        if self._ref_counts[block_id] == 0:
            self._free[block_id] = None

    def register(self, block_id: int, block_hash: bytes) -> None:
        """Remember that this (full, written) block holds the prefix with this hash."""
        if self._hash_of[block_id] is not None or block_hash in self._block_of:
            # two requests can compute the same prefix side by side, the first wins
            return
        self._hash_of[block_id] = block_hash
        self._block_of[block_hash] = block_id

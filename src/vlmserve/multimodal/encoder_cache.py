from collections import OrderedDict

import torch


class EncoderCache:
    """Recent image embeddings, so a repeated image skips the vision tower.

    The size limit counts tokens (rows of embeddings), not entries. Requests keep
    their own reference while they use an embedding, so evicting one here never
    pulls it out from under a running request.
    """

    def __init__(self, capacity_tokens: int) -> None:
        self.capacity_tokens = capacity_tokens
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._num_tokens = 0

    def __contains__(self, key: str) -> bool:
        return key in self._entries  # a peek: no counters, no change of order

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def get(self, key: str) -> torch.Tensor | None:
        embeddings = self._entries.get(key)
        if embeddings is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return embeddings

    def put(self, key: str, embeddings: torch.Tensor) -> None:
        size = embeddings.shape[0]
        if size > self.capacity_tokens:
            return  # would push everything else out and still not fit
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._num_tokens -= previous.shape[0]
        self._entries[key] = embeddings
        self._num_tokens += size
        while self._num_tokens > self.capacity_tokens:
            _, evicted = self._entries.popitem(last=False)
            self._num_tokens -= evicted.shape[0]

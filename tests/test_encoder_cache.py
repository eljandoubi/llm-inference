import torch

from vlmserve.multimodal.encoder_cache import EncoderCache


def emb(tokens: int, fill: float = 0.0) -> torch.Tensor:
    return torch.full((tokens, 8), fill)


def test_put_then_get() -> None:
    cache = EncoderCache(capacity_tokens=100)
    value = emb(4, 1.0)
    cache.put("a", value)
    assert cache.get("a") is value
    assert len(cache) == 1
    assert cache.num_tokens == 4


def test_counts_hits_and_misses() -> None:
    cache = EncoderCache(100)
    cache.put("a", emb(4))
    cache.get("a")
    cache.get("a")
    cache.get("b")
    assert (cache.hits, cache.misses) == (2, 1)


def test_peeking_changes_neither_counters_nor_order() -> None:
    cache = EncoderCache(capacity_tokens=8)
    cache.put("a", emb(4))
    cache.put("b", emb(4))
    assert "a" in cache
    assert "zzz" not in cache
    assert (cache.hits, cache.misses) == (0, 0)
    cache.put("c", emb(4))  # "a" is still the oldest, in spite of the peek
    assert "a" not in cache
    assert "b" in cache


def test_least_recently_used_goes_first() -> None:
    cache = EncoderCache(capacity_tokens=8)
    cache.put("a", emb(4))
    cache.put("b", emb(4))
    cache.get("a")
    cache.put("c", emb(4))
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache
    assert cache.num_tokens == 8


def test_a_big_entry_can_evict_several() -> None:
    cache = EncoderCache(capacity_tokens=10)
    for key in "abc":
        cache.put(key, emb(3))
    cache.put("big", emb(8))
    assert len(cache) == 1
    assert cache.num_tokens == 8


def test_something_that_can_never_fit_is_not_stored() -> None:
    cache = EncoderCache(capacity_tokens=4)
    cache.put("a", emb(4))
    cache.put("huge", emb(5))
    assert "huge" not in cache
    assert "a" in cache  # and it didn't evict anything on the way


def test_zero_capacity_turns_the_cache_off() -> None:
    cache = EncoderCache(capacity_tokens=0)
    cache.put("a", emb(1))
    assert len(cache) == 0
    assert cache.get("a") is None


def test_putting_a_key_again_replaces_it_and_fixes_the_count() -> None:
    cache = EncoderCache(capacity_tokens=10)
    cache.put("a", emb(4))
    cache.put("a", emb(6, 2.0))
    assert len(cache) == 1
    assert cache.num_tokens == 6
    got = cache.get("a")
    assert got is not None
    assert got.shape[0] == 6

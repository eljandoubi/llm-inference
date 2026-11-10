from collections.abc import Sequence

import pytest

from vlmserve.kv.manager import KVCacheManager
from vlmserve.request import ImageSegment, Request

IMG = 99


def req(
    rid: str, tokens: Sequence[int], segments: list[ImageSegment] | None = None
) -> Request:
    return Request(rid, list(tokens), segments=segments or [])


def step(mgr: KVCacheManager, r: Request, n: int) -> None:
    """What the scheduler and the engine do around one forward pass."""
    assert mgr.allocate(r, r.num_computed_tokens + n)
    r.num_computed_tokens += n
    mgr.commit(r)


def admit(mgr: KVCacheManager, r: Request) -> list[int]:
    """Admission with prefix hits; returns the hit blocks."""
    hits = mgr.match_prefix(r)
    assert mgr.allocate(r, r.num_tokens, hits)
    r.num_computed_tokens = len(hits) * mgr.block_size
    return hits


def test_table_grows_with_the_tokens() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    r = req("a", range(10))
    step(mgr, r, 10)
    assert len(r.block_ids) == 3
    r.append_token(0)  # token 11 still fits in the third block
    step(mgr, r, 1)
    assert len(r.block_ids) == 3
    r.append_token(0)
    r.append_token(0)
    step(mgr, r, 2)  # 13 tokens
    assert len(r.block_ids) == 4


def test_allocate_says_no_without_touching_anything() -> None:
    mgr = KVCacheManager(num_blocks=3, block_size=4)  # two usable blocks
    r = req("a", range(10))  # needs three
    assert not mgr.allocate(r, 10)
    assert r.block_ids == []
    assert mgr.pool.num_free == 2


def test_only_full_blocks_are_offered_to_others() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    r = req("a", range(10))
    step(mgr, r, 10)
    assert mgr.pool.num_cached == 2
    assert r.num_committed_blocks == 2


def test_blocks_are_offered_as_they_get_computed() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    r = req("a", range(10))
    step(mgr, r, 6)
    assert mgr.pool.num_cached == 1
    step(mgr, r, 4)
    assert mgr.pool.num_cached == 2


def test_generated_tokens_fill_and_register_blocks_too() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    r = req("a", range(6))
    step(mgr, r, 6)
    r.append_token(7)
    step(mgr, r, 1)
    assert mgr.pool.num_cached == 1
    r.append_token(8)
    step(mgr, r, 1)  # 8 tokens: block 1 is full
    assert mgr.pool.num_cached == 2


def test_a_second_request_shares_the_prefix_blocks() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    a = req("a", range(10))
    step(mgr, a, 10)
    b = req("b", [*range(8), 50, 51, 52])
    hits = admit(mgr, b)
    assert hits == a.block_ids[:2]
    assert b.block_ids[:2] == hits
    assert b.block_ids[2] not in a.block_ids
    assert b.num_computed_tokens == 8
    assert all(mgr.pool.ref_count(blk) == 2 for blk in hits)


def test_the_last_token_is_never_served_from_the_cache() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    a = req("a", range(8))
    step(mgr, a, 8)
    b = req("b", range(8))  # identical, would be a full hit
    assert len(mgr.match_prefix(b)) == 1


def test_freed_blocks_stay_matchable() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    a = req("a", range(9))
    step(mgr, a, 9)
    mgr.free(a)
    assert a.block_ids == []
    assert mgr.pool.num_free == 9
    b = req("b", range(9))
    hits = admit(mgr, b)
    assert len(hits) == 2
    assert all(mgr.pool.ref_count(blk) == 1 for blk in hits)


def test_revived_blocks_count_against_the_room_left() -> None:
    # 3 usable blocks; the cached prefix is in the free list, and so is the rest
    mgr = KVCacheManager(num_blocks=4, block_size=4)
    a = req("a", range(8))
    step(mgr, a, 8)
    mgr.free(a)
    b = req("b", range(16))  # needs 4 blocks, 2 of them revived
    hits = mgr.match_prefix(b)
    assert len(hits) == 2
    assert not mgr.allocate(b, 16, hits)
    assert b.block_ids == []
    assert mgr.pool.num_free == 3


def test_evicted_prefixes_stop_matching() -> None:
    mgr = KVCacheManager(num_blocks=3, block_size=4)
    a = req("a", range(9))
    step(mgr, a, 8)
    mgr.free(a)
    other = req("c", range(100, 108))
    step(mgr, other, 8)  # takes both blocks, recycling a's
    mgr.free(other)
    assert mgr.match_prefix(req("b", range(9))) == []


def test_free_hands_blocks_back_tail_first() -> None:
    mgr = KVCacheManager(num_blocks=6, block_size=4)
    a = req("a", range(12))
    step(mgr, a, 12)
    assert a.block_ids == [1, 2, 3]
    mgr.free(a)
    b = req("b", range(20))
    assert mgr.allocate(b, 20)
    # never used blocks first, then a's tail, and its head last
    assert b.block_ids == [4, 5, 3, 2, 1]


def test_images_with_other_content_do_not_share_blocks() -> None:
    mgr = KVCacheManager(num_blocks=12, block_size=4)
    tokens = [1, 2, 3, 4, IMG, IMG, IMG, IMG, 5, 6, 7, 8, 9]
    a = req("a", tokens, [ImageSegment(4, 4, "cat")])
    step(mgr, a, len(tokens))
    dog = req("b", tokens, [ImageSegment(4, 4, "dog")])
    assert len(mgr.match_prefix(dog)) == 1  # just the text before the image
    cat = req("c", tokens, [ImageSegment(4, 4, "cat")])
    assert len(mgr.match_prefix(cat)) == 3


def test_prefix_caching_can_be_off() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4, enable_prefix_caching=False)
    a = req("a", range(10))
    step(mgr, a, 10)
    assert mgr.pool.num_cached == 0
    assert mgr.match_prefix(req("b", range(10))) == []


def test_cached_blocks_cannot_be_grafted_onto_a_running_table() -> None:
    mgr = KVCacheManager(num_blocks=10, block_size=4)
    a = req("a", range(9))
    step(mgr, a, 9)
    b = req("b", range(9))
    step(mgr, b, 4)
    with pytest.raises(ValueError, match="new block table"):
        mgr.allocate(b, 9, mgr.match_prefix(a))


def test_usage_tracks_the_blocks_in_use() -> None:
    mgr = KVCacheManager(num_blocks=5, block_size=4)
    assert mgr.usage == 0
    r = req("a", range(8))
    step(mgr, r, 8)
    assert mgr.usage == 0.5

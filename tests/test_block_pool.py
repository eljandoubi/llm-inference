import pytest

from vlmserve.kv.pool import BlockPool


def test_block_zero_is_never_handed_out() -> None:
    pool = BlockPool(4)
    assert pool.num_free == 3
    assert sorted(pool.allocate() for _ in range(3)) == [1, 2, 3]
    assert pool.ref_count(0) == 0


def test_a_pool_needs_room_for_something_besides_the_null_block() -> None:
    with pytest.raises(ValueError, match="null block"):
        BlockPool(1)


def test_running_dry_is_an_error() -> None:
    pool = BlockPool(2)
    pool.allocate()
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        pool.allocate()


def test_release_puts_the_block_back_only_at_zero_references() -> None:
    pool = BlockPool(3)
    block = pool.allocate()
    pool.acquire(block)
    assert pool.ref_count(block) == 2
    pool.release(block)
    assert pool.num_free == 1
    pool.release(block)
    assert pool.num_free == 2


def test_releasing_too_often_is_caught() -> None:
    pool = BlockPool(3)
    block = pool.allocate()
    pool.release(block)
    with pytest.raises(RuntimeError, match="released more often"):
        pool.release(block)


def test_least_recently_freed_goes_first() -> None:
    pool = BlockPool(5)
    a, b, _, d = (pool.allocate() for _ in range(4))
    pool.release(b)
    pool.release(a)
    pool.release(d)
    assert [pool.allocate() for _ in range(3)] == [b, a, d]


def test_blocks_that_never_cached_anything_go_before_recycled_ones() -> None:
    pool = BlockPool(4)
    first = pool.allocate()
    pool.release(first)
    assert pool.allocate() != first


def test_a_registered_block_survives_release_and_can_be_revived() -> None:
    pool = BlockPool(4)
    block = pool.allocate()
    pool.register(block, b"h1")
    pool.release(block)
    assert pool.lookup(b"h1") == block
    assert pool.num_free == 3
    pool.acquire(block)
    assert pool.num_free == 2
    assert pool.ref_count(block) == 1


def test_recycling_a_block_forgets_what_it_cached() -> None:
    pool = BlockPool(2)
    block = pool.allocate()
    pool.register(block, b"h1")
    pool.release(block)
    assert pool.num_cached == 1
    assert pool.allocate() == block
    assert pool.lookup(b"h1") is None
    assert pool.num_cached == 0


def test_first_block_to_register_a_hash_keeps_it() -> None:
    pool = BlockPool(4)
    a, b = pool.allocate(), pool.allocate()
    pool.register(a, b"same")
    pool.register(b, b"same")
    assert pool.lookup(b"same") == a
    # b stays anonymous, so recycling it must not drop a's entry
    pool.release(b)
    assert pool.allocate() != a
    assert pool.lookup(b"same") == a


def test_a_block_registers_one_hash_only() -> None:
    pool = BlockPool(3)
    block = pool.allocate()
    pool.register(block, b"one")
    pool.register(block, b"two")
    assert pool.lookup(b"one") == block
    assert pool.lookup(b"two") is None

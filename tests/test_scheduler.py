from vlmserve.config import EngineConfig
from vlmserve.kv.manager import KVCacheManager
from vlmserve.request import FinishReason, Request, Status
from vlmserve.scheduler import Batch, Scheduler

SAMPLED = 1000


def make(
    num_blocks: int = 64,
    block_size: int = 4,
    max_seqs: int = 4,
    max_tokens: int = 16,
    caching: bool = True,
) -> Scheduler:
    config = EngineConfig(
        "m",
        max_model_len=128,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
        block_size=block_size,
        num_kv_blocks=num_blocks,
        enable_prefix_caching=caching,
    )
    return Scheduler(config, KVCacheManager(num_blocks, block_size, caching))


def tick(sched: Scheduler) -> Batch:
    """One engine step with a fake model that always samples SAMPLED."""
    batch = sched.schedule()
    sched.advance(batch)
    for item in batch.items:
        if item.samples:
            item.request.append_token(SAMPLED)
    return batch


def shape(batch: Batch) -> list[tuple[str, int, int, bool]]:
    return [
        (i.request.request_id, i.start, i.num_tokens, i.samples) for i in batch.items
    ]


def test_nothing_to_do() -> None:
    sched = make()
    assert not sched.has_unfinished
    assert sched.schedule().items == []


def test_a_short_prompt_prefills_in_one_step_then_decodes() -> None:
    sched = make()
    req = Request("a", list(range(6)))
    sched.add(req)
    assert shape(tick(sched)) == [("a", 0, 6, True)]
    assert req.status is Status.RUNNING
    assert shape(tick(sched)) == [("a", 6, 1, True)]
    assert shape(tick(sched)) == [("a", 7, 1, True)]


def test_a_long_prompt_is_cut_into_chunks() -> None:
    sched = make(max_tokens=4)
    sched.add(Request("a", list(range(10))))
    assert shape(tick(sched)) == [("a", 0, 4, False)]
    assert shape(tick(sched)) == [("a", 4, 4, False)]
    assert shape(tick(sched)) == [("a", 8, 2, True)]
    assert shape(tick(sched)) == [("a", 10, 1, True)]


def test_decodes_go_first_and_a_prompt_gets_what_is_left() -> None:
    sched = make(max_tokens=8)
    sched.add(Request("a", list(range(4))))
    tick(sched)
    sched.add(Request("b", list(range(20, 40))))
    assert shape(tick(sched)) == [("a", 4, 1, True), ("b", 0, 7, False)]


def test_requests_are_admitted_in_arrival_order_up_to_max_seqs() -> None:
    sched = make(max_seqs=2)
    for name in "abc":
        sched.add(Request(name, list(range(5))))
    assert [i.request.request_id for i in sched.schedule().items] == ["a", "b"]
    assert [r.request_id for r in sched.waiting] == ["c"]


def test_the_budget_is_shared_by_everybody() -> None:
    sched = make(max_tokens=10)
    for name in "ab":
        sched.add(Request(name, list(range(8))))
    batch = sched.schedule()
    assert shape(batch) == [("a", 0, 8, True), ("b", 0, 2, False)]
    assert batch.num_tokens == 10


def test_finishing_frees_the_blocks_and_the_slot() -> None:
    sched = make(max_seqs=1)
    a = Request("a", list(range(6)))
    b = Request("b", list(range(6)))
    sched.add(a)
    sched.add(b)
    tick(sched)
    free_while_running = sched.kv.pool.num_free
    sched.finish(a, FinishReason.STOP)
    assert a.is_finished
    assert a.block_ids == []
    assert sched.kv.pool.num_free > free_while_running
    assert shape(tick(sched))[0][0] == "b"


def test_abort_works_on_running_and_waiting_requests() -> None:
    sched = make(max_seqs=1)
    a, b = Request("a", [1, 2, 3]), Request("b", [4, 5, 6])
    sched.add(a)
    sched.add(b)
    tick(sched)
    assert sched.abort("b") is b
    assert b.finish_reason == FinishReason.ABORT
    assert sched.abort("a") is a
    assert not sched.has_unfinished
    assert sched.abort("nope") is None
    assert sched.kv.pool.num_free == 63


def test_a_cached_prefix_is_skipped() -> None:
    sched = make()
    a = Request("a", list(range(9)))
    sched.add(a)
    tick(sched)
    b = Request("b", [*range(9), 50, 51])
    sched.add(b)
    batch = tick(sched)
    assert shape(batch)[1] == ("b", 8, 3, True)
    assert b.num_cached_tokens == 8


def test_a_request_without_room_waits_for_blocks() -> None:
    sched = make(num_blocks=5, max_seqs=4, max_tokens=32)  # four usable blocks
    a = Request("a", list(range(12)))  # three blocks
    b = Request("b", list(range(100, 108)))  # two more than there are left
    sched.add(a)
    sched.add(b)
    assert shape(tick(sched)) == [("a", 0, 12, True)]
    assert b.status is Status.WAITING
    sched.finish(a, FinishReason.STOP)
    assert shape(tick(sched)) == [("b", 0, 8, True)]

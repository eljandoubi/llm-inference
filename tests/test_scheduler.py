import torch

from vlmserve.config import EngineConfig
from vlmserve.kv.manager import KVCacheManager
from vlmserve.multimodal.encoder_cache import EncoderCache
from vlmserve.request import FinishReason, ImageSegment, Request, Status
from vlmserve.scheduler import Batch, Scheduler

SAMPLED = 1000
IMG = 77


def make(
    num_blocks: int = 64,
    block_size: int = 4,
    max_seqs: int = 4,
    max_tokens: int = 16,
    caching: bool = True,
    encode_budget: int = 4096,
    cache: EncoderCache | None = None,
) -> Scheduler:
    config = EngineConfig(
        "m",
        max_model_len=128,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
        block_size=block_size,
        num_kv_blocks=num_blocks,
        enable_prefix_caching=caching,
        max_encoder_tokens_per_step=encode_budget,
    )
    return Scheduler(config, KVCacheManager(num_blocks, block_size, caching), cache)


def with_image(rid: str, before: int, image: int, key: str, after: int = 0) -> Request:
    tokens = [1] * before + [IMG] * image + [2] * after
    return Request(rid, tokens, segments=[ImageSegment(before, image, key)])


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


def two_requests_about_to_run_dry() -> tuple[Scheduler, Request, Request]:
    """Four usable blocks, two requests holding two each, both needing a third."""
    sched = make(num_blocks=5, max_seqs=4)
    a, b = Request("a", [1, 2, 3, 4]), Request("b", [5, 6, 7, 8])
    sched.add(a)
    sched.add(b)
    for _ in range(5):
        tick(sched)
    assert sched.kv.pool.num_free == 0
    assert a.num_tokens == b.num_tokens == 9
    return sched, a, b


def test_the_newest_request_is_preempted_when_blocks_run_out() -> None:
    sched, a, b = two_requests_about_to_run_dry()
    assert shape(tick(sched)) == [("a", 8, 1, True)]
    assert b.status is Status.WAITING
    assert b.num_preemptions == 1
    assert b.num_computed_tokens == 0
    assert b.block_ids == []
    assert a.num_preemptions == 0
    assert sched.running == [a]
    assert list(sched.waiting) == [b]


def test_a_preempted_request_recomputes_what_it_generated() -> None:
    sched, a, b = two_requests_about_to_run_dry()
    tick(sched)
    sched.finish(a, FinishReason.STOP)
    # its prompt block survived in the cache, the rest is computed again
    assert shape(tick(sched)) == [("b", 4, 5, True)]
    assert b.status is Status.RUNNING
    assert b.num_cached_tokens == 4


def test_a_preempted_request_goes_before_newcomers() -> None:
    sched, a, _ = two_requests_about_to_run_dry()
    sched.add(Request("c", [9, 9, 9, 9]))
    tick(sched)
    assert [r.request_id for r in sched.waiting] == ["b", "c"]
    assert sched.running == [a]


def test_nobody_new_is_admitted_in_a_step_that_preempted() -> None:
    sched, _, b = two_requests_about_to_run_dry()
    sched.add(Request("c", [9, 9, 9, 9]))
    batch = tick(sched)
    assert [i.request.request_id for i in batch.items] == ["a"]
    assert sched.kv.pool.num_free > 0  # c would have fit, it just isn't tried
    assert b.status is Status.WAITING


def test_a_request_that_cannot_fit_even_alone_preempts_itself() -> None:
    sched = make(num_blocks=3)  # two usable blocks, eight tokens
    a = Request("a", list(range(8)))
    sched.add(a)
    tick(sched)
    assert tick(sched).items == []  # the ninth token needs a third block
    assert a.status is Status.WAITING
    assert a.num_preemptions == 1
    assert list(sched.waiting) == [a]


def keys(batch: Batch) -> list[str]:
    return [seg.key for seg in batch.to_encode]


def test_an_image_in_the_chunk_has_to_be_embedded() -> None:
    sched = make()
    sched.add(with_image("a", before=2, image=4, key="cat", after=2))
    assert keys(sched.schedule()) == ["cat"]


def test_an_image_outside_the_chunk_can_wait() -> None:
    sched = make(max_tokens=4)
    sched.add(with_image("a", before=6, image=4, key="cat"))
    assert keys(sched.schedule()) == []  # the first chunk is all text


def test_images_that_are_ready_are_not_embedded_again() -> None:
    cache = EncoderCache(capacity_tokens=100)
    cache.put("cat", torch.zeros(4, 8))
    sched = make(cache=cache)
    sched.add(with_image("a", before=2, image=4, key="cat"))
    assert keys(sched.schedule()) == []

    own = with_image("b", before=2, image=4, key="dog")
    own.embeddings["dog"] = torch.zeros(4, 8)
    sched.add(own)
    assert keys(sched.schedule()) == []


def test_one_image_shared_by_two_requests_is_embedded_once() -> None:
    sched = make()
    sched.add(with_image("a", before=0, image=4, key="cat"))
    sched.add(with_image("b", before=1, image=4, key="cat"))
    batch = sched.schedule()
    assert len(batch.items) == 2
    assert keys(batch) == ["cat"]


def test_a_chunk_stops_before_an_image_that_does_not_fit_the_budget() -> None:
    sched = make(encode_budget=4)
    sched.add(with_image("a", before=0, image=4, key="x"))
    sched.add(with_image("b", before=2, image=4, key="y"))
    batch = tick(sched)
    assert shape(batch) == [("a", 0, 4, True), ("b", 0, 2, False)]
    assert keys(batch) == ["x"]
    batch = tick(sched)  # the budget starts over
    assert shape(batch) == [("a", 4, 1, True), ("b", 2, 4, True)]
    assert keys(batch) == ["y"]


def test_a_request_whose_first_token_is_an_unaffordable_image_waits() -> None:
    sched = make(encode_budget=4)
    sched.add(with_image("a", before=0, image=4, key="x"))
    b = with_image("b", before=0, image=4, key="y")
    sched.add(b)
    batch = tick(sched)
    assert shape(batch) == [("a", 0, 4, True)]
    assert b.status is Status.WAITING
    assert shape(tick(sched)) == [("a", 4, 1, True), ("b", 0, 4, True)]


def test_the_first_image_of_a_step_goes_even_if_it_is_over_the_budget() -> None:
    sched = make(encode_budget=2)
    sched.add(with_image("a", before=1, image=6, key="big"))
    batch = sched.schedule()
    assert keys(batch) == ["big"]
    assert batch.items[0].num_tokens == 7


def test_a_cache_hit_that_ends_inside_an_image_needs_the_whole_image() -> None:
    sched = make()
    sched.add(with_image("a", before=2, image=8, key="cat", after=2))
    tick(sched)
    b = with_image("b", before=2, image=8, key="cat", after=2)
    sched.add(b)
    batch = sched.schedule()
    assert shape(batch)[-1] == ("b", 8, 4, True)  # blocks 0-1 came from the cache
    assert [seg.length for seg in batch.to_encode] == [8]


def test_embeddings_are_dropped_once_the_image_is_behind_the_request() -> None:
    sched = make(max_tokens=4)
    req = with_image("a", before=2, image=4, key="cat", after=2)
    sched.add(req)
    req.embeddings["cat"] = torch.zeros(4, 8)
    tick(sched)  # tokens 0-3: the image is only half done
    assert "cat" in req.embeddings
    tick(sched)  # tokens 4-7: now it is behind us
    assert req.embeddings == {}


def test_an_image_used_twice_stays_until_its_last_use() -> None:
    sched = make(max_tokens=8)
    tokens = [1] * 2 + [IMG] * 4 + [2] * 2 + [IMG] * 4
    req = Request(
        "a",
        tokens,
        segments=[ImageSegment(2, 4, "cat"), ImageSegment(8, 4, "cat")],
    )
    sched.add(req)
    req.embeddings["cat"] = torch.zeros(4, 8)
    tick(sched)  # tokens 0-7, the second copy is still ahead
    assert "cat" in req.embeddings
    tick(sched)
    assert req.embeddings == {}

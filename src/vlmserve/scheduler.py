from collections import deque
from dataclasses import dataclass, field

from vlmserve.config import EngineConfig
from vlmserve.kv.manager import KVCacheManager
from vlmserve.request import FinishReason, ImageSegment, Request, Status


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request: Request
    start: int  # first token to run, everything before it already has its KV
    num_tokens: int
    samples: bool  # the chunk ends at the last token, so it yields the next one


@dataclass(frozen=True, slots=True)
class Batch:
    items: list[ScheduledRequest]
    to_encode: list[ImageSegment] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(item.num_tokens for item in self.items)


class Scheduler:
    """Continuous batching: every step packs decodes and prefill chunks together."""

    def __init__(self, config: EngineConfig, kv: KVCacheManager) -> None:
        self.config = config
        self.kv = kv
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # oldest first

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def schedule(self) -> Batch:
        # one budget for prefill chunks and decode tokens alike, so a long prompt
        # never holds up the decodes for more than a step's worth of tokens
        budget = self.config.max_num_batched_tokens
        items: list[ScheduledRequest] = []
        preempted = False

        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(self._remaining(req), budget)
            while not self.kv.allocate(req, req.num_computed_tokens + n):
                # out of blocks: the newest request hands its blocks back and starts
                # over later, so an older one never loses its place to a younger one
                victim = self.running.pop()
                self._preempt(victim)
                preempted = True
                if victim is req:
                    break
            if req.status is not Status.RUNNING:
                break
            items.append(self._item(req, req.num_computed_tokens, n))
            budget -= n
            i += 1

        # a step that had to preempt is not the moment to take on new work
        while (
            not preempted
            and self.waiting
            and budget > 0
            and len(self.running) < self.config.max_num_seqs
        ):
            req = self.waiting[0]
            hits = self.kv.match_prefix(req)
            start = len(hits) * self.config.block_size
            n = min(req.num_tokens - start, budget)
            if not self.kv.allocate(req, start + n, hits):
                break  # first come, first served: nobody jumps the queue
            self.waiting.popleft()
            req.status = Status.RUNNING
            req.num_computed_tokens = start
            req.num_cached_tokens = max(req.num_cached_tokens, start)
            self.running.append(req)
            items.append(self._item(req, start, n))
            budget -= n

        return Batch(items)

    def advance(self, batch: Batch) -> None:
        """Book the forward pass: the chunks now have their KV in the cache."""
        for item in batch.items:
            req = item.request
            req.num_computed_tokens = item.start + item.num_tokens
            self.kv.commit(req)

    def finish(self, req: Request, reason: FinishReason) -> None:
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        req.finish(reason)
        self.kv.free(req)

    def abort(self, request_id: str) -> Request | None:
        for req in (*self.running, *self.waiting):
            if req.request_id == request_id:
                self.finish(req, FinishReason.ABORT)
                return req
        return None

    def _preempt(self, victim: Request) -> None:
        self.kv.free(victim)
        victim.preempt()
        self.waiting.appendleft(victim)  # it goes before anything that arrived later

    @staticmethod
    def _remaining(req: Request) -> int:
        remaining = req.num_tokens - req.num_computed_tokens
        if remaining <= 0:
            raise RuntimeError(f"request {req.request_id} has nothing left to compute")
        return remaining

    @staticmethod
    def _item(req: Request, start: int, n: int) -> ScheduledRequest:
        return ScheduledRequest(req, start, n, samples=start + n >= req.num_tokens)

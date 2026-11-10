from collections import deque
from dataclasses import dataclass, field

from vlmserve.config import EngineConfig
from vlmserve.kv.manager import KVCacheManager
from vlmserve.multimodal.encoder_cache import EncoderCache
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


class _EncodeBudget:
    """Image tokens we may embed in one step, and the images queued for it so far."""

    def __init__(self, limit: int, cache: EncoderCache | None) -> None:
        self.limit = limit
        self.cache = cache
        self.queued: dict[str, ImageSegment] = {}
        self.spent = 0

    def fit(self, req: Request, start: int, n: int) -> tuple[int, list[ImageSegment]]:
        """Cut the chunk [start, start + n) short before an image we can't afford yet.

        Returns the new length and the images that have to be embedded for it.
        """
        end = start + n
        wanted: list[ImageSegment] = []
        cost = 0
        for seg in req.segments:
            if seg.end <= start:
                continue
            if seg.offset >= end:
                break
            if self._have(req, seg) or any(w.key == seg.key for w in wanted):
                continue
            # the first image of a step always goes, or a big one would never run
            if self.spent + cost > 0 and self.spent + cost + seg.length > self.limit:
                return max(seg.offset - start, 0), wanted
            wanted.append(seg)
            cost += seg.length
        return n, wanted

    def queue(self, segments: list[ImageSegment]) -> None:
        for seg in segments:
            self.queued[seg.key] = seg
            self.spent += seg.length

    def _have(self, req: Request, seg: ImageSegment) -> bool:
        return (
            seg.key in req.embeddings
            or seg.key in self.queued
            or (self.cache is not None and seg.key in self.cache)
        )


class Scheduler:
    """Continuous batching: every step packs decodes and prefill chunks together."""

    def __init__(
        self,
        config: EngineConfig,
        kv: KVCacheManager,
        encoder_cache: EncoderCache | None = None,
    ) -> None:
        self.config = config
        self.kv = kv
        self.encoder_cache = encoder_cache
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
        encode = _EncodeBudget(
            self.config.max_encoder_tokens_per_step, self.encoder_cache
        )
        items: list[ScheduledRequest] = []
        preempted = False

        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(self._remaining(req), budget)
            n, wanted = encode.fit(req, req.num_computed_tokens, n)
            if n == 0:
                i += 1  # its next image has to wait for a step with room to embed it
                continue
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
            encode.queue(wanted)
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
            n, wanted = encode.fit(req, start, n)
            if n == 0 or not self.kv.allocate(req, start + n, hits):
                break  # first come, first served: nobody jumps the queue
            self.waiting.popleft()
            req.status = Status.RUNNING
            req.num_computed_tokens = start
            req.num_cached_tokens = max(req.num_cached_tokens, start)
            self.running.append(req)
            encode.queue(wanted)
            items.append(self._item(req, start, n))
            budget -= n

        return Batch(items, list(encode.queued.values()))

    def advance(self, batch: Batch) -> None:
        """Book the forward pass: the chunks now have their KV in the cache."""
        for item in batch.items:
            req = item.request
            req.num_computed_tokens = item.start + item.num_tokens
            self.kv.commit(req)
            self._drop_used_embeddings(req)

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

    @staticmethod
    def _drop_used_embeddings(req: Request) -> None:
        """Embeddings are dead weight once the whole image has been computed."""
        done = req.num_computed_tokens
        for key in {s.key for s in req.segments if s.end <= done}:
            if all(s.end <= done for s in req.segments if s.key == key):
                req.embeddings.pop(key, None)

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

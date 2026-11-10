from collections.abc import Collection, Sequence
from dataclasses import dataclass

from vlmserve.config import EngineConfig
from vlmserve.detokenizer import Detokenizer, StopFilter, Tokenizer
from vlmserve.device import memory_info
from vlmserve.kv.manager import KVCacheManager
from vlmserve.model.decoder import Decoder
from vlmserve.multimodal.encoder_cache import EncoderCache
from vlmserve.request import FinishReason, ImageSegment, Request
from vlmserve.runner import ModelRunner, VisionEncoder
from vlmserve.sampler import Sampler
from vlmserve.sampling import SamplingParams
from vlmserve.scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: str
    text: str  # only what is new since the last output, not the whole answer
    finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_generated_tokens: int
    num_cached_tokens: int  # prompt tokens that came out of the prefix cache


@dataclass(slots=True)
class EngineStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    cached_prompt_tokens: int = 0
    finished_requests: int = 0


@dataclass(slots=True)
class _State:
    detokenizer: Detokenizer
    stop_filter: StopFilter | None


class Engine:
    """Ties the scheduler, the cache and the model runner together.

    Not thread safe: whoever owns the engine calls step() from a single thread.
    """

    def __init__(
        self,
        config: EngineConfig,
        decoder: Decoder,
        vision: VisionEncoder | None,
        tokenizer: Tokenizer,
        eos_token_ids: Collection[int],
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.eos_token_ids = frozenset(eos_token_ids)
        self.num_blocks = self._allocate_cache(decoder)
        self.kv = KVCacheManager(
            self.num_blocks, config.block_size, config.enable_prefix_caching
        )
        self.encoder_cache = (
            EncoderCache(config.encoder_cache_tokens)
            if vision is not None and config.encoder_cache_tokens
            else None
        )
        self.scheduler = Scheduler(config, self.kv, self.encoder_cache)
        self.runner = ModelRunner(
            decoder,
            vision,
            config.block_size,
            Sampler(decoder.device, config.seed),
            self.encoder_cache,
        )
        self.stats = EngineStats()
        self._states: dict[str, _State] = {}

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
        segments: Sequence[ImageSegment] = (),
    ) -> None:
        if request_id in self._states:
            raise ValueError(f"request {request_id!r} is already running")
        if len(prompt_token_ids) >= self.config.max_model_len:
            raise ValueError(
                f"the prompt has {len(prompt_token_ids)} tokens, but max_model_len "
                f"is {self.config.max_model_len} and the answer needs room too"
            )
        req = Request(request_id, list(prompt_token_ids), params, list(segments))
        stop_filter = StopFilter(params.stop) if params.stop else None
        self._states[request_id] = _State(Detokenizer(self.tokenizer), stop_filter)
        self.scheduler.add(req)

    def abort(self, request_id: str) -> RequestOutput | None:
        req = self.scheduler.abort(request_id)
        self._states.pop(request_id, None)
        return None if req is None else self._output(req, "")

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished

    def step(self) -> list[RequestOutput]:
        batch = self.scheduler.schedule()
        if not batch.items:
            return []
        sampled = iter(self.runner.execute(batch))
        self.scheduler.advance(batch)
        return [
            output
            for item in batch.items
            if item.samples and (output := self._accept(item.request, next(sampled)))
        ]

    def _accept(self, req: Request, token: int) -> RequestOutput | None:
        """Take a sampled token: text, stop conditions, and the end of the request."""
        state = self._states[req.request_id]
        params = req.params
        req.append_token(token)
        self.stats.generated_tokens += 1

        reason: FinishReason | None = None
        text = ""
        stop_string_hit = False
        ended_by_token = token in params.stop_token_ids or (
            token in self.eos_token_ids and not params.ignore_eos
        )
        if ended_by_token:
            reason = FinishReason.STOP
        else:
            text = state.detokenizer.push(token)
            if state.stop_filter is not None:
                text, stop_string_hit = state.stop_filter.push(text)
                if stop_string_hit:
                    reason = FinishReason.STOP
        if reason is None and (
            len(req.output_token_ids) >= params.max_tokens
            or req.num_tokens >= self.config.max_model_len
        ):
            reason = FinishReason.LENGTH

        if reason is None:
            return self._output(req, text) if text else None
        if not stop_string_hit:
            text += self._drain(
                state
            )  # nothing is held back for a stop string any more
        self.scheduler.finish(req, reason)
        self._states.pop(req.request_id)
        self.stats.finished_requests += 1
        self.stats.prompt_tokens += req.num_prompt_tokens
        self.stats.cached_prompt_tokens += req.num_cached_tokens
        return self._output(req, text)

    @staticmethod
    def _drain(state: _State) -> str:
        """The text still held back when a request ends without a stop string."""
        rest = state.detokenizer.flush()
        if state.stop_filter is None:
            return rest
        text, _ = state.stop_filter.push(rest)
        return text + state.stop_filter.flush()

    @staticmethod
    def _output(req: Request, text: str) -> RequestOutput:
        return RequestOutput(
            request_id=req.request_id,
            text=text,
            finished=req.is_finished,
            finish_reason=req.finish_reason,
            num_prompt_tokens=req.num_prompt_tokens,
            num_generated_tokens=len(req.output_token_ids),
            num_cached_tokens=req.num_cached_tokens,
        )

    def _allocate_cache(self, decoder: Decoder) -> int:
        cfg = self.config
        per_block = decoder.config.kv_block_bytes(cfg.block_size, decoder.dtype)
        num_blocks = cfg.plan_num_blocks(per_block, memory_info(decoder.device).free)
        cfg.check_cache_size(num_blocks)
        decoder.allocate_kv_cache(num_blocks, cfg.block_size)
        return num_blocks

import enum
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from vlmserve.sampling import SamplingParams


class FinishReason(enum.StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


class Status(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


@dataclass(frozen=True, slots=True, eq=False)
class ImageSegment:
    """A run of placeholder tokens in the prompt that the vision tower fills in.

    Segments with the same key hold the same embeddings, the key is a hash of the
    image content (and of how it was preprocessed).
    """

    offset: int
    length: int
    key: str
    inputs: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(eq=False)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    segments: list[ImageSegment] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.monotonic)

    # prompt followed by everything sampled so far, so a decode step and a prefill
    # chunk look the same: compute whatever sits past num_computed_tokens
    token_ids: list[int] = field(init=False)
    status: Status = field(init=False, default=Status.WAITING)
    finish_reason: FinishReason | None = field(init=False, default=None)
    num_computed_tokens: int = field(init=False, default=0)
    num_cached_tokens: int = field(init=False, default=0)
    num_preemptions: int = field(init=False, default=0)
    block_ids: list[int] = field(init=False, default_factory=list)
    block_hashes: list[bytes] = field(init=False, default_factory=list)
    embeddings: dict[str, torch.Tensor] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("empty prompt")
        self.token_ids = list(self.prompt_token_ids)
        previous_end = 0
        for seg in self.segments:
            if seg.length < 1 or seg.offset < previous_end:
                raise ValueError(f"image segment at {seg.offset} is empty or overlaps")
            if seg.end > len(self.prompt_token_ids):
                raise ValueError(f"image segment at {seg.offset} runs past the prompt")
            previous_end = seg.end

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def is_finished(self) -> bool:
        return self.status is Status.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def finish(self, reason: FinishReason) -> None:
        self.status = Status.FINISHED
        self.finish_reason = reason

    def preempt(self) -> None:
        """Back to the queue; the KV cache is gone, so everything is recomputed."""
        self.status = Status.WAITING
        self.num_computed_tokens = 0
        self.num_preemptions += 1

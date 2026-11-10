import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0  # 0 means greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 turns it off
    repetition_penalty: float = 1.0  # 1 turns it off
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )
        if any(not s for s in self.stop):
            raise ValueError("stop strings can't be empty")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

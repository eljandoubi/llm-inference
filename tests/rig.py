from collections import defaultdict
from collections.abc import Collection
from dataclasses import replace
from typing import Any

import torch

from tests.dense import dense_greedy, random_weights
from tests.models import CharTokenizer, TinyVision, build_decoder, tiny_config
from vlmserve.config import EngineConfig
from vlmserve.engine import Engine, RequestOutput

TOK = CharTokenizer()


class Rig:
    """An engine around the tiny model, and the dense weights to check it against."""

    def __init__(
        self,
        device: torch.device,
        eos: Collection[int] = (),
        vision: bool = True,
        **overrides: Any,
    ) -> None:
        self.model_config = tiny_config()
        self.weights = random_weights(self.model_config, seed=3)
        decoder = build_decoder(self.model_config, self.weights, device)
        self.vision = TinyVision(self.model_config.hidden_size, device)
        config = replace(
            EngineConfig(
                "tiny",
                max_model_len=64,
                max_num_seqs=4,
                max_num_batched_tokens=32,
                block_size=4,
                num_kv_blocks=48,
            ),
            **overrides,
        )
        self.engine = Engine(config, decoder, self.vision if vision else None, TOK, eos)

    def run(self, limit: int = 500) -> dict[str, list[RequestOutput]]:
        """Step until nothing is left, and collect the outputs by request."""
        seen: dict[str, list[RequestOutput]] = defaultdict(list)
        for _ in range(limit):
            if not self.engine.has_unfinished:
                return seen
            for output in self.engine.step():
                seen[output.request_id].append(output)
        raise AssertionError("the engine did not finish")

    def text(self, tokens: list[int]) -> str:
        return TOK.decode(tokens, skip_special_tokens=True)

    def greedy(self, prompt: list[int], steps: int) -> list[int]:
        return dense_greedy(self.model_config, self.weights, prompt, steps)


def joined(outputs: list[RequestOutput]) -> str:
    return "".join(o.text for o in outputs)

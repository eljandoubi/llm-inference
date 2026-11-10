from typing import Any

import pytest
import torch

from tests.dense import dense_logits, random_weights
from tests.models import PagedRun, build_decoder, tiny_config
from vlmserve.model.config import DecoderConfig

VARIANTS: dict[str, dict[str, Any]] = {
    "gqa": {},
    "mha": {"num_kv_heads": 4},
    "mqa": {"num_kv_heads": 1},
    "qwen-like": {"qkv_bias": True, "tie_word_embeddings": True},
}


class Rig:
    """A paged decoder next to the dense reference, built from the same weights."""

    def __init__(
        self, device: torch.device, block_size: int = 4, **overrides: Any
    ) -> None:
        self.config: DecoderConfig = tiny_config(**overrides)
        self.weights = random_weights(self.config, seed=3)
        decoder = build_decoder(self.config, self.weights, device)
        self.run = PagedRun(decoder, block_size=block_size)

    def step(self, work: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        """One paged step, checked against the dense logits of those positions."""
        got = self.run.step(work)
        for name, new in work.items():
            want = dense_logits(self.config, self.weights, self.run.tokens[name])
            torch.testing.assert_close(
                got[name], want[-len(new) :], rtol=1e-4, atol=1e-4
            )
        return got

    def greedy(self, name: str, steps: int) -> None:
        for _ in range(steps):
            last = dense_logits(self.config, self.weights, self.run.tokens[name])[-1]
            self.step({name: [int(last.argmax())]})


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=VARIANTS.keys())
def test_prompt_in_one_step_then_decoding(
    device: torch.device, variant: dict[str, Any]
) -> None:
    rig = Rig(device, **variant)
    rig.step({"a": [3, 1, 4, 1, 5, 9, 2]})
    rig.greedy("a", 7)  # crosses two block boundaries


@pytest.mark.parametrize("variant", VARIANTS.values(), ids=VARIANTS.keys())
def test_chunked_prefill_gives_the_same_logits(
    device: torch.device, variant: dict[str, Any]
) -> None:
    rig = Rig(device, **variant)
    rig.step({"a": [3, 1, 4]})
    rig.step({"a": [1, 5]})
    rig.step({"a": [9, 2, 6]})
    rig.greedy("a", 3)


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_sequences_at_different_stages_share_a_step(
    device: torch.device, block_size: int
) -> None:
    rig = Rig(device, block_size=block_size)
    rig.step({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8]})  # two prompts
    rig.step({"a": [9], "b": [10], "c": [11, 12, 13, 14, 15, 16]})  # decodes + a prompt
    rig.step({"a": [17], "c": [18, 19]})  # b sits this one out, c continues
    rig.step({"b": [20, 21, 22, 23], "a": [24], "c": [25]})  # b comes back with a chunk
    rig.step({"a": [26], "b": [27], "c": [28]})  # all decoding


def test_the_same_tokens_give_the_same_logits_whoever_shares_the_batch(
    device: torch.device,
) -> None:
    alone = Rig(device)
    crowded = Rig(device)
    prompt = [5, 4, 3, 2, 1, 0]
    solo = alone.step({"a": prompt})["a"]
    together = crowded.step({"x": [9, 9, 9], "a": prompt, "y": [8]})["a"]
    torch.testing.assert_close(solo, together, rtol=1e-4, atol=1e-4)

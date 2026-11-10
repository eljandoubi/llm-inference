from collections.abc import Sequence

import numpy as np
import torch

from vlmserve.sampling import SamplingParams

# only this many of the latest tokens count for the repetition penalty, which keeps
# the cost of a step independent of how long the conversation is
REPETITION_WINDOW = 256


def apply_repetition_penalty(
    logits: torch.Tensor,
    params: Sequence[SamplingParams],
    history: Sequence[Sequence[int]],
) -> None:
    """Push down, in place, the logits of tokens a row has seen recently.

    Positive logits are divided by the penalty and negative ones multiplied, once
    per distinct token however often it showed up.
    """
    rows = [i for i, p in enumerate(params) if p.repetition_penalty != 1.0]
    if not rows:
        return
    windows = [list(history[i][-REPETITION_WINDOW:]) for i in rows]
    width = max(len(w) for w in windows)
    # rows differ in length: pad by repeating a token, a duplicate changes nothing
    ids = np.array([w + [w[0]] * (width - len(w)) for w in windows], dtype=np.int64)
    device = logits.device
    row_index = torch.tensor(rows, device=device)[:, None]
    token_index = torch.from_numpy(ids).to(device)
    penalty = torch.tensor([params[i].repetition_penalty for i in rows], device=device)
    seen = logits[row_index, token_index]
    penalty = penalty[:, None].to(seen.dtype)
    logits[row_index, token_index] = torch.where(
        seen > 0, seen / penalty, seen * penalty
    )


class Sampler:
    """Picks the next token for every row of a batch in a handful of vectorized ops."""

    def __init__(self, device: torch.device, seed: int = 0) -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __call__(
        self,
        logits: torch.Tensor,
        params: Sequence[SamplingParams],
        history: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """logits [B, vocab] -> token ids [B]. history holds each row's token ids."""
        logits = logits.float()
        apply_repetition_penalty(logits, params, history)
        best = logits.argmax(-1)
        sampling = [p for p in params if not p.greedy]
        if not sampling:
            return best

        device = logits.device
        settings = torch.tensor(
            [[p.temperature, p.top_p, p.top_k] for p in params],
            dtype=torch.float32,
            device=device,
        )
        temperature, top_p, top_k = settings.unbind(1)
        greedy = temperature == 0
        scaled = (
            logits
            / torch.where(greedy, torch.ones_like(temperature), temperature)[:, None]
        )

        cut_k = any(p.top_k > 0 for p in sampling)
        cut_p = any(p.top_p < 1 for p in sampling)
        if cut_k or cut_p:
            vocab = scaled.shape[-1]
            # a row without top_k still needs the whole vocabulary as candidates
            wide = any(p.top_k == 0 for p in sampling)
            candidates = vocab if wide else min(max(p.top_k for p in sampling), vocab)
            if candidates == vocab:
                values, indices = scaled.sort(dim=-1, descending=True)
            else:
                values, indices = scaled.topk(candidates, dim=-1)
            rank = torch.arange(candidates, device=device)[None, :]
            keep_k = torch.where(top_k > 0, top_k, torch.full_like(top_k, candidates))
            values = values.masked_fill(rank >= keep_k[:, None], float("-inf"))
            probs = values.softmax(-1)
            # nucleus: drop a candidate once the ones before it already add up to top_p
            before = probs.cumsum(-1) - probs
            drop = (before >= top_p[:, None]) & (top_p < 1)[:, None]
            probs = values.masked_fill(drop, float("-inf")).softmax(-1)
            sampled = indices.gather(-1, self._draw(probs)[:, None]).squeeze(-1)
        else:
            sampled = self._draw(scaled.softmax(-1))
        return torch.where(greedy, best, sampled)

    def _draw(self, probs: torch.Tensor) -> torch.Tensor:
        # argmax(p / e) with e ~ Exp(1) is a draw from p, and needs no cumulative sums
        noise = torch.empty_like(probs).exponential_(generator=self.generator)
        return (probs / noise.clamp_min(1e-20)).argmax(-1)

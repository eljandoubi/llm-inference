import pytest
import torch

from vlmserve.sampler import REPETITION_WINDOW, Sampler, apply_repetition_penalty
from vlmserve.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0)


def draw(
    device: torch.device,
    logits: list[float],
    params: SamplingParams,
    n: int = 4000,
    seed: int = 0,
) -> torch.Tensor:
    """n independent draws from one distribution, made as a single batch."""
    batch = torch.tensor([logits] * n, device=device)
    sampler = Sampler(device, seed=seed)
    return sampler(batch, [params] * n, [[0]] * n).cpu()


def frequencies(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(tokens, minlength=vocab).float() / len(tokens)


def test_greedy_takes_the_best_token(device: torch.device) -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3], [5.0, 1.0, 0.0]], device=device)
    out = Sampler(device)(logits, [GREEDY, GREEDY], [[0], [0]])
    assert out.tolist() == [1, 0]


def test_the_penalty_divides_positive_and_multiplies_negative_logits() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.5, 4.0]])
    apply_repetition_penalty(
        logits, [SamplingParams(repetition_penalty=2.0)], [[0, 1, 0]]
    )
    assert logits.tolist() == [[1.0, -2.0, 0.5, 4.0]]  # once per distinct token


def test_rows_without_a_penalty_are_left_alone() -> None:
    logits = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
    params = [SamplingParams(), SamplingParams(repetition_penalty=2.0)]
    apply_repetition_penalty(logits, params, [[1], [1]])
    assert logits.tolist() == [[2.0, 3.0], [2.0, 1.5]]


def test_a_penalty_can_change_the_greedy_choice(device: torch.device) -> None:
    logits = torch.tensor([[3.0, 2.9]], device=device)
    params = [SamplingParams(temperature=0, repetition_penalty=1.5)]
    assert Sampler(device)(logits, params, [[0, 0]]).tolist() == [1]


def test_histories_of_different_length_are_padded_safely(device: torch.device) -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 2, device=device)
    params = [SamplingParams(repetition_penalty=2.0)] * 2
    apply_repetition_penalty(logits, params, [[0], [1, 2, 3]])
    assert logits.cpu().tolist() == [[0.5, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 0.5]]


def test_only_the_latest_tokens_count_for_the_penalty() -> None:
    logits = torch.ones(1, 5)
    history = [[0] + [1] * REPETITION_WINDOW]  # token 0 fell out of the window
    apply_repetition_penalty(logits, [SamplingParams(repetition_penalty=2.0)], history)
    assert logits.tolist() == [[1.0, 0.5, 1.0, 1.0, 1.0]]


def test_sampling_follows_the_softmax(device: torch.device) -> None:
    tokens = draw(device, [1.0, 0.0, -1.0], SamplingParams(temperature=1.0))
    expected = torch.tensor([1.0, 0.0, -1.0]).softmax(-1)
    assert torch.allclose(frequencies(tokens, 3), expected, atol=0.03)


def test_temperature_sharpens_and_flattens(device: torch.device) -> None:
    cold = frequencies(draw(device, [1.0, 0.0], SamplingParams(temperature=0.2)), 2)
    hot = frequencies(draw(device, [1.0, 0.0], SamplingParams(temperature=5.0)), 2)
    assert cold[0] > 0.98
    assert 0.5 < hot[0] < 0.6


def test_top_k_never_leaves_the_best_k(device: torch.device) -> None:
    tokens = draw(device, [3.0, 2.0, 1.0, 0.0, -1.0], SamplingParams(top_k=2))
    freq = frequencies(tokens, 5)
    assert freq[2:].sum() == 0
    assert freq[0] > freq[1] > 0


def test_top_k_of_one_is_greedy(device: torch.device) -> None:
    assert set(
        draw(device, [0.0, 1.0, 0.9], SamplingParams(top_k=1), n=200).tolist()
    ) == {1}


def test_top_p_keeps_the_smallest_set_that_covers_it(device: torch.device) -> None:
    logits = torch.tensor([0.5, 0.3, 0.15, 0.05]).log().tolist()
    wide = frequencies(draw(device, logits, SamplingParams(top_p=0.7)), 4)
    assert wide[2:].sum() == 0
    assert wide[1] > 0
    narrow = frequencies(draw(device, logits, SamplingParams(top_p=0.5)), 4)
    assert narrow[0] == 1.0


def test_top_p_and_top_k_combine(device: torch.device) -> None:
    logits = torch.tensor([0.4, 0.3, 0.2, 0.1]).log().tolist()
    tokens = draw(device, logits, SamplingParams(top_k=3, top_p=0.95))
    assert frequencies(tokens, 4)[3] == 0  # top-p alone would still keep the 4th


def test_every_row_gets_its_own_settings(device: torch.device) -> None:
    n = 600
    logits = torch.zeros(n, 50, device=device)
    logits[:, 7] = 0.5  # a slight favourite for the greedy rows
    params = [GREEDY] * 200 + [SamplingParams(top_k=1)] * 200 + [SamplingParams()] * 200
    out = Sampler(device, seed=1)(logits, params, [[0]] * n).cpu()
    assert set(out[:200].tolist()) == {7}
    assert set(out[200:400].tolist()) == {7}
    # a row with no top_k must not be cut down to the others' candidates
    assert len(set(out[400:].tolist())) > 15


def test_the_seed_decides(device: torch.device) -> None:
    params = SamplingParams()
    first = draw(device, [0.0] * 20, params, n=64, seed=5)
    again = draw(device, [0.0] * 20, params, n=64, seed=5)
    other = draw(device, [0.0] * 20, params, n=64, seed=6)
    assert first.tolist() == again.tolist()
    assert first.tolist() != other.tolist()


def test_half_precision_logits_are_fine(device: torch.device) -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3]], device=device, dtype=torch.float16)
    assert Sampler(device)(logits, [GREEDY], [[0]]).tolist() == [1]


@pytest.mark.parametrize("vocab", [1, 2])
def test_tiny_vocabularies_work(device: torch.device, vocab: int) -> None:
    out = draw(device, [0.0] * vocab, SamplingParams(top_k=5, top_p=0.9), n=10)
    assert out.max() < vocab

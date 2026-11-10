from typing import Any

import pytest

from vlmserve.sampling import SamplingParams


def test_defaults_are_valid_and_sample() -> None:
    params = SamplingParams()
    assert not params.greedy
    assert params.top_k == 0
    assert params.repetition_penalty == 1.0


def test_zero_temperature_means_greedy() -> None:
    assert SamplingParams(temperature=0).greedy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_p": 0.0},
        {"top_p": 1.01},
        {"top_k": -1},
        {"repetition_penalty": 0.0},
        {"repetition_penalty": float("nan")},
        {"stop": ("ok", "")},
    ],
)
def test_nonsense_is_rejected(kwargs: dict[str, Any]) -> None:
    # the message names the offending field
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        SamplingParams(**kwargs)


def test_params_are_immutable() -> None:
    params = SamplingParams()
    with pytest.raises(AttributeError):
        params.temperature = 0.5  # type: ignore[misc]

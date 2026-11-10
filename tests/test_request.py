import pytest

from vlmserve.request import FinishReason, ImageSegment, Request, Status
from vlmserve.sampling import SamplingParams


def make(prompt_len: int = 6, segments: list[ImageSegment] | None = None) -> Request:
    return Request("r1", list(range(prompt_len)), segments=segments or [])


def test_new_request_waits_with_nothing_computed() -> None:
    req = make()
    assert req.status is Status.WAITING
    assert req.num_computed_tokens == 0
    assert req.num_tokens == req.num_prompt_tokens == 6
    assert req.output_token_ids == []
    assert req.block_ids == []
    assert req.num_committed_blocks == 0
    assert not req.is_finished


def test_appended_tokens_are_the_output() -> None:
    req = make()
    req.append_token(42)
    req.append_token(43)
    assert req.num_tokens == 8
    assert req.num_prompt_tokens == 6
    assert req.output_token_ids == [42, 43]


def test_prompt_list_is_copied() -> None:
    prompt = [1, 2, 3]
    req = Request("r", prompt)
    req.append_token(4)
    assert prompt == [1, 2, 3]
    assert req.prompt_token_ids == [1, 2, 3]


def test_empty_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="empty prompt"):
        Request("r", [])


def test_finish_records_the_reason() -> None:
    req = make()
    req.finish(FinishReason.LENGTH)
    assert req.is_finished
    assert req.finish_reason == "length"


def test_preempt_goes_back_to_waiting_and_recomputes_everything() -> None:
    req = make()
    req.status = Status.RUNNING
    req.num_computed_tokens = 5
    req.append_token(9)
    req.preempt()
    assert req.status is Status.WAITING
    assert req.num_computed_tokens == 0
    assert req.num_preemptions == 1
    assert req.output_token_ids == [9]  # what was generated is kept


def test_segments_must_fit_inside_the_prompt() -> None:
    ok = ImageSegment(offset=2, length=4, key="a")
    assert ok.end == 6
    assert make(6, segments=[ok]).segments == [ok]
    with pytest.raises(ValueError, match="runs past the prompt"):
        make(5, segments=[ok])


@pytest.mark.parametrize(
    "segments",
    [
        [ImageSegment(0, 3, "a"), ImageSegment(2, 2, "b")],  # overlap
        [ImageSegment(3, 2, "a"), ImageSegment(0, 2, "b")],  # out of order
        [ImageSegment(1, 0, "a")],  # empty
    ],
)
def test_bad_segment_layouts_are_refused(segments: list[ImageSegment]) -> None:
    with pytest.raises(ValueError, match="empty or overlaps"):
        make(10, segments=segments)


def test_params_default_to_sampling_defaults() -> None:
    assert make().params == SamplingParams()

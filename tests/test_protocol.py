from typing import Any

import pytest
from pydantic import ValidationError

from vlmserve.server.protocol import (
    ChatCompletionRequest,
    ImageContent,
    TextContent,
)

HI = {"role": "user", "content": "hi"}


def parse(**fields: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"messages": [HI], **fields})


def test_the_smallest_request() -> None:
    request = parse()
    assert request.messages[0].content == "hi"
    assert not request.stream


def test_content_can_mix_text_and_images() -> None:
    content = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": content}]}
    )
    parts = request.messages[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], TextContent)
    assert isinstance(parts[1], ImageContent)
    assert parts[1].image_url.url.startswith("data:image/png")


@pytest.mark.parametrize(
    "bad",
    [
        {"messages": []},
        {"messages": [{"role": "robot", "content": "hi"}]},
        {"messages": [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
    ],
)
def test_malformed_messages_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(bad)


def test_fields_we_do_not_care_about_are_ignored() -> None:
    request = parse(user="u-1", seed=3, store=True, metadata={"a": "b"})
    assert request.messages


@pytest.mark.parametrize(
    "field",
    [
        {"tools": [{"type": "function"}]},
        {"tool_choice": "auto"},
        {"functions": [{"name": "f"}]},
        {"response_format": {"type": "json_object"}},
        {"response_format": "json"},
        {"logprobs": True},
        {"top_logprobs": 3},
        {"logit_bias": {"50": 5}},
        {"frequency_penalty": 0.5},
        {"presence_penalty": -1.0},
        {"n": 2},
    ],
)
def test_what_the_server_cannot_do_is_a_clear_error(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse(**field)


def test_harmless_defaults_of_those_fields_pass() -> None:
    parse(
        n=1,
        logprobs=False,
        frequency_penalty=0,
        presence_penalty=0.0,
        response_format={"type": "text"},
        tools=[],
    )


@pytest.mark.parametrize(
    "field",
    [
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"top_p": 0},
        {"top_p": 1.1},
        {"max_tokens": 0},
        {"max_completion_tokens": -3},
        {"top_k": -1},
        {"repetition_penalty": 0},
        {"stop": [str(i) for i in range(17)]},
    ],
)
def test_out_of_range_values_are_rejected(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse(**field)


def test_sampling_defaults() -> None:
    params = parse().sampling_params(room=100)
    assert params.max_tokens == 100
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == 0
    assert params.repetition_penalty == 1.0
    assert params.stop == ()
    assert params.ignore_eos is False


def test_the_prompt_leaves_only_so_much_room() -> None:
    assert parse(max_tokens=500).sampling_params(room=40).max_tokens == 40
    assert parse(max_tokens=10).sampling_params(room=40).max_tokens == 10


def test_max_completion_tokens_wins_over_max_tokens() -> None:
    assert (
        parse(max_tokens=10, max_completion_tokens=20).sampling_params(50).max_tokens
        == 20
    )


def test_everything_the_client_set_reaches_the_params() -> None:
    params = parse(
        temperature=0, top_p=0.9, top_k=40, repetition_penalty=1.2, stop="END"
    ).sampling_params(room=10)
    assert params.greedy
    assert (params.top_p, params.top_k, params.repetition_penalty) == (0.9, 40, 1.2)
    assert params.stop == ("END",)
    assert parse(stop=["a", "b"]).sampling_params(10).stop == ("a", "b")


def test_ignore_eos_is_passed_on() -> None:
    assert parse(ignore_eos=True).sampling_params(10).ignore_eos is True
    assert parse(ignore_eos=False).sampling_params(10).ignore_eos is False

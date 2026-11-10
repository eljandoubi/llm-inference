import base64
import io
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from httpx2 import Response
from PIL import Image

from tests.frontend import TinyFrontend
from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine
from vlmserve.engine import RequestOutput
from vlmserve.multimodal.images import ImageLimits
from vlmserve.server.app import create_app


@dataclass
class Served:
    client: TestClient
    rig: Rig
    engine: AsyncEngine
    frontend: TinyFrontend


@pytest.fixture
def served() -> Iterator[Served]:
    rig = Rig(torch.device("cpu"))
    engine = AsyncEngine(rig.engine)
    frontend = TinyFrontend()
    app = create_app(engine, frontend, "tiny-vlm", ImageLimits(max_images=2))
    with TestClient(app) as client:
        yield Served(client, rig, engine, frontend)


def test_health_is_ok_while_the_engine_runs(served: Served) -> None:
    reply = served.client.get("/health")
    assert reply.status_code == 200
    assert reply.json() == {"status": "ok"}


def test_health_turns_into_a_503_when_the_engine_is_gone(served: Served) -> None:
    served.engine.stop()
    reply = served.client.get("/health")
    assert reply.status_code == 503
    assert reply.json() == {"status": "unavailable"}


def test_the_model_list_has_the_served_name(served: Served) -> None:
    body = served.client.get("/v1/models").json()
    assert body["object"] == "list"
    (card,) = body["data"]
    assert card["id"] == "tiny-vlm"
    assert card["max_model_len"] == 64


def test_unknown_routes_are_404(served: Served) -> None:
    assert served.client.get("/nope").status_code == 404


URL = "/v1/chat/completions"
HI = {"role": "user", "content": "hi"}
HI_PROMPT = TinyFrontend.encode("user: hi\nassistant: ")


def post(served: Served, **fields: Any) -> Response:
    return served.client.post(URL, json={"messages": [HI], **fields})


def png_url(color: str = "red", size: tuple[int, int] = (16, 16)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_message(url: str) -> dict[str, Any]:
    content = [
        {"type": "text", "text": "look "},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    return {"role": "user", "content": content}


def test_a_plain_completion(served: Served) -> None:
    reply = post(served, temperature=0, max_tokens=6)
    assert reply.status_code == 200
    body = reply.json()
    expected = served.rig.text(served.rig.greedy(HI_PROMPT, 6))
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "tiny-vlm"
    (choice,) = body["choices"]
    assert choice["message"] == {"role": "assistant", "content": expected}
    assert choice["finish_reason"] == "length"
    assert body["usage"] == {
        "prompt_tokens": len(HI_PROMPT),
        "completion_tokens": 6,
        "total_tokens": len(HI_PROMPT) + 6,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_without_max_tokens_the_answer_may_fill_the_context(served: Served) -> None:
    usage = post(served, temperature=0).json()["usage"]
    assert usage["prompt_tokens"] + usage["completion_tokens"] == 64


def test_a_stop_string_ends_the_answer(served: Served) -> None:
    full = served.rig.text(served.rig.greedy(HI_PROMPT, 8))
    stop = full[3:5]
    reply = post(served, temperature=0, max_tokens=8, stop=stop).json()
    (choice,) = reply["choices"]
    assert choice["message"]["content"] == full[: full.find(stop)]
    assert choice["finish_reason"] == "stop"


def test_an_image_goes_through_and_a_repeat_hits_the_cache(served: Served) -> None:
    messages = [image_message(png_url())]
    first = post_messages(served, messages)
    second = post_messages(served, messages)
    assert first.status_code == second.status_code == 200
    one, two = first.json(), second.json()
    prompt_len = len("user: look \nassistant: ") + 4  # one token per character
    assert one["usage"]["prompt_tokens"] == prompt_len
    assert one["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert two["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert one["choices"] == two["choices"]


def post_messages(served: Served, messages: list[dict[str, Any]]) -> Response:
    body = {"messages": messages, "temperature": 0, "max_tokens": 5}
    return served.client.post(URL, json=body)


def test_a_different_image_changes_the_answer_and_the_cache(served: Served) -> None:
    red = post_messages(served, [image_message(png_url("red"))]).json()
    blue = post_messages(served, [image_message(png_url("blue"))]).json()
    cached = blue["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached < red["usage"]["prompt_tokens"] - 4  # not the blocks after the image
    assert served.rig.vision.images == 2


def error(reply: Response) -> dict[str, Any]:
    body = reply.json()
    assert set(body) == {"error"}
    assert body["error"]["type"] in {"invalid_request_error", "server_error"}
    return body["error"]


@pytest.mark.parametrize(
    ("fields", "mentions"),
    [
        ({"tools": [{"type": "function"}]}, "tools"),
        ({"n": 2}, "n"),
        ({"temperature": 5}, "temperature"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "user", "content": [{"type": "audio"}]}]}, "messages"),
    ],
)
def test_bad_requests_get_a_400_in_openai_shape(
    served: Served, fields: dict[str, Any], mentions: str
) -> None:
    reply = post(served, **fields)
    assert reply.status_code == 400
    assert mentions in error(reply)["message"]


def test_a_body_that_is_not_json_is_a_400(served: Served) -> None:
    reply = served.client.post(
        URL, content=b"{nope", headers={"content-type": "application/json"}
    )
    assert reply.status_code == 400
    error(reply)


def test_remote_image_urls_are_refused(served: Served) -> None:
    reply = post_messages(served, [image_message("http://example.com/cat.png")])
    assert reply.status_code == 400
    assert "data: URLs" in error(reply)["message"]


def test_a_broken_image_is_a_400(served: Served) -> None:
    reply = post_messages(served, [image_message("data:image/png;base64,AAAA")])
    assert reply.status_code == 400


def test_too_many_images_is_a_400(served: Served) -> None:
    url = png_url()
    messages = [image_message(url), image_message(url), image_message(url)]
    reply = post_messages(served, messages)
    assert reply.status_code == 400
    assert "at most 2 images" in error(reply)["message"]


def test_an_image_over_the_byte_limit_is_a_413() -> None:
    rig = Rig(torch.device("cpu"))
    app = create_app(
        AsyncEngine(rig.engine), TinyFrontend(), "tiny", ImageLimits(max_bytes=500)
    )
    noise = Image.frombytes("RGB", (40, 40), os.urandom(40 * 40 * 3))
    buffer = io.BytesIO()
    noise.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with TestClient(app) as client:
        reply = client.post(URL, json={"messages": [image_message(url)]})
    assert reply.status_code == 413


def test_a_prompt_longer_than_the_context_is_a_400(served: Served) -> None:
    reply = post(served, messages=[{"role": "user", "content": "x" * 100}])
    assert reply.status_code == 400
    assert error(reply)["code"] == "context_length_exceeded"


def test_a_dead_engine_is_a_503(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> list[RequestOutput]:
        raise RuntimeError("cuda out of memory, say")

    monkeypatch.setattr(served.rig.engine, "step", boom)
    reply = post(served, max_tokens=3)
    assert reply.status_code == 503
    assert error(reply)["type"] == "server_error"
    assert served.client.get("/health").status_code == 503
    assert post(served).status_code == 503


def events(reply_lines: list[str]) -> list[str]:
    return [
        line.removeprefix("data: ") for line in reply_lines if line.startswith("data: ")
    ]


def stream(served: Served, **fields: Any) -> list[str]:
    with served.client.stream(
        "POST", URL, json={"messages": [HI], "stream": True, **fields}
    ) as reply:
        assert reply.status_code == 200
        assert reply.headers["content-type"].startswith("text/event-stream")
        return events(list(reply.iter_lines()))


def test_a_stream_is_role_then_text_then_finish_then_done(served: Served) -> None:
    data = stream(served, temperature=0, max_tokens=6)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(d) for d in data[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[1:-1])
    assert text == served.rig.text(served.rig.greedy(HI_PROMPT, 6))
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1


def test_usage_can_be_asked_for_in_a_stream(served: Served) -> None:
    data = stream(
        served, temperature=0, max_tokens=4, stream_options={"include_usage": True}
    )
    last = json.loads(data[-2])
    assert last["choices"] == []
    assert last["usage"]["completion_tokens"] == 4
    assert json.loads(data[-3])["choices"][0]["finish_reason"] == "length"


def test_a_stream_without_include_usage_has_no_usage(served: Served) -> None:
    data = stream(served, temperature=0, max_tokens=4)
    assert all("usage" not in json.loads(d) for d in data[:-1])


def test_an_engine_failure_in_a_stream_is_reported_in_the_stream(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> list[RequestOutput]:
        raise RuntimeError("boom")

    monkeypatch.setattr(served.rig.engine, "step", boom)
    data = stream(served, max_tokens=3)
    assert "error" in json.loads(data[-1])
    assert "[DONE]" not in data

import asyncio
import json

import torch
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from tests.frontend import TinyFrontend
from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine
from vlmserve.multimodal.images import ImageLimits
from vlmserve.server.app import create_app
from vlmserve.server.limits import BodyLimitMiddleware


class Recorder:
    """A bare asgi app that reads the whole body, and remembers whether it ran."""

    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.called = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def run(
    middleware: BodyLimitMiddleware,
    chunks: list[bytes],
    headers: list[tuple[bytes, bytes]] | None = None,
) -> list[Message]:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": headers or [],
    }
    pending = list(chunks)
    sent: list[Message] = []

    async def receive() -> Message:
        body = pending.pop(0) if pending else b""
        return {"type": "http.request", "body": body, "more_body": bool(pending)}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def status_of(sent: list[Message]) -> int:
    return int(next(m["status"] for m in sent if m["type"] == "http.response.start"))


def test_a_body_under_the_limit_goes_through() -> None:
    app = Recorder()
    sent = run(BodyLimitMiddleware(app, max_bytes=10), [b"12345", b"678"])
    assert status_of(sent) == 200
    assert app.body == b"12345678"


def test_a_declared_length_over_the_limit_never_reaches_the_app() -> None:
    app = Recorder()
    sent = run(BodyLimitMiddleware(app, 10), [b"x"], [(b"content-length", b"11")])
    assert status_of(sent) == 413
    assert not app.called


def test_a_chunked_body_is_counted_as_it_arrives() -> None:
    app = Recorder()
    sent = run(BodyLimitMiddleware(app, 10), [b"123456", b"789012", b"3"])
    assert status_of(sent) == 413
    body = json.loads(
        next(m["body"] for m in sent if m["type"] == "http.response.body")
    )
    assert body["error"]["code"] == "request_too_large"


def test_exactly_the_limit_is_fine() -> None:
    sent = run(
        BodyLimitMiddleware(Recorder(), 10),
        [b"1234567890"],
        [(b"content-length", b"10")],
    )
    assert status_of(sent) == 200


def test_other_scope_types_are_left_alone() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware = BodyLimitMiddleware(app, max_bytes=1)
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"]


def test_the_chat_endpoint_refuses_an_oversized_body() -> None:
    rig = Rig(torch.device("cpu"))
    app = create_app(
        AsyncEngine(rig.engine),
        TinyFrontend(),
        "tiny",
        ImageLimits(),
        max_body_bytes=200,
    )
    with TestClient(app) as client:
        big = {"messages": [{"role": "user", "content": "x" * 500}]}
        reply = client.post("/v1/chat/completions", json=big)
        assert reply.status_code == 413
        small = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        assert client.post("/v1/chat/completions", json=small).status_code == 200

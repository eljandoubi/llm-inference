import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vlmserve.async_engine import AsyncEngine, EngineDeadError
from vlmserve.engine import RequestOutput
from vlmserve.frontend import Frontend
from vlmserve.multimodal.images import ImageLimits
from vlmserve.server.chat import prepare_prompt
from vlmserve.server.errors import error_response, install_error_handlers
from vlmserve.server.limits import BodyLimitMiddleware
from vlmserve.server.metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from vlmserve.server.metrics import render as render_metrics
from vlmserve.server.protocol import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelList,
)
from vlmserve.server.sse import DONE, chunk, sse, usage_chunk, usage_of


async def drain(
    stream: AsyncGenerator[RequestOutput, None],
) -> tuple[str, RequestOutput]:
    """Wait for a whole answer: its text, and the last output with the counts."""
    pieces: list[str] = []
    last: RequestOutput | None = None
    async for output in stream:
        pieces.append(output.text)
        last = output
    if last is None:
        raise EngineDeadError("the engine ended the request without an answer")
    return "".join(pieces), last


async def drain_unless_disconnected(
    stream: AsyncGenerator[RequestOutput, None],
    is_disconnected: Callable[[], Awaitable[bool]],
    poll: float = 0.25,
) -> tuple[str, RequestOutput] | None:
    """drain(), but give up and abort the request if the client goes away first.

    Without this a dropped connection would keep its request running to the end.
    """

    async def wait_for_disconnect() -> None:
        while not await is_disconnected():
            await asyncio.sleep(poll)

    reading = asyncio.ensure_future(drain(stream))
    watching = asyncio.ensure_future(wait_for_disconnect())
    try:
        await asyncio.wait({reading, watching}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:  # we were cancelled ourselves, take the read down with us
        reading.cancel()
        raise
    finally:
        watching.cancel()
    if reading.done():
        return reading.result()
    reading.cancel()  # cancelling the read is what aborts the request
    await asyncio.gather(reading, return_exceptions=True)
    return None


async def server_sent_events(
    stream: AsyncGenerator[RequestOutput, None],
    request_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    yield sse(chunk(request_id, created, model, {"role": "assistant", "content": ""}))
    last: RequestOutput | None = None
    try:
        async for output in stream:
            last = output
            if output.text:
                yield sse(chunk(request_id, created, model, {"content": output.text}))
        if last is not None:
            reason = last.finish_reason
            yield sse(chunk(request_id, created, model, {}, reason))
            if include_usage:
                yield sse(usage_chunk(request_id, created, model, last))
        yield DONE
    except EngineDeadError as exc:
        # the status line is long gone, so the error has to travel in the stream
        yield sse({"error": {"message": str(exc), "type": "server_error"}})
    finally:
        await stream.aclose()  # a client that left must not keep the request alive


def create_app(
    engine: AsyncEngine,
    frontend: Frontend,
    model_name: str,
    limits: ImageLimits | None = None,
    prep_workers: int = 4,
    max_body_bytes: int | None = None,
) -> FastAPI:
    limits = limits or ImageLimits()
    if max_body_bytes is None:
        # every image as base64 (4/3 of its size), plus room for the text
        max_body_bytes = limits.max_images * limits.max_bytes * 4 // 3 + 2**20
    # tokenizing and decoding images is cpu work: keep it off the event loop
    prep = ThreadPoolExecutor(
        max_workers=prep_workers, thread_name_prefix="vlmserve-prep"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        yield
        engine.stop()
        prep.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="vlmserve", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)
    app.state.prep = prep
    app.state.limits = limits
    install_error_handlers(app)

    @app.get("/health")
    async def health() -> JSONResponse:
        if engine.healthy:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(render_metrics(engine), media_type=METRICS_CONTENT_TYPE)

    @app.get("/v1/models")
    async def models() -> ModelList:
        card = ModelCard(
            id=model_name, max_model_len=engine.engine.config.max_model_len
        )
        return ModelList(data=[card])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest, http: Request
    ) -> ChatCompletion | StreamingResponse | JSONResponse:
        if not engine.healthy:
            raise EngineDeadError("the engine is not running")
        loop = asyncio.get_running_loop()
        prompt = await loop.run_in_executor(
            prep, prepare_prompt, body.messages, frontend, limits
        )
        max_len = engine.engine.config.max_model_len
        room = max_len - len(prompt.token_ids)
        if room < 1:
            return error_response(
                400,
                f"the prompt has {len(prompt.token_ids)} tokens, the context "
                f"window is {max_len} and the answer needs room too",
                param="messages",
                code="context_length_exceeded",
            )
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = engine.generate(
            request_id, prompt.token_ids, body.sampling_params(room), prompt.segments
        )
        if body.stream:
            include_usage = bool(
                body.stream_options and body.stream_options.include_usage
            )
            events = server_sent_events(
                stream, request_id, created, model_name, include_usage
            )
            return StreamingResponse(
                events,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        answer = await drain_unless_disconnected(stream, http.is_disconnected)
        if answer is None:  # nobody is listening any more, the request is aborted
            return error_response(
                499, "the client closed the request", code="client_closed_request"
            )
        text, last = answer
        return ChatCompletion(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                Choice(
                    message=ChoiceMessage(content=text),
                    finish_reason=last.finish_reason,
                )
            ],
            usage=usage_of(last),
        )

    return app

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from vlmserve.async_engine import AsyncEngine, EngineDeadError
from vlmserve.engine import RequestOutput
from vlmserve.frontend import Frontend
from vlmserve.multimodal.images import ImageError, ImageLimits, ImageTooLargeError
from vlmserve.server.chat import prepare_prompt
from vlmserve.server.protocol import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    ChoiceMessage,
    ErrorBody,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from vlmserve.server.sse import DONE, chunk, sse, usage_chunk, usage_of


def error_response(
    status: int,
    message: str,
    *,
    kind: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(message=message, type=kind, param=param, code=code)
    )
    return JSONResponse(body.model_dump(), status_code=status)


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
) -> FastAPI:
    limits = limits or ImageLimits()
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
    app.state.prep = prep
    app.state.limits = limits

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = [str(p) for p in first["loc"] if p != "body"]
        message = f"{'.'.join(where)}: {first['msg']}" if where else first["msg"]
        return error_response(400, message, param=where[-1] if where else None)

    @app.exception_handler(ImageError)
    async def bad_image(_: Request, exc: ImageError) -> JSONResponse:
        status = 413 if isinstance(exc, ImageTooLargeError) else 400
        return error_response(status, str(exc), param="messages")

    @app.exception_handler(EngineDeadError)
    async def engine_dead(_: Request, exc: EngineDeadError) -> JSONResponse:
        return error_response(503, str(exc), kind="server_error")

    @app.get("/health")
    async def health() -> JSONResponse:
        if engine.healthy:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.get("/v1/models")
    async def models() -> ModelList:
        card = ModelCard(
            id=model_name, max_model_len=engine.engine.config.max_model_len
        )
        return ModelList(data=[card])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
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
        text, last = await drain(stream)
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

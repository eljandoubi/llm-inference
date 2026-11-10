from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vlmserve.async_engine import AsyncEngine, EngineDeadError
from vlmserve.frontend import Frontend
from vlmserve.multimodal.images import ImageError, ImageLimits, ImageTooLargeError
from vlmserve.server.protocol import ErrorBody, ErrorResponse, ModelCard, ModelList


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

    return app

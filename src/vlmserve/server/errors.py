from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vlmserve.async_engine import EngineDeadError
from vlmserve.multimodal.images import ImageError, ImageTooLargeError
from vlmserve.server.protocol import ErrorBody, ErrorResponse


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


def install_error_handlers(app: FastAPI) -> None:
    """Answer every failure with an OpenAI-shaped error body."""

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

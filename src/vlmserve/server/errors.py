from fastapi.responses import JSONResponse

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

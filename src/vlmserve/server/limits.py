from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vlmserve.server.errors import error_response


class _TooLargeError(Exception):
    pass


class BodyLimitMiddleware:
    """Refuses request bodies over a size limit before the app buffers them.

    Images arrive inline as base64, so without a cap a single request could make the
    server hold as much memory as the client cares to send.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        responded = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # chunked uploads declare no length
                    raise _TooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal responded
            responded = responded or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _TooLargeError:
            if not responded:
                await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = error_response(
            413,
            f"the request body is over the {self.max_bytes} byte limit",
            code="request_too_large",
        )
        await response(scope, receive, send)

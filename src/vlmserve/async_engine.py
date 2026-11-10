import asyncio
import contextlib
import logging
import queue
import threading
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

from vlmserve.engine import Engine, RequestOutput
from vlmserve.request import ImageSegment
from vlmserve.sampling import SamplingParams

log = logging.getLogger(__name__)


class EngineDeadError(RuntimeError):
    """The engine thread is gone, nothing will be generated any more."""


@dataclass(slots=True)
class _Add:
    request_id: str
    prompt_token_ids: Sequence[int]
    params: SamplingParams
    segments: Sequence[ImageSegment]


@dataclass(slots=True)
class _Abort:
    request_id: str


class _Stop:
    pass


_Command = _Add | _Abort | _Stop


class _Stream:
    """Where the engine thread leaves the outputs of one request for the event loop."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()

    def put(self, item: RequestOutput | BaseException) -> None:
        with contextlib.suppress(RuntimeError):  # the loop may be closed already
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


class AsyncEngine:
    """Runs the engine on its own thread so the event loop never waits for the model."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._streams: dict[str, _Stream] = {}
        self._thread = threading.Thread(
            target=self._run, name="vlmserve-engine", daemon=True
        )
        self._started = False
        self._failure: BaseException | None = None

    @property
    def healthy(self) -> bool:
        return self._started and self._thread.is_alive() and self._failure is None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._commands.put(_Stop())
        self._thread.join(timeout=10)

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        params: SamplingParams,
        segments: Sequence[ImageSegment] = (),
    ) -> AsyncGenerator[RequestOutput, None]:
        if not self.healthy:
            raise EngineDeadError("the engine is not running") from self._failure
        if request_id in self._streams:
            raise ValueError(f"request {request_id!r} is already running")
        stream = _Stream()
        self._streams[request_id] = stream
        self._commands.put(_Add(request_id, prompt_token_ids, params, segments))
        finished = False
        try:
            while not finished:
                item = await stream.queue.get()
                if isinstance(item, BaseException):
                    raise item
                finished = item.finished
                yield item
        finally:
            self._streams.pop(request_id, None)
            if not finished:  # the client left, or the generator was closed
                self._commands.put(_Abort(request_id))

    def _run(self) -> None:
        engine = self.engine
        try:
            while True:
                # sleep only when there is nothing to compute
                command = self._next_command(block=not engine.has_unfinished)
                while command is not None:
                    if isinstance(command, _Stop):
                        self._fail(EngineDeadError("the engine was stopped"))
                        return
                    self._apply(command)
                    command = self._next_command(block=False)
                if engine.has_unfinished:
                    self._dispatch(engine.step())
        except BaseException as exc:
            log.exception("the engine thread died")
            self._fail(EngineDeadError(f"the engine failed: {exc!r}"), exc)

    def _next_command(self, block: bool) -> _Command | None:
        try:
            return self._commands.get(block=block)
        except queue.Empty:
            return None

    def _apply(self, command: _Add | _Abort) -> None:
        if isinstance(command, _Abort):
            self.engine.abort(command.request_id)
            return
        try:
            self.engine.add_request(
                command.request_id,
                command.prompt_token_ids,
                command.params,
                command.segments,
            )
        except ValueError as exc:  # the request is at fault, not the engine
            if stream := self._streams.get(command.request_id):
                stream.put(exc)

    def _dispatch(self, outputs: list[RequestOutput]) -> None:
        for output in outputs:
            if stream := self._streams.get(output.request_id):
                stream.put(output)

    def _fail(self, error: EngineDeadError, cause: BaseException | None = None) -> None:
        self._failure = cause or error
        error.__cause__ = cause
        for stream in list(self._streams.values()):
            stream.put(error)

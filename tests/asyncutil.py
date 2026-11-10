import asyncio
import functools
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any

from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine


def async_test[**P](test: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    """Run an async test function in its own event loop, fixtures and params intact."""

    @functools.wraps(test)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


@contextmanager
def running(rig: Rig) -> Iterator[AsyncEngine]:
    """An async engine around the rig's engine, stopped again afterwards."""
    engine = AsyncEngine(rig.engine)
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


async def until(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine


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

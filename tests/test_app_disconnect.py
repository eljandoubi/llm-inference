import asyncio
import contextlib
import time

import torch

from tests.asyncutil import running, until
from tests.rig import Rig
from vlmserve.engine import RequestOutput
from vlmserve.sampling import SamplingParams
from vlmserve.server.app import drain_unless_disconnected

CPU = torch.device("cpu")
PROMPT = [5, 6, 7, 8, 9, 10]
LONG = SamplingParams(temperature=0, max_tokens=50)


async def never() -> bool:
    return False


def slow_down(rig: Rig, seconds: float = 0.02) -> None:
    """Make every engine step take a while, so a request is still running later."""
    step = rig.engine.step

    def slow() -> list[RequestOutput]:
        time.sleep(seconds)
        return step()

    rig.engine.step = slow  # type: ignore[method-assign]


def test_an_answer_that_nobody_walks_away_from_comes_back() -> None:
    rig = Rig(CPU)

    async def main() -> tuple[str, RequestOutput] | None:
        with running(rig) as engine:
            stream = engine.generate(
                "a", PROMPT, SamplingParams(temperature=0, max_tokens=8)
            )
            return await drain_unless_disconnected(stream, never, poll=0.01)

    answer = asyncio.run(main())
    assert answer is not None
    text, last = answer
    assert last.finished
    assert text == rig.text(rig.greedy(PROMPT, 8))


def test_a_client_that_leaves_gets_its_request_aborted() -> None:
    rig = Rig(CPU)
    slow_down(rig)
    free = rig.engine.kv.pool.num_free
    polls = 0

    async def gone_after_three_polls() -> bool:
        nonlocal polls
        polls += 1
        return polls > 3

    async def main() -> object:
        with running(rig) as engine:
            stream = engine.generate("a", PROMPT, LONG)
            answer = await drain_unless_disconnected(
                stream, gone_after_three_polls, poll=0.01
            )
            await until(lambda: not rig.engine.has_unfinished)
            return answer

    assert asyncio.run(main()) is None
    assert rig.engine.kv.pool.num_free == free


def test_cancelling_the_wait_aborts_the_request_too() -> None:
    rig = Rig(CPU)
    slow_down(rig)
    free = rig.engine.kv.pool.num_free

    async def main() -> None:
        with running(rig) as engine:
            stream = engine.generate("a", PROMPT, LONG)
            task = asyncio.ensure_future(
                drain_unless_disconnected(stream, never, poll=0.01)
            )
            await until(lambda: rig.engine.has_unfinished)
            task.cancel()  # the server shutting down, say
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await until(lambda: not rig.engine.has_unfinished)

    asyncio.run(main())
    assert rig.engine.kv.pool.num_free == free

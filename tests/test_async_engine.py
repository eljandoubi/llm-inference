import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import torch

from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine, EngineDeadError
from vlmserve.engine import RequestOutput
from vlmserve.request import FinishReason
from vlmserve.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0, max_tokens=8)
PROMPT = [5, 6, 7, 8, 9, 10]


@contextmanager
def running(rig: Rig) -> Iterator[AsyncEngine]:
    engine = AsyncEngine(rig.engine)
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()


async def collect(
    engine: AsyncEngine,
    request_id: str,
    prompt: list[int],
    params: SamplingParams = GREEDY,
) -> list[RequestOutput]:
    return [o async for o in engine.generate(request_id, prompt, params)]


async def until(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.005)


def test_a_stream_gives_the_same_text_as_stepping_by_hand(device: torch.device) -> None:
    expected = Rig(device).text(Rig(device).greedy(PROMPT, 8))

    async def main() -> list[RequestOutput]:
        with running(Rig(device)) as engine:
            return await collect(engine, "a", PROMPT)

    outputs = asyncio.run(main())
    assert "".join(o.text for o in outputs) == expected
    assert outputs[-1].finished
    assert outputs[-1].finish_reason == FinishReason.LENGTH


def test_concurrent_requests_all_get_their_own_answer(device: torch.device) -> None:
    prompts = {f"r{i}": [5 + i, 6 + i, 7 + i, 8 + i] for i in range(6)}
    rig = Rig(device)
    expected = {k: rig.text(rig.greedy(p, 8)) for k, p in prompts.items()}

    async def main() -> dict[str, str]:
        with running(Rig(device)) as engine:
            results = await asyncio.gather(
                *(collect(engine, k, p) for k, p in prompts.items())
            )
        return {
            k: "".join(o.text for o in r) for k, r in zip(prompts, results, strict=True)
        }

    assert asyncio.run(main()) == expected


def test_leaving_a_stream_early_aborts_the_request(device: torch.device) -> None:
    rig = Rig(device)
    free = rig.engine.kv.pool.num_free

    async def main() -> None:
        with running(rig) as engine:
            params = SamplingParams(temperature=0, max_tokens=50)
            stream = engine.generate("a", PROMPT, params)
            await anext(stream)
            await stream.aclose()  # what a dropped connection does
            await until(lambda: not rig.engine.has_unfinished)

    asyncio.run(main())
    assert rig.engine.kv.pool.num_free == free


def test_a_request_that_is_wrong_fails_alone(device: torch.device) -> None:
    async def main() -> list[RequestOutput]:
        with running(Rig(device)) as engine:
            with pytest.raises(ValueError, match="max_model_len"):
                await collect(engine, "too-long", [1] * 100)
            return await collect(engine, "fine", PROMPT)  # the engine is still fine

    assert asyncio.run(main())[-1].finished


def test_a_request_id_can_only_run_once_at_a_time(device: torch.device) -> None:
    async def main() -> None:
        with running(Rig(device)) as engine:
            first = engine.generate(
                "a", PROMPT, SamplingParams(temperature=0, max_tokens=50)
            )
            await anext(first)
            with pytest.raises(ValueError, match="already running"):
                await collect(engine, "a", PROMPT)
            await first.aclose()

    asyncio.run(main())


def test_a_crash_in_the_engine_reaches_the_waiting_clients(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(device)

    def boom() -> list[RequestOutput]:
        raise RuntimeError("out of memory, say")

    monkeypatch.setattr(rig.engine, "step", boom)

    async def main() -> AsyncEngine:
        engine = AsyncEngine(rig.engine)
        engine.start()
        with pytest.raises(EngineDeadError, match="out of memory"):
            await collect(engine, "a", PROMPT)
        await until(lambda: not engine.healthy)
        with pytest.raises(EngineDeadError):
            await collect(engine, "b", PROMPT)
        return engine

    engine = asyncio.run(main())
    assert not engine.healthy


def test_health_follows_the_thread(device: torch.device) -> None:
    engine = AsyncEngine(Rig(device).engine)
    assert not engine.healthy  # not started yet
    engine.start()
    assert engine.healthy
    engine.stop()
    assert not engine.healthy

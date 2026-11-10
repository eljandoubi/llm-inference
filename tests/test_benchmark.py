import asyncio
import base64
import io
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
import torch
from PIL import Image

from benchmarks import load_test as lt
from tests.asyncutil import async_test
from tests.frontend import TinyFrontend
from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine
from vlmserve.multimodal.images import ImageLimits
from vlmserve.server.app import create_app


def sse(*events: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def chunk(text: str | None) -> dict[str, Any]:
    delta = {} if text is None else {"content": text}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def usage(prompt: int, done: int, cached: int = 0) -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": done,
            "total_tokens": prompt + done,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


ANSWER = sse(chunk(""), chunk("hel"), chunk("lo"), chunk(None), usage(30, 2, 16))


def client_for(
    handler: Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]],
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://test"
    )


async def ok(_: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, content=ANSWER)


def spec() -> lt.Spec:
    return lt.Spec({"messages": []}, delay=0.0)


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1, 2, 3, 4, 5], 50, 3),
        ([1, 2, 3, 4], 50, 2.5),
        ([5, 1, 3], 0, 1),
        ([5, 1, 3], 100, 5),
        ([10], 99, 10),
        ([], 50, 0),
        (list(range(101)), 90, 90),
    ],
)
def test_percentiles_interpolate(
    values: list[float], q: float, expected: float
) -> None:
    assert lt.percentile(values, q) == pytest.approx(expected)


def test_the_same_seed_makes_the_same_run() -> None:
    workload = lt.Workload(requests=5, rate=3, images=1, image_size=16)
    assert lt.make_specs(workload) == lt.make_specs(workload)
    other = lt.Workload(requests=5, rate=3, images=1, image_size=16, seed=1)
    assert lt.make_specs(other) != lt.make_specs(workload)


def test_requests_ask_for_streamed_fixed_length_answers() -> None:
    (one,) = lt.make_specs(lt.Workload(requests=1, max_tokens=7, model="m"))
    body = one.body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert (body["max_tokens"], body["temperature"], body["ignore_eos"]) == (7, 0, True)
    assert body["model"] == "m"


def test_everything_is_sent_at_once_without_a_rate() -> None:
    specs = lt.make_specs(lt.Workload(requests=10, rate=0))
    assert {s.delay for s in specs} == {0.0}


def test_arrivals_follow_the_rate() -> None:
    specs = lt.make_specs(lt.Workload(requests=2000, rate=10, prompt_words=1))
    delays = [s.delay for s in specs]
    assert delays == sorted(delays)
    assert delays[-1] == pytest.approx(200, rel=0.15)  # 2000 requests at 10 a second


def test_a_shared_system_prompt_is_the_same_for_everybody() -> None:
    specs = lt.make_specs(lt.Workload(requests=6, shared_words=30))
    firsts = [s.body["messages"][0] for s in specs]
    assert {m["role"] for m in firsts} == {"system"}
    assert len({m["content"] for m in firsts}) == 1
    assert len(firsts[0]["content"].split()) == 30
    plain = lt.make_specs(lt.Workload(requests=2, shared_words=0))
    assert plain[0].body["messages"][0]["role"] == "user"


def test_images_are_drawn_from_a_small_pool() -> None:
    workload = lt.Workload(requests=12, images=2, image_pool=3, image_size=16)
    urls = []
    for one in lt.make_specs(workload):
        *pictures, text = one.body["messages"][-1]["content"]
        assert text["type"] == "text"
        assert len(pictures) == 2
        urls += [p["image_url"]["url"] for p in pictures]
    assert 1 < len(set(urls)) <= 3  # repeats are the point
    header, _, payload = urls[0].partition(",")
    assert header == "data:image/png;base64"
    image = Image.open(io.BytesIO(base64.b64decode(payload)))
    assert image.size == (16, 16)


def test_without_images_there_is_no_pool() -> None:
    (one,) = lt.make_specs(lt.Workload(requests=1, images=0, image_pool=5))
    assert [p["type"] for p in one.body["messages"][-1]["content"]] == ["text"]


@async_test
async def test_a_streamed_answer_is_read_into_numbers() -> None:
    async with client_for(ok) as client:
        result = await lt.send(client, spec())
    assert result.ok
    assert (result.prompt_tokens, result.completion_tokens) == (30, 2)
    assert result.cached_tokens == 16
    assert len(result.gaps) == 1  # two pieces of text, one gap between them
    assert 0 < result.ttft <= result.latency


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (httpx2.Response(500, text="boom"), "HTTP 500: boom"),
        (httpx2.Response(200, content=sse(chunk("x"))), "no tokens"),
        (
            httpx2.Response(200, content=sse(chunk(""), chunk(None), usage(9, 3))),
            "no text",
        ),
        (
            httpx2.Response(200, content=sse({"error": {"message": "engine died"}})),
            "engine died",
        ),
    ],
)
@async_test
async def test_failures_are_recorded_not_raised(
    reply: httpx2.Response, message: str
) -> None:
    async def handler(_: httpx2.Request) -> httpx2.Response:
        return reply

    async with client_for(handler) as client:
        result = await lt.send(client, spec())
    assert not result.ok
    assert message in result.error


@async_test
async def test_a_refused_connection_is_a_failed_request() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    async with client_for(handler) as client:
        result = await lt.send(client, spec())
    assert not result.ok
    assert "ConnectError" in result.error


@async_test
async def test_concurrency_caps_the_requests_in_flight() -> None:
    active = peak = 0

    async def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx2.Response(200, content=ANSWER)

    specs = [spec() for _ in range(6)]
    async with client_for(handler) as client:
        results, _ = await lt.run_load(client, specs, concurrency=2)
    assert all(r.ok for r in results)
    assert peak == 2
    async with client_for(handler) as client:
        peak = 0
        await lt.run_load(client, specs)
    assert peak == 6


@async_test
async def test_requests_leave_when_the_schedule_says() -> None:
    arrivals: list[float] = []

    async def handler(_: httpx2.Request) -> httpx2.Response:
        arrivals.append(time.perf_counter())
        return httpx2.Response(200, content=ANSWER)

    specs = [lt.Spec({}, 0.0), lt.Spec({}, 0.15)]
    async with client_for(handler) as client:
        _, seconds = await lt.run_load(client, specs)
    assert arrivals[1] - arrivals[0] >= 0.14
    assert seconds >= 0.15


def good_result(ttft: float, done: int, cached: int) -> lt.Result:
    return lt.Result(
        ok=True,
        ttft=ttft,
        gaps=[0.01, 0.03],
        latency=ttft + 0.04,
        prompt_tokens=100,
        cached_tokens=cached,
        completion_tokens=done,
    )


def results_for_summary() -> list[lt.Result]:
    return [
        good_result(0.1, 10, 0),
        good_result(0.2, 10, 50),
        good_result(0.3, 20, 100),
        lt.Result(ok=False, error="HTTP 500: boom"),
    ]


def test_the_summary_adds_up() -> None:
    summary = lt.summarize(results_for_summary(), seconds=2.0)
    assert (summary.ok, summary.failed) == (3, 1)
    assert summary.output_tokens == 40
    assert summary.output_rate == pytest.approx(20)
    assert summary.request_rate == pytest.approx(1.5)
    assert (summary.prompt_tokens, summary.cached_tokens) == (300, 150)
    assert summary.ttft_ms["p50"] == pytest.approx(200)
    assert summary.ttft_ms["mean"] == pytest.approx(200)
    assert summary.gap_ms["p50"] == pytest.approx(20)
    assert summary.errors == ["HTTP 500: boom"]


def test_the_report_names_what_it_measured() -> None:
    text = lt.format_summary(lt.summarize(results_for_summary(), seconds=2.0))
    assert "3 ok, 1 failed" in text
    assert "20.0 tokens/s" in text
    assert "150 from the prefix cache, 50%" in text
    for name in ("time to first token", "gap between tokens", "end to end"):
        assert name in text
    assert "error: HTTP 500: boom" in text


def test_an_empty_run_does_not_divide_by_zero() -> None:
    summary = lt.summarize([], seconds=0.0)
    assert summary.request_rate == 0 == summary.output_rate
    assert "0 ok, 0 failed" in lt.format_summary(summary)


def test_metrics_pages_are_read_back() -> None:
    page = "# HELP a x\n# TYPE a gauge\na 1.5\nb_total 7\n"
    assert lt.parse_metrics(page) == {"a": 1.5, "b_total": 7.0}


@async_test
async def test_a_run_against_the_real_app_gets_the_answers_asked_for() -> None:
    rig = Rig(torch.device("cpu"))
    engine = AsyncEngine(rig.engine)
    app = create_app(engine, TinyFrontend(), "tiny", ImageLimits())
    workload = lt.Workload(
        requests=6, prompt_words=3, images=1, image_pool=2, image_size=16, max_tokens=5
    )
    engine.start()
    try:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://app"
        ) as client:
            before = await lt.read_metrics(client)
            results, seconds = await lt.run_load(client, lt.make_specs(workload))
            after = await lt.read_metrics(client)
    finally:
        engine.stop()
    summary = lt.summarize(results, seconds)
    assert summary.errors == []
    assert summary.ok == 6
    assert summary.output_tokens == 6 * 5  # ignore_eos: as long as asked
    assert summary.prompt_tokens > 0
    assert (
        after["vlmserve_requests_finished_total"]
        - before["vlmserve_requests_finished_total"]
        == 6
    )

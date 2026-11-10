"""Load test for an OpenAI-style chat server: Poisson arrivals, streamed answers,
time to first token, gaps between tokens, throughput and cache hits.

It talks plain HTTP, so it works against vlmserve and against any other server with
a /v1/chat/completions route:

    vlmserve serve --model <model> &
    python -m benchmarks.load_test --requests 64 --rate 8 --images 1
"""

import argparse
import asyncio
import base64
import io
import json
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from PIL import Image, ImageDraw

# a list literal would take one line per word
WORDS = (  # noqa: SIM905
    "river stone market window engine garden paper signal winter copper harbor "
    "violin shadow lantern pocket bridge meadow thunder marble button candle "
    "forest island ladder mirror pepper rocket saddle teapot velvet wagon yellow "
    "anchor basket castle desert feather glacier hammer jungle kettle magnet"
).split()


@dataclass(frozen=True, slots=True)
class Workload:
    requests: int = 32
    rate: float = 0.0  # requests per second, 0 sends them all at once
    concurrency: int = 0  # most requests in flight, 0 means no limit
    prompt_words: int = 60
    shared_words: int = 0  # words of system prompt every request has in common
    images: int = 0  # per request
    image_pool: int = 4  # distinct images to draw from, so some repeat
    image_size: int = 224
    max_tokens: int = 64
    seed: int = 0
    model: str | None = None  # only for servers that check the name


@dataclass(frozen=True, slots=True)
class Spec:
    """One request to send."""

    body: dict[str, Any]
    delay: float  # seconds after the start of the run


@dataclass(slots=True)
class Result:
    ok: bool
    error: str = ""
    ttft: float = 0.0  # seconds to the first piece of text
    gaps: list[float] = field(default_factory=list)  # between pieces of text
    latency: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0


def make_image_url(size: int, index: int) -> str:
    """A flat picture with a few shapes, different for every index and cheap to send."""
    rng = random.Random(index)
    image = Image.new("RGB", (size, size), tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        x, y = rng.randrange(size), rng.randrange(size)
        low, high = max(1, size // 8), max(2, size // 2)
        w, h = rng.randrange(low, high), rng.randrange(low, high)
        draw.rectangle(
            (x, y, x + w, y + h), fill=tuple(rng.randrange(256) for _ in range(3))
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def make_specs(workload: Workload) -> list[Spec]:
    """Every request of the run, built up front so a seed repeats a run."""
    rng = random.Random(workload.seed)
    shared = " ".join(rng.choices(WORDS, k=workload.shared_words))
    pool = [
        make_image_url(workload.image_size, i)
        for i in range(workload.image_pool if workload.images else 0)
    ]
    specs: list[Spec] = []
    clock = 0.0
    for _ in range(workload.requests):
        if workload.rate > 0:
            clock += rng.expovariate(workload.rate)
        question = " ".join(rng.choices(WORDS, k=workload.prompt_words))
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": rng.choice(pool)}}
            for _ in range(workload.images)
        ]
        content.append({"type": "text", "text": question})
        messages: list[dict[str, Any]] = []
        if shared:
            messages.append({"role": "system", "content": shared})
        messages.append({"role": "user", "content": content})
        body = {
            "messages": messages,
            "max_tokens": workload.max_tokens,
            "temperature": 0,
            "ignore_eos": True,  # every answer as long as asked, so runs compare
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if workload.model:
            body["model"] = workload.model
        specs.append(Spec(body, clock))
    return specs


async def send(client: httpx2.AsyncClient, spec: Spec) -> Result:
    """One streamed request, timed."""
    started = time.perf_counter()
    result = Result(ok=False)
    last = started
    try:
        async with client.stream(
            "POST", "/v1/chat/completions", json=spec.body
        ) as reply:
            if reply.status_code != 200:
                body = (await reply.aread()).decode(errors="replace")
                result.error = f"HTTP {reply.status_code}: {body[:200]}"
                return result
            async for line in reply.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[len("data: ") :])
                if "error" in event:
                    result.error = str(event["error"].get("message", event["error"]))
                    return result
                if usage := event.get("usage"):
                    result.prompt_tokens = usage["prompt_tokens"]
                    result.completion_tokens = usage["completion_tokens"]
                    details = usage.get("prompt_tokens_details") or {}
                    result.cached_tokens = details.get("cached_tokens", 0)
                choices = event.get("choices") or []
                if choices and choices[0]["delta"].get("content"):
                    now = time.perf_counter()
                    if result.ttft == 0.0:
                        result.ttft = now - started
                    else:
                        result.gaps.append(now - last)
                    last = now
    except httpx2.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.latency = time.perf_counter() - started
    if result.completion_tokens == 0:
        result.error = "the answer had no tokens"
    elif result.ttft == 0.0:
        result.error = "the stream carried no text, only invisible tokens"
    else:
        result.ok = True
    return result


async def run_load(
    client: httpx2.AsyncClient, specs: Sequence[Spec], concurrency: int = 0
) -> tuple[list[Result], float]:
    """Send everything on the schedule. Returns the results and the wall time."""
    gate = asyncio.Semaphore(concurrency) if concurrency > 0 else None
    begin = time.perf_counter()

    async def at_its_time(spec: Spec) -> Result:
        await asyncio.sleep(max(0.0, begin + spec.delay - time.perf_counter()))
        if gate is None:
            return await send(client, spec)
        async with gate:
            return await send(client, spec)

    results = await asyncio.gather(*(at_its_time(s) for s in specs))
    return list(results), time.perf_counter() - begin


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100), interpolating between the nearest values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True, slots=True)
class Summary:
    ok: int
    failed: int
    seconds: float
    request_rate: float
    output_tokens: int
    output_rate: float
    prompt_tokens: int
    cached_tokens: int
    ttft_ms: dict[str, float]
    gap_ms: dict[str, float]
    latency_ms: dict[str, float]
    errors: list[str]


def _spread(values: Sequence[float]) -> dict[str, float]:
    ms = [v * 1000 for v in values]
    return {
        "mean": sum(ms) / len(ms) if ms else 0.0,
        "p50": percentile(ms, 50),
        "p90": percentile(ms, 90),
        "p99": percentile(ms, 99),
    }


def _count_errors(results: Sequence[Result]) -> list[str]:
    """One line per kind of failure with how often it happened. Numbers don't count
    (except an HTTP status), or "has 570 tokens" and "has 571 tokens" are two kinds."""
    counts = Counter(
        re.sub(r"(?<!\d)(?<!HTTP )\d+", "#", r.error)[:100] for r in results if not r.ok
    )
    return [f"{n}x {message}" for message, n in counts.most_common()]


def summarize(results: Sequence[Result], seconds: float) -> Summary:
    good = [r for r in results if r.ok]
    tokens = sum(r.completion_tokens for r in good)
    return Summary(
        ok=len(good),
        failed=len(results) - len(good),
        seconds=seconds,
        request_rate=len(good) / seconds if seconds else 0.0,
        output_tokens=tokens,
        output_rate=tokens / seconds if seconds else 0.0,
        prompt_tokens=sum(r.prompt_tokens for r in good),
        cached_tokens=sum(r.cached_tokens for r in good),
        ttft_ms=_spread([r.ttft for r in good]),
        gap_ms=_spread([g for r in good for g in r.gaps]),
        latency_ms=_spread([r.latency for r in good]),
        errors=_count_errors(results),
    )


def format_summary(summary: Summary) -> str:
    cached = (
        summary.cached_tokens / summary.prompt_tokens if summary.prompt_tokens else 0
    )
    lines = [
        f"requests            {summary.ok} ok, {summary.failed} failed",
        f"duration            {summary.seconds:.2f} s",
        f"request throughput  {summary.request_rate:.2f} req/s",
        f"output throughput   {summary.output_rate:.1f} tokens/s "
        f"({summary.output_tokens} tokens)",
        f"prompt tokens       {summary.prompt_tokens} "
        f"({summary.cached_tokens} from the prefix cache, {cached:.0%})",
    ]
    for name, spread in (
        ("time to first token", summary.ttft_ms),
        ("gap between tokens", summary.gap_ms),
        ("end to end", summary.latency_ms),
    ):
        lines.append(
            f"{name:<20}p50 {spread['p50']:.1f}  p90 {spread['p90']:.1f}  "
            f"p99 {spread['p99']:.1f}  mean {spread['mean']:.1f}  (ms)"
        )
    lines += [f"error: {message}" for message in summary.errors]
    return "\n".join(lines)


def parse_metrics(text: str) -> dict[str, float]:
    """Samples of a Prometheus text page, by metric name."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            values[name] = float(value)
    return values


async def read_metrics(client: httpx2.AsyncClient) -> dict[str, float]:
    reply = await client.get("/metrics")
    return parse_metrics(reply.text) if reply.status_code == 200 else {}


def build_parser() -> argparse.ArgumentParser:
    defaults = Workload()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="only needed by servers that check the name")
    parser.add_argument("--requests", type=int, default=defaults.requests)
    parser.add_argument(
        "--rate", type=float, default=defaults.rate, help="req/s, 0 = all at once"
    )
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--prompt-words", type=int, default=defaults.prompt_words)
    parser.add_argument("--shared-words", type=int, default=defaults.shared_words)
    parser.add_argument("--images", type=int, default=defaults.images)
    parser.add_argument("--image-pool", type=int, default=defaults.image_pool)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--warmup", type=int, default=2, help="requests sent first, not counted"
    )
    parser.add_argument("--json", type=Path, help="also write the numbers here")
    return parser


async def run(args: argparse.Namespace) -> int:
    workload = Workload(**{name: getattr(args, name) for name in asdict(Workload())})
    specs = make_specs(workload)
    timeout = httpx2.Timeout(600.0, connect=10.0)
    async with httpx2.AsyncClient(base_url=args.url, timeout=timeout) as client:
        if args.warmup:
            warm = make_specs(
                Workload(**{**asdict(workload), "requests": args.warmup, "rate": 0.0})
            )
            await run_load(client, warm)
        before = await read_metrics(client)
        results, seconds = await run_load(client, specs, workload.concurrency)
        after = await read_metrics(client)
    summary = summarize(results, seconds)
    print(format_summary(summary))
    extras = {
        name: after[name] - before.get(name, 0.0)
        for name in (
            "vlmserve_encoder_cache_hits_total",
            "vlmserve_image_tokens_encoded_total",
        )
        if name in after
    }
    for name, delta in extras.items():
        print(f"{name.removeprefix('vlmserve_'):<36}{delta:.0f}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "workload": asdict(workload),
                    "summary": asdict(summary),
                    "server": extras,
                },
                indent=2,
            )
        )
    return 0 if summary.failed == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())

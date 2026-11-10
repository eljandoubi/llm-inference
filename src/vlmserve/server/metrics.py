from typing import NamedTuple

from vlmserve.async_engine import AsyncEngine

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class Sample(NamedTuple):
    name: str
    kind: str
    help: str
    value: float


def gauge(name: str, help_text: str, value: float) -> Sample:
    return Sample(name, "gauge", help_text, value)


def counter(name: str, help_text: str, value: float) -> Sample:
    return Sample(name, "counter", help_text, value)


def render(engine: AsyncEngine) -> str:
    """The engine's numbers in the Prometheus text format.

    The engine thread keeps counting while this reads, so the values are a snapshot
    that may be a step apart from one another. That is fine for a scrape.
    """
    inner = engine.engine
    stats, queue = inner.stats, inner.scheduler
    samples = [
        gauge("engine_healthy", "1 while the engine thread runs.", int(engine.healthy)),
        gauge("requests_running", "Requests in the running batch.", len(queue.running)),
        gauge("requests_waiting", "Requests queued for a slot.", len(queue.waiting)),
        gauge("kv_cache_usage_ratio", "Share of KV blocks in use.", inner.kv.usage),
        counter(
            "requests_finished_total",
            "Requests run to the end.",
            stats.finished_requests,
        ),
        counter(
            "prompt_tokens_total",
            "Prompt tokens of finished requests.",
            stats.prompt_tokens,
        ),
        counter(
            "prefix_cache_hit_tokens_total",
            "Prompt tokens of finished requests that came from the prefix cache.",
            stats.cached_prompt_tokens,
        ),
        counter(
            "generation_tokens_total", "Tokens sampled so far.", stats.generated_tokens
        ),
    ]
    if (cache := inner.encoder_cache) is not None:
        samples += [
            counter(
                "image_tokens_encoded_total",
                "Image tokens that went through the vision tower.",
                inner.runner.images_encoded,
            ),
            counter(
                "encoder_cache_hits_total",
                "Image embeddings reused from the encoder cache.",
                cache.hits,
            ),
            gauge(
                "encoder_cache_tokens",
                "Image tokens in the encoder cache.",
                cache.num_tokens,
            ),
        ]
    lines = []
    for sample in samples:
        lines += [
            f"# HELP vlmserve_{sample.name} {sample.help}",
            f"# TYPE vlmserve_{sample.name} {sample.kind}",
            f"vlmserve_{sample.name} {sample.value}",
        ]
    return "\n".join(lines) + "\n"

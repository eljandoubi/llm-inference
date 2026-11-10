from collections.abc import Iterator

from vlmserve.async_engine import AsyncEngine

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _lines(name: str, kind: str, help_text: str, value: float) -> Iterator[str]:
    yield f"# HELP vlmserve_{name} {help_text}"
    yield f"# TYPE vlmserve_{name} {kind}"
    yield f"vlmserve_{name} {value}"


def render(engine: AsyncEngine) -> str:
    """The engine's numbers in the Prometheus text format.

    The engine thread keeps counting while this reads, so the values are a snapshot
    that may be a step apart from one another. That is fine for a scrape.
    """
    inner = engine.engine
    stats = inner.stats
    scheduler = inner.scheduler
    samples: list[tuple[str, str, str, float]] = [
        (
            "engine_healthy",
            "gauge",
            "1 while the engine thread runs.",
            int(engine.healthy),
        ),
        (
            "requests_running",
            "gauge",
            "Requests in the running batch.",
            len(scheduler.running),
        ),
        (
            "requests_waiting",
            "gauge",
            "Requests queued for a slot.",
            len(scheduler.waiting),
        ),
        ("kv_cache_usage_ratio", "gauge", "Share of KV blocks in use.", inner.kv.usage),
        (
            "requests_finished_total",
            "counter",
            "Requests that ran to their end.",
            stats.finished_requests,
        ),
        (
            "prompt_tokens_total",
            "counter",
            "Prompt tokens of finished requests.",
            stats.prompt_tokens,
        ),
        (
            "prefix_cache_hit_tokens_total",
            "counter",
            "Prompt tokens of finished requests that came from the prefix cache.",
            stats.cached_prompt_tokens,
        ),
        (
            "generation_tokens_total",
            "counter",
            "Tokens sampled so far.",
            stats.generated_tokens,
        ),
    ]
    cache = inner.encoder_cache
    if cache is not None:
        samples += [
            (
                "image_tokens_encoded_total",
                "counter",
                "Image tokens that went through the vision tower.",
                inner.runner.images_encoded,
            ),
            (
                "encoder_cache_hits_total",
                "counter",
                "Image embeddings reused from the encoder cache.",
                cache.hits,
            ),
            (
                "encoder_cache_tokens",
                "gauge",
                "Image tokens held in the encoder cache.",
                cache.num_tokens,
            ),
        ]
    return "\n".join(line for sample in samples for line in _lines(*sample)) + "\n"

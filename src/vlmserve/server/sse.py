import json
from typing import Any

from vlmserve.engine import RequestOutput
from vlmserve.server.protocol import PromptTokensDetails, Usage

DONE = "data: [DONE]\n\n"


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def usage_of(output: RequestOutput) -> Usage:
    return Usage(
        prompt_tokens=output.num_prompt_tokens,
        completion_tokens=output.num_generated_tokens,
        total_tokens=output.num_prompt_tokens + output.num_generated_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=output.num_cached_tokens
        ),
    )


def chunk(
    request_id: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def usage_chunk(
    request_id: str, created: int, model: str, output: RequestOutput
) -> dict[str, Any]:
    """The extra last chunk of a stream that asked for include_usage."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_of(output).model_dump(),
    }

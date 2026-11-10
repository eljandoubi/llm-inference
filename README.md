# llm-inference

`vlmserve` is a small vision-language model (VLM) server that is meant to be read.
The engine ideas come from [tinyserve](https://github.com/priyammaz/tinyserve) and
the PagedAttention paper: a paged KV cache, prefix caching, continuous batching with
chunked prefill. On top of that it handles what a VLM needs: image inputs, a cache
for vision embeddings, and prefix caching that knows which image a block came from.

It is plain PyTorch, so the same code runs on CPU, Apple silicon (MPS) and CUDA.

Work in progress: the sections below get filled in as the pieces land.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). With uv installed:

```sh
uv sync            # creates .venv with the locked dependencies
uv run pytest      # unit tests, no GPU and no model download needed
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Add a dependency with `uv add <package>` (or `uv add --dev <package>` for tooling),
and commit the updated `uv.lock`.
# vlmserve

[![ci](https://github.com/eljandoubi/llm-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/eljandoubi/llm-inference/actions/workflows/ci.yml)

A small vision-language model server, written to be read. It batches requests
continuously on a paged KV cache, reuses the cache of prompts it has seen (images
included) and speaks the OpenAI chat API. It is plain PyTorch, so one code path runs on
CPU, Apple silicon, CUDA and Intel XPU.

The design is the one from [PagedAttention](https://arxiv.org/abs/2309.06180) and vLLM,
as shown in the small [tinyserve](https://github.com/priyammaz/tinyserve). The
scheduling loop and the prefix cache are written again from those ideas, not copied from
its code. What is new here is the vision side (images inside the block hashes, a cache of
image embeddings, the Hugging Face loaders) and running on any device. About 3,700 lines
of source, strict typing, 665 tests.

Status: 0.1. On CPU and Apple silicon it matches `transformers` token for token on tiny
random LLaVA and text models. It has not been run on real weights or on a CUDA GPU.

- Continuous batching with chunked prefill: decodes and prefill chunks share one token
  budget per step, so a long prompt does not stall the requests already generating.
- Paged KV cache with prefix caching. A block that holds part of an image also hashes
  the image, so the same text with another picture never reuses it.
- An embedding cache: a repeated image skips the vision tower, even when the text around
  it changed.
- LLaVA checkpoints (CLIP or SigLIP tower, Llama or Qwen2 text model) and plain Llama or
  Qwen2 models, from safetensors.
- Device and dtype are picked for you by trying the ops the engine needs, not by trusting
  capability flags.
- Streaming, usage with cached token counts, stop strings, size limits, aborts when the
  client hangs up, Prometheus metrics.

## Quick start

The project uses [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/eljandoubi/llm-inference && cd llm-inference
uv sync
uv run vlmserve devices        # what this machine can run on, and in which dtype
```

Serve a model, from the Hub or a local folder:

```sh
uv run vlmserve serve --model llava-hf/llava-interleave-qwen-0.5b-hf
```

Ask it about a picture:

```sh
IMG=$(base64 < cat.jpg | tr -d '\n')
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d @- <<EOF
{"messages": [{"role": "user", "content": [
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG"}},
   {"type": "text", "text": "What is in this picture?"}]}],
 "max_tokens": 128}
EOF
```

Any OpenAI client works too:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="llava-hf/llava-interleave-qwen-0.5b-hf",
    messages=[{"role": "user", "content": "Say hi in three languages."}],
    stream=True,
)
for part in stream:
    print(part.choices[0].delta.content or "", end="")
```

To answer one prompt in the terminal, without a server:

```sh
uv run vlmserve generate --model <model> --image cat.jpg --prompt "What is in this picture?"
```

To try everything without downloading a model, write a random one. The answers are
noise, the plumbing is real:

```sh
uv run python -m randomhf /tmp/tiny --preset tiny
uv run vlmserve generate --model /tmp/tiny --prompt "hello" --max-tokens 16
```

## Models and hardware

| Device | State | Default dtype |
| --- | --- | --- |
| CPU | runs the whole test suite; CI covers Linux and macOS | float32 |
| Apple silicon (MPS) | tested on an M4, used for the numbers below | float16 |
| CUDA, ROCm | same code, never run (no GPU was available) | bfloat16 on sm80+, else float16 |
| Intel XPU | same, never run | bfloat16 |

`vlmserve devices` warns when a CPU-only torch sits next to an NVIDIA card. On Linux the
locked torch wheels are the PyPI builds with the CUDA libraries (a few GB). On a machine
without a GPU, install the CPU wheel instead:
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.

| Model family | Example | State |
| --- | --- | --- |
| LLaVA, CLIP tower, Llama text | `llava-hf/llava-1.5-7b-hf` | config, tokenizer, chat template and weight index checked against the real repo (`pytest -m hub`), weights never run |
| LLaVA, SigLIP tower, Qwen2 text | `llava-hf/llava-interleave-qwen-0.5b-hf` | same, but one weight file means no index to check |
| Plain Llama or Qwen2 text | any | matches `transformers` on random weights |

No real weights were downloaded: the network this was written on blocks the weight CDN.
What could be checked is that the real tokenizers, templates and image processors give
exactly the prompts and pixel tensors `transformers` gives (576 image tokens for
llava-1.5, 729 for llava-interleave), and that every weight name in llava-1.5's index
finds its place. For the numbers, tiny random models of both kinds are written to disk,
loaded by vlmserve and decoded greedily next to `transformers`: same tokens, on CPU and
MPS, with several requests in flight, chunked prefill and cache hits. If you run a real
checkpoint, an issue with the result is welcome.

Refused with an error: LLaVA-NeXT tiling, Idefics3 and SmolVLM, Qwen2-VL (it needs
M-RoPE), PaliGemma and Gemma 3 (bidirectional attention over images), Llama 3.1 RoPE
scaling, sliding-window and MoE text models.

## How it works

```mermaid
flowchart LR
    client([client]) -->|HTTP, SSE| app
    subgraph proc[one process]
        direction LR
        app["server/app.py<br/>limits, validation"] --> prep["prep threads<br/>decode images, chat template, tokenize"]
        prep --> ae["AsyncEngine<br/>event loop to engine thread"]
        subgraph eng[engine thread]
            direction TB
            E[Engine] --> S["Scheduler<br/>continuous batching"]
            S --> KV["KVCacheManager<br/>block pool, prefix cache"]
            S --> EC[EncoderCache]
            E --> R[ModelRunner]
            R --> V["vision encoder<br/>tower + projector"]
            R --> D["Decoder<br/>paged attention"]
            R --> SM[Sampler]
        end
        ae --> E
    end
```

**Scheduling.** Each step has one token budget (`--max-num-batched-tokens`). Running
requests go first: a decode takes one token, a request still reading its prompt takes a
chunk. Waiting requests come after, oldest first, and nobody jumps the queue. When the
cache runs out of blocks the newest request hands its blocks back and starts over later,
so an old request never loses its place to a young one. Embedding images has a budget of
its own, and a chunk stops short of an image that does not fit yet.

**KV cache.** Memory is a pool of fixed-size blocks, 16 tokens by default. A request keeps
a list of block ids and adds to it as it grows, so nothing is reserved for the longest
possible answer. The pool is sized from the memory left after the weights, and never
bigger than `--max-num-seqs` full-length sequences could use.

**Prefix cache.** Every full block gets a hash of its tokens and of its parent's hash, so
a hit means the whole prefix is the same. A block that overlaps an image also hashes the
image's content key and the part of the image it covers, because the placeholder tokens
are the same for every picture. Finished requests leave their blocks in the pool until
they are evicted. The response says how much of the prompt was a hit
(`usage.prompt_tokens_details.cached_tokens`).

**Images.** The frontend turns each image into a run of placeholder tokens plus a pixel
tensor. The runner embeds the images a step needs in one batch, writes the embeddings over
the placeholder rows, and keeps them in an LRU cache counted in tokens. A request drops its
own copy once the whole image has been computed.

**Serving.** One thread owns the engine, so the event loop never waits for the model.
Tokenizing and image decoding run on a small thread pool. Each request has a queue that the
engine thread fills with `call_soon_threadsafe`. When a client disconnects, streamed or
not, the request is aborted and its blocks come back. If the engine thread dies, new
requests get a 503 and open streams get an error event.

**Images are data URLs only.** A server that fetches any URL a client names can be pointed
at internal addresses, so remote image URLs are refused. Images are limited in count,
bytes and pixels, the request body is capped while it streams in, and EXIF rotation is
applied. A JPEG is decoded at reduced size when the model will shrink it anyway.

## Using the server

| Route | |
| --- | --- |
| `POST /v1/chat/completions` | text and `image_url` parts (base64 data URLs), `stream`, `stream_options.include_usage`, `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop`, `ignore_eos` (vLLM's extension, useful for benchmarks) |
| `GET /v1/models` | the served name and its context window |
| `GET /health` | 200, or 503 once the engine thread is gone |
| `GET /metrics` | Prometheus text: running and waiting requests, KV usage, token counters, prefix and embedding cache hits |

What the server cannot do and would otherwise get wrong (`tools`, `logprobs`, `logit_bias`,
`response_format` other than text, `frequency_penalty`, `n > 1`) is a 400, not silently
ignored. Other errors look like OpenAI's: 413 for an oversized image or body, 499 when the
client left, 503 when the engine is down.

Commands: `vlmserve serve`, `vlmserve generate` and `vlmserve devices`. `--help` lists every
flag; these are the ones you will touch.

| Flag | Default | |
| --- | --- | --- |
| `--model` | required | a local folder or a Hub repo id (`--revision` for a branch or commit) |
| `--device`, `--dtype` | `auto` | `cpu`, `cuda`, `cuda:1`, `mps`, `xpu`; `float32`, `float16`, `bfloat16` |
| `--max-model-len` | the model's, at most 4096 | context window |
| `--max-num-seqs` | 32 | requests running at once |
| `--max-num-batched-tokens` | 2048 | the per-step token budget |
| `--num-kv-blocks` or `--kv-memory-fraction` | free memory x 0.8 | KV cache size |
| `--no-prefix-caching` | off | turn the prefix cache off |
| `--host`, `--port` | `127.0.0.1`, 8000 | not reachable from the network unless you ask |
| `--max-images`, `--max-image-bytes`, `--max-image-pixels` | 8, 10 MiB, 40 MP | per-request limits |

There is no authentication. Keep the server on localhost, or put a reverse proxy that checks
credentials in front of it before using `--host 0.0.0.0`.

Out of memory: lower `--max-model-len`, `--max-num-seqs` or `--kv-memory-fraction`. Gated
models need `HF_TOKEN`; without a network, use `HF_HUB_OFFLINE=1` with a local folder or a
filled cache.

## Performance

These numbers are about the serving stack (batching, prefix cache, image cache), not model
speed. The model is a random llava-style checkpoint with real shapes: 249M parameters, a
30-layer Llama plus a ViT-B tower, 196 image tokens per image (`python -m randomhf`). Apple
M4, 16 GB, MPS, float16, one run per scenario, nothing else heavy running. Read the tables
as ratios between settings. The raw output is in
[`benchmarks/results`](benchmarks/results/apple-m4-mps.txt) and `benchmarks/suite.sh` runs
it again.

Continuous batching, 64 new tokens per request, prompts of about 575 tokens:

| Requests in flight | Output tokens/s | Time to first token (p50 / p90) | Gap between tokens (p50) |
| ---: | ---: | ---: | ---: |
| 1 | 37.6 | 142 / 149 ms | 24.7 ms |
| 4 | 63.6 | 670 / 1086 ms | 47.4 ms |
| 16 | 100.3 | 639 / 2428 ms | 123 ms |

Sixteen requests in flight give 2.7 times the throughput of one, and each token takes
longer, as expected when the device is close to saturated.

Prefix cache, 32 requests with 4 in flight, prompts of about 940 tokens of which about 860
are one shared system prompt, 32 new tokens:

| Prefix cache | Prompt tokens from cache | Time to first token (p50) | Output tokens/s | End to end (p50) |
| --- | ---: | ---: | ---: | ---: |
| on | 91% | 168 ms | 72.5 | 1.73 s |
| off | 0% | 1036 ms | 39.8 | 3.06 s |

Images, one image of 196 tokens plus a short question, 32 requests with 4 in flight:

| Setup | Prompt tokens from cache | Time to first token (p50) | Output tokens/s |
| --- | ---: | ---: | ---: |
| same image every time, both caches on | 71% | 134 ms | 111 |
| images drawn from a pool of 32, both caches on | 36% | 213 ms | 98 |
| same image, prefix cache off (embedding cache only) | 0% | 368 ms | 96 |

To measure any OpenAI-style server:

```sh
uv run vlmserve serve --model <model> &
uv run python -m benchmarks.load_test --requests 64 --rate 8 --images 1 --shared-words 100
```

It sends streamed requests on a Poisson schedule (or at a fixed concurrency) and reports
time to first token, the gap between tokens, throughput and cache hits.

## Development

```sh
uv sync
make check                 # ruff, mypy and pytest with coverage: what CI runs
uv run pytest -m hub       # opt in, needs the network: the real LLaVA repos
```

`uv add <package>` (`--dev` for tooling) updates `uv.lock`; commit it. CI runs lint, mypy
and the tests on Linux (Python 3.12 and 3.13) and macOS, and builds the package.

The tests never use the network. They check numbers against two references: a plain,
unpaged implementation (`tests/dense.py`) and `transformers` itself, on tiny random
checkpoints that `randomhf` writes in the real file layouts (classic and nested LLaVA
layouts, tied and untied embeddings, CLIP and SigLIP). Model tests run on every device the
machine has: CPU, plus MPS on a Mac.

```
src/vlmserve/
  engine.py, scheduler.py, runner.py   the core loop: schedule, run a batch, sample
  kv/                                  block pool, prefix hashing, KV cache manager
  model/                               decoder, paged attention, checkpoint reading
  multimodal/                          image decoding and limits, embedding cache
  hf/                                  LLaVA vision tower, chat template frontend, loader
  server/                              FastAPI app, protocol, SSE, limits, metrics
  async_engine.py                      the engine thread behind an asyncio interface
  device.py, config.py, cli.py         device and dtype choice, settings, commands
benchmarks/                            load test, suite script, saved results
randomhf/                              random checkpoints for tests and benchmarks
tests/
```

To add a model family, write a `VisionEncoder` (embeds `pixel_values` into the text model's
space) and a `Frontend` (chat template, tokenizer, image placeholders), then teach
`hf/loader.py` the new `model_type`. The engine only knows those two interfaces.

## Limits

- One process, one device. No tensor or pipeline parallelism, no speculative decoding, no
  quantization.
- The decoder handles Llama-style models with default RoPE. RoPE scaling, sliding windows
  and MoE are the first things to add for more text models.
- Attention is PyTorch's SDPA over gathered blocks, not a fused paged kernel. It is correct
  everywhere and fast enough to show the scheduling wins above; a fused CUDA or Triton
  kernel is the next step for a big GPU.
- More VLM families (Idefics3, SmolVLM, Qwen2-VL) need their own frontends, and Qwen2-VL
  a different position encoding.

## Credits

[PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al.) and the vLLM project for
the design. [tinyserve](https://github.com/priyammaz/tinyserve) for showing how small a
readable version can be: the scheduler's step loop (running requests first, evict the
newest when blocks run out, admit nothing new in a step that evicted) and the shape of the
prefix cache follow its structure, written again here with different data structures.
Hugging Face `transformers` for the model definitions this is checked against.

## License

Apache 2.0, see [LICENSE](LICENSE).

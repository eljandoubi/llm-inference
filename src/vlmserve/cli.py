"""The vlmserve command line: serve a model over HTTP, generate from a prompt in the
terminal, or list the devices this machine can run on."""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import uvicorn

from vlmserve import __version__
from vlmserve.async_engine import AsyncEngine
from vlmserve.config import EngineConfig
from vlmserve.device import (
    accelerators,
    build_hint,
    describe,
    resolve_device,
    resolve_dtype,
)
from vlmserve.engine import Engine
from vlmserve.frontend import ImagePart, TextPart, Turn
from vlmserve.hf.loader import LoadedModel, load_model, open_model
from vlmserve.multimodal.images import ImageLimits, image_key, open_image
from vlmserve.sampling import SamplingParams
from vlmserve.server.app import create_app

log = logging.getLogger("vlmserve")

DEFAULT_CONTEXT = 4096  # what a window is capped at when the model allows far more
LOG_LEVELS = ("debug", "info", "warning", "error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmserve", description="A small vision-language model server."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="serve an OpenAI-style HTTP API")
    _model_options(serve)
    _server_options(serve)
    serve.set_defaults(run=run_serve)

    generate = commands.add_parser("generate", help="answer one prompt and exit")
    _model_options(generate)
    _generation_options(generate)
    generate.set_defaults(run=run_generate)

    devices = commands.add_parser("devices", help="show what this machine can run on")
    devices.set_defaults(run=run_devices)
    return parser


def _model_options(parser: argparse.ArgumentParser) -> None:
    defaults = EngineConfig(model="")  # so the defaults live in one place
    model = parser.add_argument_group("model")
    model.add_argument(
        "--model", required=True, help="a model directory or a Hugging Face repo id"
    )
    model.add_argument("--revision", help="branch, tag or commit of the repo")
    model.add_argument(
        "--device", default=defaults.device, help="auto, cpu, cuda, cuda:1, mps, xpu"
    )
    model.add_argument(
        "--dtype", default=defaults.dtype, help="auto, float32, float16 or bfloat16"
    )
    model.add_argument("--log-level", choices=LOG_LEVELS, default="info")

    engine = parser.add_argument_group("engine")
    engine.add_argument(
        "--max-model-len",
        type=int,
        help=f"context window in tokens (default: the model's, max {DEFAULT_CONTEXT})",
    )
    engine.add_argument("--max-num-seqs", type=int, default=defaults.max_num_seqs)
    engine.add_argument(
        "--max-num-batched-tokens", type=int, default=defaults.max_num_batched_tokens
    )
    engine.add_argument("--block-size", type=int, default=defaults.block_size)
    engine.add_argument(
        "--num-kv-blocks",
        type=int,
        help="size of the KV cache in blocks (default: fill the free memory)",
    )
    engine.add_argument(
        "--kv-memory-fraction",
        type=float,
        default=defaults.kv_memory_fraction,
        help="share of the memory left after the weights that the KV cache may take",
    )
    engine.add_argument(
        "--no-prefix-caching",
        dest="prefix_caching",
        action="store_false",
        help="don't reuse the KV blocks of a prompt prefix seen before",
    )
    engine.add_argument(
        "--encoder-cache-tokens", type=int, default=defaults.encoder_cache_tokens
    )
    engine.add_argument(
        "--max-encoder-tokens-per-step",
        type=int,
        default=defaults.max_encoder_tokens_per_step,
    )
    engine.add_argument("--seed", type=int, default=defaults.seed)


def _server_options(parser: argparse.ArgumentParser) -> None:
    limits = ImageLimits()
    server = parser.add_argument_group("server")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument(
        "--served-model-name", help="the model id in the API (default: --model)"
    )
    server.add_argument("--max-images", type=int, default=limits.max_images)
    server.add_argument("--max-image-bytes", type=int, default=limits.max_bytes)
    server.add_argument("--max-image-pixels", type=int, default=limits.max_pixels)
    server.add_argument(
        "--prep-workers", type=int, default=4, help="threads that decode images"
    )


def _generation_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("generation")
    group.add_argument("--prompt", required=True)
    group.add_argument("--system", help="an optional system message")
    group.add_argument(
        "--image", action="append", default=[], type=Path, help="may be repeated"
    )
    group.add_argument("--max-tokens", type=int, default=256)
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--repetition-penalty", type=float, default=1.0)


def build_engine(args: argparse.Namespace) -> tuple[Engine, LoadedModel]:
    """Load the model onto the chosen device and put an engine around it."""
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if hint := build_hint():
        log.warning(hint)
    source = open_model(args.model, args.revision)
    max_len = args.max_model_len or min(source.context_length, DEFAULT_CONTEXT)
    config = EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        max_model_len=max_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        num_kv_blocks=args.num_kv_blocks,
        kv_memory_fraction=args.kv_memory_fraction,
        enable_prefix_caching=args.prefix_caching,
        max_encoder_tokens_per_step=args.max_encoder_tokens_per_step,
        encoder_cache_tokens=args.encoder_cache_tokens,
        seed=args.seed,
    )
    log.info("loading %s on %s as %s", args.model, describe(device), dtype)
    started = time.perf_counter()
    loaded = load_model(source, device, dtype, max_len)
    log.info("weights loaded in %.1fs", time.perf_counter() - started)
    frontend = loaded.frontend
    engine = Engine(
        config,
        loaded.decoder,
        loaded.vision,
        frontend.tokenizer,
        frontend.eos_token_ids,
    )
    log.info(
        "KV cache: %d blocks of %d tokens, context window %d",
        engine.num_blocks,
        config.block_size,
        max_len,
    )
    return engine, loaded


def run_serve(args: argparse.Namespace) -> int:
    engine, loaded = build_engine(args)
    limits = ImageLimits(
        max_images=args.max_images,
        max_bytes=args.max_image_bytes,
        max_pixels=args.max_image_pixels,
    )
    app = create_app(
        AsyncEngine(engine),
        loaded.frontend,
        args.served_model_name or args.model,
        limits,
        prep_workers=args.prep_workers,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def run_generate(args: argparse.Namespace) -> int:
    engine, loaded = build_engine(args)
    frontend = loaded.frontend
    limits = ImageLimits()
    parts: list[TextPart | ImagePart] = []
    for path in args.image:
        raw = path.read_bytes()
        image = open_image(raw, limits, frontend.draft_size)
        parts.append(ImagePart(image, image_key(raw)))
    parts.append(TextPart(args.prompt))
    turns = [Turn("user", parts)]
    if args.system:
        turns.insert(0, Turn("system", [TextPart(args.system)]))
    prompt = frontend.build_prompt(turns)

    room = engine.config.max_model_len - len(prompt.token_ids)
    params = SamplingParams(
        max_tokens=max(1, min(args.max_tokens, room)),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    engine.add_request("cli", prompt.token_ids, params, prompt.segments)
    started = time.perf_counter()
    last = None
    while engine.has_unfinished:
        for output in engine.step():
            sys.stdout.write(output.text)
            sys.stdout.flush()
            last = output
    elapsed = time.perf_counter() - started
    sys.stdout.write("\n")
    if last is not None:
        print(
            f"[{last.num_prompt_tokens} prompt tokens "
            f"({last.num_cached_tokens} cached), {last.num_generated_tokens} generated "
            f"in {elapsed:.2f}s, {last.num_generated_tokens / elapsed:.1f} tokens/s]",
            file=sys.stderr,
        )
    return 0


def run_devices(_: argparse.Namespace) -> int:
    print(f"torch {torch.__version__}")
    for name in accelerators():
        device = resolve_device(name)
        print(f"{describe(device)}, dtype auto: {resolve_dtype('auto', device)}")
    if hint := build_hint():
        print(f"note: {hint}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, getattr(args, "log_level", "info").upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.run(args))
    except (ValueError, OSError) as exc:
        print(f"vlmserve: error: {exc}", file=sys.stderr)
        return 2

"""Write a random llava-style checkpoint of a chosen size, to benchmark with.

The weights are noise, so the answers are noise. But the matrix shapes, the KV cache
per token and the number of image tokens are those of a real model of that size, so
the speed of the server is what it would be for one. Run it from the repository root:

    python -m benchmarks.random_model /tmp/small-vlm --preset small
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from tests.tinyhf import CLIP_LLAMA, TINY, Sizes, build_tiny_llava

from vlmserve.device import DTYPES

PRESETS = {
    "tiny": TINY,
    # about SmolVLM-256M: a 135M llama (SmolLM2) and a ViT-B tower, 196 tokens an image
    "small": Sizes(
        hidden=576,
        intermediate=1536,
        layers=30,
        heads=9,
        kv_heads=3,
        max_positions=2048,
        vocab_size=49152,
        init_range=0.02,
        vision_hidden=768,
        vision_intermediate=3072,
        vision_layers=12,
        vision_heads=12,
        image_size=224,
        patch=16,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("path", type=Path, help="where to write the model")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.path.mkdir(parents=True, exist_ok=True)
    build_tiny_llava(
        args.path, CLIP_LLAMA, args.seed, PRESETS[args.preset], DTYPES[args.dtype]
    )
    params = sum(p.numel() for p in _weights(args.path))
    print(f"wrote {args.preset} ({params / 1e6:.0f}M parameters) to {args.path}")
    return 0


def _weights(path: Path) -> list[torch.Tensor]:
    from safetensors.torch import load_file

    return list(load_file(path / "model.safetensors").values())


if __name__ == "__main__":
    sys.exit(main())

"""Write a random llava-style checkpoint of a chosen size.

The weights are noise, so the answers are noise. But the matrix shapes, the KV cache
per token and the number of image tokens are those of a real model of that size, so
the speed of the server is what it would be for one. From the repository root:

    python -m randomhf /tmp/small-vlm --preset small
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file

from randomhf.checkpoints import CLIP_LLAMA, PRESETS, build_tiny_llava
from vlmserve.device import DTYPES


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
    return list(load_file(path / "model.safetensors").values())


if __name__ == "__main__":
    sys.exit(main())

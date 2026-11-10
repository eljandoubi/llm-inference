from pathlib import Path

import pytest
import torch

from tests.tinyhf import (
    VARIANTS,
    TinyLlava,
    TinyText,
    Variant,
    build_tiny_llava,
    build_tiny_text,
)
from vlmserve.device import accelerators


@pytest.fixture(params=accelerators())
def device(request: pytest.FixtureRequest) -> torch.device:
    """Every device type this machine has, so numerics get checked on each of them."""
    return torch.device(request.param)


@pytest.fixture(scope="session", params=VARIANTS, ids=lambda v: v.name)
def tiny_llava(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> TinyLlava:
    """A tiny llava checkpoint of each kind, written once per test run."""
    variant: Variant = request.param
    path: Path = tmp_path_factory.mktemp(variant.name)
    return build_tiny_llava(path, variant)


@pytest.fixture(scope="session", params=VARIANTS, ids=lambda v: v.name)
def tiny_text(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> TinyText:
    """A tiny plain text checkpoint (llama or qwen2), written once per test run."""
    variant: Variant = request.param
    path: Path = tmp_path_factory.mktemp(f"text-{variant.name}")
    return build_tiny_text(path, variant)

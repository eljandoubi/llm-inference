import pytest
import torch

from vlmserve.device import accelerators


@pytest.fixture(params=accelerators())
def device(request: pytest.FixtureRequest) -> torch.device:
    """Every device type this machine has, so numerics get checked on each of them."""
    return torch.device(request.param)

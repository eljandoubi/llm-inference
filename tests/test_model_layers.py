import math

import pytest
import torch
import torch.nn.functional as F

from vlmserve.model.layers import MLP, RMSNorm, Rotary, apply_rope


def test_rmsnorm_matches_the_textbook_formula(device: torch.device) -> None:
    torch.manual_seed(0)
    norm = RMSNorm(16, eps=1e-6, device=device, dtype=torch.float32)
    norm.weight.data = torch.rand(16, device=device) + 0.5
    x = torch.randn(5, 16, device=device)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_keeps_the_dtype_in_half_precision(dtype: torch.dtype) -> None:
    norm = RMSNorm(16, eps=1e-6, device=torch.device("cpu"), dtype=dtype)
    x = torch.randn(3, 16).to(dtype)
    out = norm(x)
    assert out.dtype == dtype
    reference = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=2e-2)


def test_rotary_tables_live_on_the_device_in_float32(device: torch.device) -> None:
    rotary = Rotary(head_dim=8, theta=10000.0, max_positions=32, device=device)
    cos, sin = rotary(torch.tensor([0, 5, 31], device=device))
    assert cos.shape == sin.shape == (3, 4)
    assert cos.dtype == torch.float32
    assert cos.device.type == device.type


def test_position_zero_is_the_identity(device: torch.device) -> None:
    rotary = Rotary(8, 10000.0, 16, device)
    x = torch.randn(1, 2, 8, device=device)
    cos, sin = rotary(torch.tensor([0], device=device))
    torch.testing.assert_close(apply_rope(x, cos, sin), x)


def test_rope_pairs_each_element_with_the_one_half_a_head_away() -> None:
    rotary = Rotary(
        head_dim=4, theta=10000.0, max_positions=8, device=torch.device("cpu")
    )
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    cos, sin = rotary(torch.tensor([3]))
    out = apply_rope(x, cos, sin)[0, 0]
    f0, f1 = 3 * 1.0, 3 * 10000.0 ** (-2 / 4)  # position * inverse frequency
    expected = [
        1 * math.cos(f0) - 3 * math.sin(f0),
        2 * math.cos(f1) - 4 * math.sin(f1),
        3 * math.cos(f0) + 1 * math.sin(f0),
        4 * math.cos(f1) + 2 * math.sin(f1),
    ]
    torch.testing.assert_close(out, torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_attention_scores_only_depend_on_the_distance(device: torch.device) -> None:
    rotary = Rotary(16, 10000.0, 64, device)
    q = torch.randn(1, 1, 16, device=device)
    k = torch.randn(1, 1, 16, device=device)

    def score(m: int, n: int) -> torch.Tensor:
        cq, sq = rotary(torch.tensor([m], device=device))
        ck, sk = rotary(torch.tensor([n], device=device))
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum()

    torch.testing.assert_close(score(10, 4), score(40, 34), rtol=1e-4, atol=1e-4)
    assert not torch.allclose(score(10, 4), score(10, 5), atol=1e-4)


def test_mlp_is_swiglu(device: torch.device) -> None:
    torch.manual_seed(1)
    mlp = MLP(8, 12, device, torch.float32)
    x = torch.randn(4, 8, device=device)
    gate_w, up_w = mlp.gate_up_proj.weight.chunk(2, dim=0)
    expected = (F.silu(x @ gate_w.T) * (x @ up_w.T)) @ mlp.down_proj.weight.T
    torch.testing.assert_close(mlp(x), expected, rtol=1e-4, atol=1e-5)

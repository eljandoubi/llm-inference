import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Rotary(nn.Module):
    """Rotary embedding tables: a row of cos and sin for every absolute position."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self, head_dim: int, theta: float, max_positions: int, device: torch.device
    ) -> None:
        super().__init__()
        # built in float64 on the cpu: mps has no float64, and the tables are small
        inv_freq = theta ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        )
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float64), inv_freq)
        self.register_buffer("cos", angles.cos().float().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().float().to(device), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[positions], self.sin[positions]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x [tokens, heads, head_dim] by cos/sin [tokens, head_dim/2].

    Element i of a head is paired with element i + head_dim/2, the layout llama
    and qwen checkpoints were trained with.
    """
    half = x.shape[-1] // 2
    cos = cos[:, None, :].to(x.dtype)
    sin = sin[:, None, :].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class MLP(nn.Module):
    """SwiGLU with the gate and up projections fused into one matmul."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

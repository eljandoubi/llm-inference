"""Pick a device and a dtype so the rest of the code never checks for CUDA by hand."""

import os
import platform
import shutil
from dataclasses import dataclass

import psutil
import torch
import torch.nn.functional as F

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def accelerators() -> list[str]:
    """Device types this machine can run on, best first. "cpu" is always there."""
    found = [
        name
        for name, ok in (
            ("cuda", torch.cuda.is_available()),  # ROCm builds show up as cuda too
            ("mps", torch.backends.mps.is_available()),
            ("xpu", torch.xpu.is_available()),
        )
        if ok
    ]
    return [*found, "cpu"]


def resolve_device(spec: str = "auto") -> torch.device:
    available = accelerators()
    if spec == "auto":
        spec = available[0]
    try:
        device = torch.device(spec)
    except RuntimeError as exc:
        raise ValueError(f"unknown device {spec!r}") from exc
    if device.type not in available:
        raise ValueError(
            f"device {spec!r} is not available, "
            f"this machine has: {', '.join(available)}"
        )
    if device.type in {"cuda", "xpu"}:
        backend = torch.cuda if device.type == "cuda" else torch.xpu
        index = backend.current_device() if device.index is None else device.index
        if index >= backend.device_count():
            raise ValueError(
                f"{device.type}:{index} does not exist, "
                f"found {backend.device_count()} device(s)"
            )
        device = torch.device(device.type, index)
    return device


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """Try the two ops the engine leans on, the is_bf16_supported() flags lie."""
    if dtype == torch.float32:
        return True
    try:
        x = torch.ones(1, 1, 4, 8, device=device, dtype=dtype)
        attn = F.scaled_dot_product_attention(x, x, x)
        mm = x[0, 0] @ x[0, 0].T
        float(attn.float().sum() + mm.float().sum())
    except (RuntimeError, TypeError, NotImplementedError):
        return False
    return True


def _preferred_dtypes(device: torch.device) -> tuple[torch.dtype, ...]:
    match device.type:
        case "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            # older cards "support" bf16 by emulating it, which is slow
            return (torch.bfloat16, torch.float16) if major >= 8 else (torch.float16,)
        case "xpu":
            return (torch.bfloat16, torch.float16)
        case "mps":
            return (torch.float16,)
        case _:
            return (torch.float32,)


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "auto":
        return next(
            (d for d in _preferred_dtypes(device) if supports_dtype(device, d)),
            torch.float32,
        )
    if spec not in DTYPES:
        raise ValueError(
            f"unknown dtype {spec!r}, pick one of: auto, {', '.join(DTYPES)}"
        )
    dtype = DTYPES[spec]
    if not supports_dtype(device, dtype):
        raise ValueError(f"{spec} does not work on {device}")
    return dtype


@dataclass(frozen=True, slots=True)
class MemoryInfo:
    free: int
    total: int


def memory_info(device: torch.device) -> MemoryInfo:
    ram = psutil.virtual_memory()
    match device.type:
        case "cuda":
            free, total = torch.cuda.mem_get_info(device)
        case "xpu":
            free, total = torch.xpu.mem_get_info(device)
        case "mps":
            # unified memory: what the driver may hand out, capped by what the OS has
            total = torch.mps.recommended_max_memory()
            used = torch.mps.driver_allocated_memory()
            free = min(max(total - used, 0), ram.available)
        case _:
            free, total = ram.available, ram.total
    return MemoryInfo(free=free, total=total)


def synchronize(device: torch.device) -> None:
    match device.type:
        case "cuda":
            torch.cuda.synchronize(device)
        case "xpu":
            torch.xpu.synchronize(device)
        case "mps":
            torch.mps.synchronize()


def describe(device: torch.device) -> str:
    match device.type:
        case "cuda":
            name = torch.cuda.get_device_name(device)
        case "xpu":
            name = torch.xpu.get_device_name(device)
        case "mps":
            name = "Apple GPU"
        case _:
            name = f"{platform.machine()}, {os.cpu_count()} threads"
    return f"{device} ({name}, {memory_info(device).total / 2**30:.1f} GiB)"


def build_hint() -> str | None:
    """Catch the classic mistake of a CPU-only torch sitting next to an NVIDIA card."""
    cpu_build = torch.version.cuda is None and torch.version.hip is None
    if cpu_build and shutil.which("nvidia-smi"):
        return "nvidia-smi is there but this torch has no CUDA, install a CUDA build"
    return None

import shutil
from types import SimpleNamespace

import psutil
import pytest
import torch

from vlmserve import device as dev


def test_cpu_is_listed_last_and_once() -> None:
    names = dev.accelerators()
    assert names[-1] == "cpu"
    assert len(set(names)) == len(names)


def test_auto_picks_the_best_accelerator() -> None:
    assert dev.resolve_device("auto").type == dev.accelerators()[0]


def test_cpu_resolves() -> None:
    assert dev.resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("spec", ["gpu", "cuda:x", ""])
def test_garbage_spec_is_a_value_error(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown device"):
        dev.resolve_device(spec)


def test_missing_accelerator_says_what_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=r"not available.*cpu"):
        dev.resolve_device("cuda")


def test_cuda_index_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert dev.resolve_device("cuda") == torch.device("cuda", 1)
    assert dev.resolve_device("cuda:0") == torch.device("cuda", 0)
    with pytest.raises(ValueError, match=r"cuda:3 does not exist"):
        dev.resolve_device("cuda:3")


def test_auto_dtype_on_cpu_is_float32() -> None:
    assert dev.resolve_dtype("auto", torch.device("cpu")) == torch.float32


def test_unknown_dtype_lists_the_choices() -> None:
    with pytest.raises(ValueError, match=r"float64.*bfloat16"):
        dev.resolve_dtype("float64", torch.device("cpu"))


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 0), torch.bfloat16), ((9, 0), torch.bfloat16), ((7, 5), torch.float16)],
)
def test_auto_dtype_on_cuda_follows_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected: torch.dtype,
) -> None:
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(dev, "supports_dtype", lambda device, dtype: True)
    assert dev.resolve_dtype("auto", torch.device("cuda", 0)) == expected


def test_auto_dtype_falls_back_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dev, "supports_dtype", lambda device, dtype: dtype == torch.float32
    )
    assert dev.resolve_dtype("auto", torch.device("mps")) == torch.float32


def test_explicit_dtype_that_does_not_work_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dev, "supports_dtype", lambda device, dtype: False)
    with pytest.raises(ValueError, match="bfloat16 does not work"):
        dev.resolve_dtype("bfloat16", torch.device("cpu"))


def test_float32_always_passes_the_probe(device: torch.device) -> None:
    assert dev.supports_dtype(device, torch.float32)
    assert dev.resolve_dtype("float32", device) == torch.float32


def test_memory_info_is_sane(device: torch.device) -> None:
    info = dev.memory_info(device)
    assert 0 < info.free <= info.total


def test_cuda_memory_comes_from_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (3, 8))
    assert dev.memory_info(torch.device("cuda", 0)) == dev.MemoryInfo(free=3, total=8)


def test_mps_free_memory_is_capped_by_what_the_os_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 2**30
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10 * gib)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4 * gib)
    ram = SimpleNamespace(available=1 * gib, total=16 * gib)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: ram)
    info = dev.memory_info(torch.device("mps"))
    assert info == dev.MemoryInfo(free=1 * gib, total=10 * gib)
    ram.available = 8 * gib
    assert dev.memory_info(torch.device("mps")).free == 6 * gib


def test_synchronize_does_not_raise(device: torch.device) -> None:
    dev.synchronize(device)


def test_describe_starts_with_the_device(device: torch.device) -> None:
    text = dev.describe(device)
    assert text.startswith(str(device))
    assert "GiB" in text


def test_only_cuda_stages_through_pinned_memory() -> None:
    assert dev.uses_pinned_staging(torch.device("cuda", 0))
    assert not dev.uses_pinned_staging(torch.device("cpu"))
    assert not dev.uses_pinned_staging(torch.device("mps"))


@pytest.mark.parametrize(
    ("cuda", "hip", "nvidia_smi", "warns"),
    [
        (None, None, True, True),
        ("12.8", None, True, False),
        (None, "6.2", True, False),
        (None, None, False, False),
    ],
)
def test_build_hint_only_for_cpu_torch_next_to_an_nvidia_card(
    monkeypatch: pytest.MonkeyPatch,
    cuda: str | None,
    hip: str | None,
    nvidia_smi: bool,
    warns: bool,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/x" if nvidia_smi else None)
    assert (dev.build_hint() is not None) is warns

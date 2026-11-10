from pathlib import Path
from typing import Any

import pytest

from vlmserve.hf import hub
from vlmserve.hf.hub import PATTERNS, resolve_model_dir


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Stands in for the Hub: records what was asked and hands out a folder."""
    calls: list[dict[str, Any]] = []

    def fake(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(hub, "snapshot_download", fake)
    return calls


def test_a_local_directory_is_used_as_it_is(
    tmp_path: Path, downloads: list[dict[str, Any]]
) -> None:
    assert resolve_model_dir(str(tmp_path)) == tmp_path
    assert downloads == []


def test_a_repo_id_goes_to_the_hub_with_the_file_filter(
    downloads: list[dict[str, Any]], tmp_path: Path
) -> None:
    got = resolve_model_dir("org/model", revision="abc123")
    assert got == tmp_path / "snapshot"
    assert downloads == [
        {"repo_id": "org/model", "revision": "abc123", "allow_patterns": PATTERNS}
    ]


def test_weights_are_safetensors_only() -> None:
    assert "*.safetensors" in PATTERNS
    assert not any(p in PATTERNS for p in ("*.bin", "*.pt", "*.gguf", "*.onnx"))


@pytest.mark.parametrize("ref", ["./nope", "/nope/nothing", "~/nope", "a/b/c"])
def test_a_missing_path_is_not_mistaken_for_a_repo(
    ref: str, downloads: list[dict[str, Any]]
) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        resolve_model_dir(ref)
    assert downloads == []

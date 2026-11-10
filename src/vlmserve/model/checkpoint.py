import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from safetensors import safe_open

_EMBED = "embed_tokens.weight"
_HEAD = "lm_head.weight"


def find_text_prefix(names: Iterable[str]) -> str:
    """Where the text model's tensors start inside a (possibly multimodal) checkpoint.

    "model." for a plain llama, "language_model.model." or "model.language_model."
    for llava, "model.text_model." for idefics3. The embedding table gives it away.
    """
    prefixes = {n.removesuffix(_EMBED) for n in names if n.endswith(_EMBED)}
    if len(prefixes) != 1:
        raise ValueError(
            f"can't tell where the text model lives, candidates: {prefixes}"
        )
    return prefixes.pop()


class Checkpoint:
    """The safetensors files of a model directory, read lazily tensor by tensor."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._file_of: dict[str, Path] = {}
        for file in self._weight_files():
            with safe_open(file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118 (safe_open is no mapping)
                    self._file_of[name] = file

    @property
    def names(self) -> list[str]:
        return list(self._file_of)

    def items(self, names: Iterable[str]) -> Iterator[tuple[str, torch.Tensor]]:
        """The named tensors, on the cpu, reading each file once."""
        by_file: dict[Path, list[str]] = defaultdict(list)
        for name in names:
            by_file[self._file_of[name]].append(name)
        for file, wanted in by_file.items():
            with safe_open(file, framework="pt") as f:
                for name in wanted:
                    yield name, f.get_tensor(name)

    def text_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """The text model's tensors, named the way load_decoder_weights expects."""
        prefix = find_text_prefix(self._file_of)
        renamed = {
            n: n.removeprefix(prefix) for n in self._file_of if n.startswith(prefix)
        }
        heads = [n for n in self._file_of if n.endswith(_HEAD)]
        if len(heads) > 1:
            raise ValueError(f"more than one lm_head in the checkpoint: {heads}")
        renamed.update(dict.fromkeys(heads, _HEAD))
        for name, tensor in self.items(renamed):
            yield renamed[name], tensor

    def _weight_files(self) -> list[Path]:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"{self.directory} is not a directory")
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            return sorted(self.directory / s for s in shards)
        files = sorted(self.directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {self.directory}")
        return files

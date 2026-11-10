from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from vlmserve.detokenizer import Tokenizer
from vlmserve.request import ImageSegment


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    image: Image.Image
    key: str  # content hash, see multimodal.images.image_key


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    parts: Sequence[TextPart | ImagePart]


@dataclass(frozen=True, slots=True)
class PromptInputs:
    token_ids: list[int]
    segments: list[ImageSegment]


class Frontend(Protocol):
    """Everything model specific that happens before the engine: chat template,
    tokenizing, and turning images into placeholder tokens plus vision inputs."""

    @property
    def tokenizer(self) -> Tokenizer: ...

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    @property
    def draft_size(self) -> tuple[int, int] | None:
        """Roughly what images get shrunk to, so jpegs can be decoded smaller."""
        ...

    def build_prompt(self, turns: Sequence[Turn]) -> PromptInputs: ...

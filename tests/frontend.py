from collections.abc import Sequence

from tests.models import CharTokenizer
from vlmserve.frontend import ImagePart, PromptInputs, TextPart, Turn
from vlmserve.multimodal.preprocess import ImageTransform
from vlmserve.request import ImageSegment

IMAGE_TOKEN = 49


class TinyFrontend:
    """A chat template and an image path for the tiny model: 4 tokens per image."""

    tokenizer = CharTokenizer()
    eos_token_ids = frozenset({2})
    draft_size = (8, 8)

    def __init__(self) -> None:
        self.transform = ImageTransform(8, 8)
        self.seen: list[Sequence[Turn]] = []  # what build_prompt was given

    @staticmethod
    def encode(text: str) -> list[int]:
        return [3 + ord(c) % 45 for c in text]

    def build_prompt(self, turns: Sequence[Turn]) -> PromptInputs:
        self.seen.append(turns)
        ids: list[int] = []
        segments: list[ImageSegment] = []
        for turn in turns:
            ids += self.encode(f"{turn.role}: ")
            for part in turn.parts:
                if isinstance(part, TextPart):
                    ids += self.encode(part.text)
                elif isinstance(part, ImagePart):
                    pixels = {"pixel_values": self.transform(part.image)}
                    segments.append(ImageSegment(len(ids), 4, part.key, pixels))
                    ids += [IMAGE_TOKEN] * 4
            ids += self.encode("\n")
        ids += self.encode("assistant: ")
        return PromptInputs(ids, segments)

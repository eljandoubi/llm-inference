"""Chat template, tokenizer and image preprocessing of a Hugging Face model, put
behind the Frontend interface the server talks to."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizerBase

from vlmserve.frontend import ImagePart, PromptInputs, TextPart, Turn
from vlmserve.multimodal.images import ImageError
from vlmserve.request import ImageSegment


@dataclass(frozen=True, slots=True)
class ImageSupport:
    """What it takes to put images into a prompt."""

    processor: Any  # a transformers image processor: images in, pixel_values out
    token: str  # what the chat template writes for an image, "<image>" for llava
    token_id: int
    tokens_per_image: int
    size: int  # the processor produces size x size pixels

    def spans(self, ids: Sequence[int]) -> list[tuple[int, int]]:
        """(offset, length) of the placeholder tokens of each image in a prompt.

        A stretch of them can hold several images when the template doesn't put
        anything in between, every image takes exactly tokens_per_image of them.
        """
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for i, token in enumerate((*ids, None)):
            if token == self.token_id:
                start = i if start is None else start
                continue
            if start is not None:
                length = i - start
                if length % self.tokens_per_image:
                    raise ImageError(
                        f"{length} image tokens in a row, an image takes "
                        f"{self.tokens_per_image}"
                    )
                spans += [
                    (at, self.tokens_per_image)
                    for at in range(start, i, self.tokens_per_image)
                ]
                start = None
        return spans


class HFFrontend:
    """Everything between a chat request and the engine for a Hugging Face model.

    Safe to call from several threads: nothing here changes after construction.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chat: Any,  # whatever has apply_chat_template: the processor, or the tokenizer
        eos_token_ids: Collection[int],
        images: ImageSupport | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._chat = chat
        self._eos_token_ids = frozenset(eos_token_ids)
        self._images = images

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids

    @property
    def draft_size(self) -> tuple[int, int] | None:
        if self._images is None:
            return None
        return (self._images.size, self._images.size)

    def build_prompt(self, turns: Sequence[Turn]) -> PromptInputs:
        pictures = [p for turn in turns for p in turn.parts if isinstance(p, ImagePart)]
        if self._images is None and pictures:
            raise ImageError("this model does not take images")
        text = self._chat.apply_chat_template(
            [self._message(turn) for turn in turns],
            add_generation_prompt=True,
            tokenize=False,
        )
        if self._images is None:
            return PromptInputs(self._encode(text), [])
        return self._with_images(self._images, text, pictures)

    def _message(self, turn: Turn) -> dict[str, Any]:
        if self._images is None:  # text models want a plain string
            text = "".join(p.text for p in turn.parts if isinstance(p, TextPart))
            return {"role": turn.role, "content": text}
        content: list[dict[str, str]] = []
        for part in turn.parts:
            if isinstance(part, TextPart):
                if self._images.token in part.text:
                    raise ImageError(
                        f"{self._images.token!r} stands for an image, "
                        "it can't be part of the text"
                    )
                content.append({"type": "text", "text": part.text})
            else:
                content.append({"type": "image"})
        return {"role": turn.role, "content": content}

    def _with_images(
        self, images: ImageSupport, text: str, pictures: list[ImagePart]
    ) -> PromptInputs:
        pieces = text.split(images.token)
        if len(pieces) - 1 != len(pictures):
            raise ImageError(
                f"the chat template wrote {len(pieces) - 1} image placeholders "
                f"for {len(pictures)} images"
            )
        if not pictures:
            return PromptInputs(self._encode(text), [])
        pixels = images.processor(
            images=[p.image for p in pictures], return_tensors="pt"
        )["pixel_values"]
        # a placeholder in the template stands for all the tokens of its image
        ids = self._encode((images.token * images.tokens_per_image).join(pieces))
        spans = images.spans(ids)
        if len(spans) != len(pictures):
            raise ImageError(
                f"the prompt has room for {len(spans)} images, {len(pictures)} came"
            )
        segments = [
            ImageSegment(offset, length, part.key, {"pixel_values": pixels[i]})
            for i, ((offset, length), part) in enumerate(
                zip(spans, pictures, strict=True)
            )
        ]
        return PromptInputs(ids, segments)

    def _encode(self, text: str) -> list[int]:
        # templates that write the BOS themselves must not get a second one
        bos = self._tokenizer.bos_token
        has_bos = bool(bos) and text.startswith(bos)
        return list(self._tokenizer(text, add_special_tokens=not has_bos)["input_ids"])

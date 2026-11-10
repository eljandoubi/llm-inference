import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from tests.tinyhf import TOKENS_PER_IMAGE, TinyLlava, picture
from vlmserve.frontend import ImagePart, PromptInputs, TextPart, Turn
from vlmserve.hf.frontend import HFFrontend, ImageSupport
from vlmserve.hf.llava import llava_images
from vlmserve.multimodal.images import ImageError

PLAIN_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
)


def image(seed: int) -> ImagePart:
    return ImagePart(picture(seed), f"key-{seed}")


SINGLE = [Turn("user", [image(1), TextPart("what is this?")])]
MULTI = [
    Turn("system", [TextPart("be brief")]),
    Turn("user", [TextPart("look: "), image(1), TextPart("and "), image(2)]),
    Turn("assistant", [TextPart("they differ")]),
    Turn("user", [image(3), TextPart("and this?")]),
]
TEXT_ONLY = [Turn("user", [TextPart("hello there")])]


def frontend_of(tiny: TinyLlava) -> HFFrontend:
    processor = AutoProcessor.from_pretrained(tiny.path)
    config = AutoConfig.from_pretrained(tiny.path)
    tokenizer = processor.tokenizer
    return HFFrontend(
        tokenizer,
        processor,
        {tokenizer.eos_token_id},
        llava_images(processor, config),
    )


def reference(tiny: TinyLlava, turns: Sequence[Turn]) -> Any:
    """What transformers' own processor makes of the same conversation."""
    conversation = [
        {
            "role": turn.role,
            "content": [
                {"type": "image"}
                if isinstance(p, ImagePart)
                else {"type": "text", "text": p.text}
                for p in turn.parts
            ],
        }
        for turn in turns
    ]
    text = tiny.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    pictures = [p.image for t in turns for p in t.parts if isinstance(p, ImagePart)]
    return tiny.processor(images=pictures or None, text=text, return_tensors="pt")


@pytest.mark.parametrize(
    "turns", [SINGLE, MULTI, TEXT_ONLY], ids=["one", "many", "text"]
)
def test_the_prompt_is_what_the_transformers_processor_makes(
    tiny_llava: TinyLlava, turns: list[Turn]
) -> None:
    want = reference(tiny_llava, turns)
    got = frontend_of(tiny_llava).build_prompt(turns)
    assert got.token_ids == want["input_ids"][0].tolist()
    assert len(got.segments) == len(want.get("pixel_values", []))
    for segment, pixels in zip(got.segments, want.get("pixel_values", []), strict=True):
        assert (segment.inputs["pixel_values"] == pixels).all()


def test_segments_cover_exactly_the_placeholder_tokens(tiny_llava: TinyLlava) -> None:
    frontend = frontend_of(tiny_llava)
    got = frontend.build_prompt(MULTI)
    token_id = tiny_llava.model.config.image_token_index
    covered = {i for s in got.segments for i in range(s.offset, s.end)}
    assert covered == {i for i, t in enumerate(got.token_ids) if t == token_id}
    assert [s.length for s in got.segments] == [TOKENS_PER_IMAGE] * 3
    assert [s.key for s in got.segments] == ["key-1", "key-2", "key-3"]
    assert [s.offset for s in got.segments] == sorted(s.offset for s in got.segments)


def test_the_frontend_describes_the_model(tiny_llava: TinyLlava) -> None:
    frontend = frontend_of(tiny_llava)
    assert frontend.draft_size == (28, 28)
    assert frontend.eos_token_ids == {frontend.tokenizer.eos_token_id}
    assert frontend.tokenizer.decode(frontend.build_prompt(TEXT_ONLY).token_ids)


def test_the_image_token_can_not_be_typed_into_the_text(tiny_llava: TinyLlava) -> None:
    turns = [Turn("user", [image(1), TextPart("fake <image> here")])]
    with pytest.raises(ImageError, match="stands for an image"):
        frontend_of(tiny_llava).build_prompt(turns)


class Chat:
    """A chat template that always writes the same text, whatever it is asked."""

    def __init__(self, text: str) -> None:
        self.text = text

    def apply_chat_template(self, conversation: Any, **_: Any) -> str:
        return self.text


def with_chat(tiny: TinyLlava, chat: Chat) -> HFFrontend:
    real = frontend_of(tiny)
    config = AutoConfig.from_pretrained(tiny.path)
    processor = AutoProcessor.from_pretrained(tiny.path)
    return HFFrontend(
        real.tokenizer, chat, real.eos_token_ids, llava_images(processor, config)
    )


def test_a_template_that_drops_the_images_is_an_error(tiny_llava: TinyLlava) -> None:
    frontend = with_chat(tiny_llava, Chat("no placeholder at all"))
    with pytest.raises(ImageError, match="0 image placeholders for 1 images"):
        frontend.build_prompt(SINGLE)


def test_images_with_nothing_between_them_still_get_a_segment_each(
    tiny_llava: TinyLlava,
) -> None:
    frontend = with_chat(tiny_llava, Chat("<image><image>"))
    turns = [Turn("user", [image(1), image(2)])]
    got = frontend.build_prompt(turns)
    first = got.segments[0].offset
    assert [(s.offset, s.length) for s in got.segments] == [
        (first, TOKENS_PER_IMAGE),
        (first + TOKENS_PER_IMAGE, TOKENS_PER_IMAGE),
    ]
    assert [s.key for s in got.segments] == ["key-1", "key-2"]


def test_a_template_that_writes_the_bos_does_not_get_a_second_one(
    tiny_llava: TinyLlava,
) -> None:
    bos = tiny_llava.processor.tokenizer.bos_token
    if bos is None:
        pytest.skip("this tokenizer has no bos token")
    frontend = with_chat(tiny_llava, Chat(f"{bos}hello"))
    ids = frontend.build_prompt(TEXT_ONLY).token_ids
    assert ids.count(tiny_llava.processor.tokenizer.bos_token_id) == 1


def test_spans_split_a_long_stretch_between_the_images() -> None:
    support = ImageSupport(None, "<image>", 9, tokens_per_image=2, size=8)
    assert support.spans([1, 9, 9, 9, 9, 2, 9, 9]) == [(1, 2), (3, 2), (6, 2)]
    assert support.spans([1, 2, 3]) == []


def test_a_stretch_that_is_not_a_whole_number_of_images_is_an_error() -> None:
    support = ImageSupport(None, "<image>", 9, tokens_per_image=4, size=8)
    with pytest.raises(ImageError, match="3 image tokens in a row"):
        support.spans([1, 9, 9, 9, 2])


def test_a_prompt_with_the_wrong_number_of_image_runs_is_an_error(
    tiny_llava: TinyLlava,
) -> None:
    # "a" stands in for the image token, and "banana" has three of them
    real = frontend_of(tiny_llava)
    processor = AutoProcessor.from_pretrained(tiny_llava.path)
    letter_a = real.tokenizer.convert_tokens_to_ids("a")
    support = ImageSupport(
        processor.image_processor, "<image>", letter_a, tokens_per_image=1, size=28
    )
    frontend = HFFrontend(
        real.tokenizer, Chat("<image> banana"), real.eos_token_ids, support
    )
    with pytest.raises(ImageError, match="room for 3 images, 1 came"):
        frontend.build_prompt([Turn("user", [image(1)])])


@pytest.fixture
def text_frontend(tiny_llava: TinyLlava) -> HFFrontend:
    tokenizer = AutoTokenizer.from_pretrained(tiny_llava.path)
    tokenizer.chat_template = PLAIN_TEMPLATE
    return HFFrontend(tokenizer, tokenizer, {tokenizer.eos_token_id})


def test_a_text_model_gets_plain_string_messages(text_frontend: HFFrontend) -> None:
    turns = [Turn("user", [TextPart("hello "), TextPart("there")])]
    got = text_frontend.build_prompt(turns)
    assert got.segments == []
    text = text_frontend.tokenizer.decode(got.token_ids, skip_special_tokens=True)
    assert text == "user: hello there\nassistant: "
    assert text_frontend.draft_size is None


def test_a_text_model_refuses_images(text_frontend: HFFrontend) -> None:
    with pytest.raises(ImageError, match="does not take images"):
        text_frontend.build_prompt(SINGLE)


def test_several_threads_can_build_prompts_at_once(tiny_llava: TinyLlava) -> None:
    frontend = frontend_of(tiny_llava)
    expected = frontend.build_prompt(MULTI)
    start = threading.Barrier(8)

    def work(_: int) -> list[PromptInputs]:
        start.wait()
        return [frontend.build_prompt(MULTI) for _ in range(5)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for batch in pool.map(work, range(8)) for r in batch]
    assert len(results) == 40
    assert all(r.token_ids == expected.token_ids for r in results)
    assert all(
        [(s.offset, s.key) for s in r.segments]
        == [(s.offset, s.key) for s in expected.segments]
        for r in results
    )

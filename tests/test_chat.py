import base64
import io
from typing import Any

import pytest
from PIL import Image

from tests.frontend import IMAGE_TOKEN, TinyFrontend
from vlmserve.frontend import ImagePart, PromptInputs
from vlmserve.multimodal.images import (
    ImageError,
    ImageLimits,
    ImageSourceError,
    InvalidImageError,
    image_key,
)
from vlmserve.server.chat import count_images, prepare_prompt
from vlmserve.server.protocol import ChatCompletionRequest

LIMITS = ImageLimits(max_images=3)


def png(size: tuple[int, int] = (16, 16), color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def conversation(*messages: dict[str, Any]) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"messages": list(messages)})


def image_part(raw: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url(raw)}}


def prepare(
    request: ChatCompletionRequest, frontend: TinyFrontend | None = None
) -> PromptInputs:
    return prepare_prompt(request.messages, frontend or TinyFrontend(), LIMITS)


def test_a_text_conversation_has_no_segments() -> None:
    request = conversation({"role": "user", "content": "hi"})
    prompt = prepare(request)
    assert prompt.segments == []
    assert prompt.token_ids == TinyFrontend.encode("user: hi\nassistant: ")


def test_developer_messages_become_system_messages() -> None:
    frontend = TinyFrontend()
    prepare(conversation({"role": "developer", "content": "be brief"}), frontend)
    assert frontend.seen[0][0].role == "system"


def test_a_message_without_content_is_fine() -> None:
    frontend = TinyFrontend()
    prepare(conversation({"role": "assistant", "content": None}), frontend)
    assert frontend.seen[0][0].parts == []


def test_images_become_segments_where_they_appear() -> None:
    raw = png()
    request = conversation(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see "},
                image_part(raw),
                {"type": "text", "text": "!"},
            ],
        }
    )
    prompt = prepare(request)
    (segment,) = prompt.segments
    assert segment.key == image_key(raw)
    assert segment.length == 4
    assert prompt.token_ids[segment.offset : segment.end] == [IMAGE_TOKEN] * 4
    assert prompt.token_ids[: segment.offset] == TinyFrontend.encode("user: see ")
    assert segment.inputs["pixel_values"].shape == (3, 8, 8)


def test_images_keep_their_order_and_the_same_image_keeps_its_key() -> None:
    red, blue = png(color="red"), png(color="blue")
    content = [image_part(red), image_part(blue), image_part(red)]
    prompt = prepare(conversation({"role": "user", "content": content}))
    keys = [s.key for s in prompt.segments]
    assert keys == [image_key(red), image_key(blue), image_key(red)]
    assert keys[0] == keys[2] != keys[1]


def test_images_in_earlier_turns_are_found_too() -> None:
    raw = png()
    request = conversation(
        {"role": "user", "content": [image_part(raw)]},
        {"role": "assistant", "content": "a red square"},
        {"role": "user", "content": "and now?"},
    )
    assert count_images(request.messages) == 1
    assert len(prepare(request).segments) == 1


def test_too_many_images() -> None:
    content = [image_part(png())] * 4
    with pytest.raises(ImageError, match="at most 3 images"):
        prepare(conversation({"role": "user", "content": content}))


def test_remote_urls_are_refused() -> None:
    content = [
        {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}}
    ]
    with pytest.raises(ImageSourceError):
        prepare(conversation({"role": "user", "content": content}))


def test_broken_images_are_refused() -> None:
    bad = {"type": "image_url", "image_url": {"url": url(b"not an image")}}
    with pytest.raises(InvalidImageError):
        prepare(conversation({"role": "user", "content": [bad]}))


def test_big_jpegs_are_decoded_at_the_size_the_frontend_wants() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "green").save(buffer, format="JPEG")
    jpeg = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        },
    }
    frontend = TinyFrontend()
    prepare(conversation({"role": "user", "content": [jpeg]}), frontend)
    part = frontend.seen[0][0].parts[0]
    assert isinstance(part, ImagePart)
    assert part.image.width < 800  # drafted, not decoded in full
    assert part.image.width >= 8

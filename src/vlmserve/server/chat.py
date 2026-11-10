from collections.abc import Sequence

from vlmserve.frontend import Frontend, ImagePart, PromptInputs, TextPart, Turn
from vlmserve.multimodal.images import (
    ImageError,
    ImageLimits,
    decode_data_url,
    image_key,
    open_image,
)
from vlmserve.server.protocol import ChatMessage, ImageContent, TextContent


def count_images(messages: Sequence[ChatMessage]) -> int:
    return sum(
        isinstance(part, ImageContent)
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
    )


def prepare_prompt(
    messages: Sequence[ChatMessage], frontend: Frontend, limits: ImageLimits
) -> PromptInputs:
    """Decode the images of a conversation and let the frontend turn it into tokens.

    This does real work (base64, image decoding, tokenizing), so it belongs on a
    worker thread, not on the event loop.
    """
    if count_images(messages) > limits.max_images:
        raise ImageError(f"at most {limits.max_images} images per request")
    turns = []
    for message in messages:
        parts: list[TextPart | ImagePart] = []
        content = message.content
        if isinstance(content, str):
            parts.append(TextPart(content))
        for part in content if isinstance(content, list) else ():
            if isinstance(part, TextContent):
                parts.append(TextPart(part.text))
            else:
                raw = decode_data_url(part.image_url.url, limits)
                image = open_image(raw, limits, frontend.draft_size)
                parts.append(ImagePart(image, image_key(raw)))
        # openai renamed the system role to developer, models only know the old name
        role = "system" if message.role == "developer" else message.role
        turns.append(Turn(role, parts))
    return frontend.build_prompt(turns)

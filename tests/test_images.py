import base64
import io

import pytest
from PIL import Image

from vlmserve.multimodal.images import (
    ImageError,
    ImageLimits,
    ImageSourceError,
    ImageTooLargeError,
    InvalidImageError,
    decode_data_url,
    image_key,
    open_image,
)

LIMITS = ImageLimits()


def encode(size: tuple[int, int], fmt: str = "PNG", mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "red").save(buffer, format=fmt)
    return buffer.getvalue()


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def test_a_data_url_round_trips() -> None:
    raw = encode((8, 8))
    assert decode_data_url(data_url(raw), LIMITS) == raw


def test_wrapped_base64_is_tolerated() -> None:
    raw = encode((8, 8))
    b64 = base64.b64encode(raw).decode()
    wrapped = "\n".join(b64[i : i + 20] for i in range(0, len(b64), 20))
    assert decode_data_url(f"data:image/png;base64,{wrapped}", LIMITS) == raw


@pytest.mark.parametrize(
    "url",
    ["http://example.com/a.png", "https://x/y.jpg", "file:///etc/passwd", "a.png"],
)
def test_only_data_urls_are_accepted(url: str) -> None:
    with pytest.raises(ImageSourceError, match="data: URLs"):
        decode_data_url(url, LIMITS)


@pytest.mark.parametrize(
    "url",
    [
        "data:image/png;base64",  # no comma
        "data:image/png,AAAA",  # not base64
        "data:text/plain;base64,AAAA",  # not an image
        "data:image/png;base64,@@@@",  # not base64 at all
    ],
)
def test_malformed_data_urls_are_refused(url: str) -> None:
    with pytest.raises(InvalidImageError):
        decode_data_url(url, LIMITS)


def test_the_size_limit_is_checked_before_decoding() -> None:
    url = "data:image/png;base64," + "A" * 4000
    with pytest.raises(ImageTooLargeError):
        decode_data_url(url, ImageLimits(max_bytes=1000))


def test_the_size_limit_holds_for_the_decoded_bytes_too() -> None:
    raw = encode((64, 64))
    with pytest.raises(ImageTooLargeError):
        decode_data_url(data_url(raw), ImageLimits(max_bytes=len(raw) - 1))


def test_open_image_reads_png_and_jpeg() -> None:
    for fmt in ("PNG", "JPEG", "WEBP", "GIF", "BMP"):
        image = open_image(encode((20, 10), fmt), LIMITS)
        assert image.size == (20, 10)


def test_garbage_is_not_an_image() -> None:
    with pytest.raises(InvalidImageError, match="not an image"):
        open_image(b"definitely not an image", LIMITS)


def test_other_formats_are_refused() -> None:
    with pytest.raises(InvalidImageError, match="TIFF"):
        open_image(encode((8, 8), "TIFF"), LIMITS)


def test_too_many_pixels_is_refused_before_decoding() -> None:
    with pytest.raises(ImageTooLargeError, match="100x100"):
        open_image(encode((100, 100)), ImageLimits(max_pixels=9999))


def test_a_truncated_file_is_reported_as_damaged() -> None:
    raw = encode((64, 64), "PNG")
    with pytest.raises(ImageError):
        open_image(raw[: len(raw) // 2], LIMITS)


def test_exif_orientation_is_applied() -> None:
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    Image.new("RGB", (40, 20), "red").save(buffer, format="JPEG", exif=exif)
    assert open_image(buffer.getvalue(), LIMITS).size == (20, 40)


def test_a_big_jpeg_is_decoded_smaller_when_the_model_will_shrink_it_anyway() -> None:
    raw = encode((1600, 1200), "JPEG")
    drafted = open_image(raw, LIMITS, draft_to=(224, 224))
    assert drafted.width < 1600
    assert drafted.width >= 224
    assert drafted.height >= 224
    # png has no such shortcut
    assert open_image(encode((300, 200), "PNG"), LIMITS, draft_to=(8, 8)).size == (
        300,
        200,
    )


def test_keys_follow_the_content() -> None:
    raw = encode((8, 8))
    assert image_key(raw) == image_key(bytes(raw))
    assert image_key(raw) != image_key(encode((9, 8)))
    assert len(image_key(raw)) == 32

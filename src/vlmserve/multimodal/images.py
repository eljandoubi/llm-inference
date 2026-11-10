import base64
import binascii
import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "GIF", "BMP"})


class ImageError(ValueError):
    """The client sent an image we won't process."""


class ImageSourceError(ImageError):
    pass


class InvalidImageError(ImageError):
    pass


class ImageTooLargeError(ImageError):
    pass


@dataclass(frozen=True, slots=True)
class ImageLimits:
    max_images: int = 8  # per request
    max_bytes: int = 10 * 2**20
    max_pixels: int = 40_000_000


def decode_data_url(url: str, limits: ImageLimits) -> bytes:
    """Bytes of a base64 data: URL.

    Remote URLs are refused on purpose: fetching whatever address a client names
    from inside the server is a request forgery hole, so clients inline their images.
    """
    if not url.startswith("data:"):
        scheme = url.partition(":")[0] if ":" in url else "none"
        raise ImageSourceError(
            f"image URLs must be base64 data: URLs (got scheme {scheme!r}), "
            "the server does not fetch remote images"
        )
    header, comma, payload = url[len("data:") :].partition(",")
    if not comma:
        raise InvalidImageError("malformed data URL: no comma before the payload")
    mime, *flags = header.split(";")
    if "base64" not in flags:
        raise InvalidImageError("data URLs must be base64 encoded")
    if not mime.startswith("image/"):
        raise InvalidImageError(f"data URL holds {mime or 'no type'}, not an image")
    payload = "".join(payload.split())  # clients sometimes wrap the base64
    if len(payload) // 4 * 3 > limits.max_bytes + 3:
        raise ImageTooLargeError(f"image is over the {limits.max_bytes} byte limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("data URL is not valid base64") from exc
    if len(raw) > limits.max_bytes:
        raise ImageTooLargeError(f"image is over the {limits.max_bytes} byte limit")
    return raw


def open_image(
    raw: bytes, limits: ImageLimits, draft_to: tuple[int, int] | None = None
) -> Image.Image:
    """Decode, check and orient an image.

    draft_to is the size the model will shrink it to: a jpeg can then be decoded at
    1/2, 1/4 or 1/8 of its size, which is much cheaper for a phone photo.
    """
    try:
        image = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("image has far too many pixels") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("not an image we can read") from exc
    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError(f"image format {image.format} is not supported")
    if image.width * image.height > limits.max_pixels:
        raise ImageTooLargeError(
            f"image is {image.width}x{image.height}, over {limits.max_pixels} pixels"
        )
    if draft_to is not None and image.format == "JPEG":
        image.draft("RGB", draft_to)
    try:
        image.load()  # truncated files fail here, not later in the model
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("image data is damaged") from exc
    return ImageOps.exif_transpose(image)


def image_key(raw: bytes) -> str:
    """Content hash that identifies an image for the caches."""
    return hashlib.blake2b(raw, digest_size=16).hexdigest()

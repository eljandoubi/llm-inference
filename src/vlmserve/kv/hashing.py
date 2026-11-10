import struct
from array import array
from collections.abc import Sequence
from hashlib import blake2b

from vlmserve.request import ImageSegment

_FRAME = struct.Struct("<HII")


def hash_block(parent: bytes, token_ids: Sequence[int], extra: bytes = b"") -> bytes:
    """Digest of one block that also commits to everything before it (the parent)."""
    digest = blake2b(digest_size=16)
    # lengths first, so bytes can't slide from one field into the next
    digest.update(_FRAME.pack(len(parent), len(token_ids), len(extra)))
    digest.update(parent)
    digest.update(array("q", token_ids).tobytes())
    digest.update(extra)
    return digest.digest()


def extend_block_hashes(
    hashes: list[bytes],
    token_ids: Sequence[int],
    block_size: int,
    segments: Sequence[ImageSegment] = (),
    salt: bytes = b"",
) -> None:
    """Append a hash for every full block that doesn't have one yet.

    Placeholder tokens look the same whatever the image, so a block that overlaps an
    image also hashes which image it was and which part of it the block covers.
    """
    parent = hashes[-1] if hashes else salt
    first = 0  # first segment that can still reach the current block
    for i in range(len(hashes), len(token_ids) // block_size):
        start, stop = i * block_size, (i + 1) * block_size
        while first < len(segments) and segments[first].end <= start:
            first += 1
        extra = bytearray()
        j = first
        while j < len(segments) and segments[j].offset < stop:
            seg = segments[j]
            low, high = max(start, seg.offset), min(stop, seg.end)
            extra += f"{seg.key}:{low - seg.offset}:{high - seg.offset};".encode()
            j += 1
        parent = hash_block(parent, token_ids[start:stop], bytes(extra))
        hashes.append(parent)

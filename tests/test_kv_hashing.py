from collections.abc import Sequence

from vlmserve.kv.hashing import extend_block_hashes, hash_block
from vlmserve.request import ImageSegment

IMG = 9  # placeholder token id


def chain(
    tokens: Sequence[int],
    segments: Sequence[ImageSegment] = (),
    salt: bytes = b"",
    block_size: int = 4,
) -> list[bytes]:
    hashes: list[bytes] = []
    extend_block_hashes(hashes, tokens, block_size, segments, salt)
    return hashes


def test_one_hash_per_full_block() -> None:
    assert len(chain(range(10))) == 2
    assert len(chain(range(8))) == 2
    assert chain(range(3)) == []


def test_hashes_are_stable_and_16_bytes() -> None:
    first, second = chain(range(8)), chain(range(8))
    assert first == second
    assert all(len(h) == 16 for h in first)


def test_a_block_commits_to_its_whole_prefix() -> None:
    a = chain([1, 2, 3, 4, 5, 6, 7, 8])
    b = chain([0, 2, 3, 4, 5, 6, 7, 8])  # only block 0 differs
    assert a[0] != b[0]
    assert a[1] != b[1]


def test_same_tokens_in_another_position_hash_differently() -> None:
    h = chain([1, 2, 3, 4, 1, 2, 3, 4])
    assert h[0] != h[1]


def test_extending_in_steps_matches_hashing_in_one_go() -> None:
    tokens = list(range(13))
    hashes: list[bytes] = []
    extend_block_hashes(hashes, tokens[:6], 4)
    assert len(hashes) == 1
    extend_block_hashes(hashes, tokens, 4)
    assert hashes == chain(tokens)


def test_extending_with_nothing_new_changes_nothing() -> None:
    hashes = chain(range(8))
    before = list(hashes)
    extend_block_hashes(hashes, list(range(8)), 4)
    assert hashes == before


def test_different_images_split_the_cache_from_the_image_on() -> None:
    tokens = [1, 2, 3, 4, IMG, IMG, IMG, IMG, IMG, IMG, IMG, IMG, 5, 6, 7, 8]
    cat = chain(tokens, [ImageSegment(4, 8, "cat")])
    dog = chain(tokens, [ImageSegment(4, 8, "dog")])
    assert cat[0] == dog[0]  # the text before the image is shared
    assert cat[1] != dog[1]
    assert cat[2] != dog[2]
    assert cat[3] != dog[3]  # and so is everything that follows


def test_the_same_image_hashes_the_same() -> None:
    tokens = [1, 2, 3, 4, IMG, IMG, IMG, IMG, 5, 6, 7, 8]
    assert chain(tokens, [ImageSegment(4, 4, "cat")]) == chain(
        tokens, [ImageSegment(4, 4, "cat")]
    )


def test_the_part_of_the_image_a_block_covers_matters() -> None:
    tokens = [IMG] * 12
    shifted = [ImageSegment(1, 8, "x")]
    straight = [ImageSegment(0, 8, "x")]
    assert chain(tokens, shifted)[1] != chain(tokens, straight)[1]


def test_a_segment_that_misses_a_block_leaves_its_hash_alone() -> None:
    tokens = [1, 2, 3, 4, IMG, IMG, IMG, IMG]
    assert chain(tokens)[0] == chain(tokens, [ImageSegment(4, 4, "x")])[0]
    assert chain(tokens)[1] != chain(tokens, [ImageSegment(4, 4, "x")])[1]


def test_salt_keeps_users_apart() -> None:
    tokens = list(range(8))
    assert chain(tokens, salt=b"alice") != chain(tokens, salt=b"bob")
    assert chain(tokens, salt=b"alice") != chain(tokens)


def test_fields_cannot_bleed_into_each_other() -> None:
    assert hash_block(b"ab", [], b"c") != hash_block(b"a", [], b"bc")
    assert hash_block(b"", [1], b"") != hash_block(b"", [], b"\x01" + b"\x00" * 7)

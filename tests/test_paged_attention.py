import math
import random

import pytest
import torch

from vlmserve.model.attention import (
    SeqPlan,
    build_metadata,
    paged_attention,
    write_kv,
)

BLOCK = 4
HEAD_DIM = 8


def dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int
) -> torch.Tensor:
    """The slow, obvious version, with q at positions start.. and k, v up to them."""
    heads, kv_heads = q.shape[1], k.shape[1]
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    query_pos = torch.arange(start, start + q.shape[0])[:, None]
    key_pos = torch.arange(k.shape[0])[None, :]
    scores = scores.masked_fill(key_pos > query_pos, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def check_against_dense(
    device: torch.device,
    heads: int,
    kv_heads: int,
    layout: list[tuple[int, int]],
    block_size: int = BLOCK,
    seed: int = 0,
) -> None:
    """layout is one (already cached tokens, new tokens) pair per sequence."""
    gen = torch.Generator().manual_seed(seed)
    needed = [math.ceil((s + n) / block_size) for s, n in layout]
    num_blocks = 1 + sum(needed) + 2
    cache = torch.zeros(2, kv_heads, num_blocks, block_size, HEAD_DIM, device=device)
    free = list(range(1, num_blocks))
    random.Random(seed).shuffle(free)  # blocks are scattered, like after a long uptime

    plans, queries, new_k, new_v, full = [], [], [], [], []
    for (start, length), count in zip(layout, needed, strict=True):
        blocks = [free.pop() for _ in range(count)]
        k = torch.randn(start + length, kv_heads, HEAD_DIM, generator=gen)
        v = torch.randn(start + length, kv_heads, HEAD_DIM, generator=gen)
        if start:  # what earlier steps left behind
            pos = torch.arange(start)
            slots = (
                torch.tensor(blocks)[pos // block_size] * block_size + pos % block_size
            )
            write_kv(
                cache, slots.to(device), k[:start].to(device), v[:start].to(device)
            )
        plans.append(SeqPlan(start, length, blocks))
        queries.append(torch.randn(length, heads, HEAD_DIM, generator=gen))
        new_k.append(k[start:])
        new_v.append(v[start:])
        full.append((k, v))

    meta = build_metadata(plans, block_size, heads // kv_heads, device)
    write_kv(
        cache,
        meta.slot_mapping,
        torch.cat(new_k).to(device),
        torch.cat(new_v).to(device),
    )
    out = paged_attention(torch.cat(queries).to(device), cache, meta, HEAD_DIM**-0.5)

    row = 0
    for (start, length), q, (k, v) in zip(layout, queries, full, strict=True):
        got = out[row : row + length].view(length, heads, HEAD_DIM).cpu()
        torch.testing.assert_close(
            got, dense_attention(q, k, v, start), rtol=1e-4, atol=1e-5
        )
        row += length
    assert cache[:, :, 0].abs().sum() == 0, "the null block must stay empty"


HEADS = [(4, 4), (4, 2), (6, 1)]
LAYOUTS = {
    "decode": [(5, 1), (12, 1), (0, 1), (7, 1)],
    "prefill": [(0, 7), (0, 16)],
    "chunked": [(8, 5), (3, 9)],
    "mixed": [(10, 1), (0, 6), (4, 3), (20, 1)],
    "single": [(0, 1)],
}


@pytest.mark.parametrize(("heads", "kv_heads"), HEADS)
@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=LAYOUTS.keys())
def test_paged_attention_matches_dense_attention(
    device: torch.device, heads: int, kv_heads: int, layout: list[tuple[int, int]]
) -> None:
    check_against_dense(device, heads, kv_heads, layout)


@pytest.mark.parametrize("block_size", [1, 3, 16])
def test_any_block_size_works(device: torch.device, block_size: int) -> None:
    check_against_dense(device, 4, 2, LAYOUTS["mixed"], block_size=block_size)


def test_write_kv_lands_in_the_right_slots(device: torch.device) -> None:
    cache = torch.zeros(2, 2, 3, 4, HEAD_DIM, device=device)  # 12 slots
    key = torch.randn(3, 2, HEAD_DIM, device=device)
    value = torch.randn(3, 2, HEAD_DIM, device=device)
    slots = torch.tensor([5, 2, 9], device=device)
    write_kv(cache, slots, key, value)
    flat = cache.view(2, 2, 12, HEAD_DIM)
    for i, slot in enumerate([5, 2, 9]):
        torch.testing.assert_close(flat[0, :, slot], key[i])
        torch.testing.assert_close(flat[1, :, slot], value[i])
    untouched = [s for s in range(12) if s not in (5, 2, 9)]
    assert flat[:, :, untouched].abs().sum() == 0


def test_metadata_maps_tokens_to_slots_and_sorts_out_decodes() -> None:
    cpu = torch.device("cpu")
    plans = [SeqPlan(5, 1, [3, 7, 2]), SeqPlan(0, 3, [4])]
    meta = build_metadata(plans, block_size=4, group=2, device=cpu)
    # position 5 is offset 1 of the second block (7), the prompt fills block 4
    assert meta.slot_mapping.tolist() == [7 * 4 + 1, 16, 17, 18]
    assert meta.num_tokens == 4
    assert not meta.all_decode
    assert meta.decode is not None
    assert meta.decode.rows.tolist() == [0]
    assert meta.decode.block_table.tolist() == [[3, 7]]  # six tokens need two blocks
    assert meta.decode.mask.shape == (1, 1, 1, 8)
    assert meta.decode.mask.flatten().tolist() == [True] * 6 + [False] * 2
    (chunk,) = meta.prefills
    assert (chunk.start, chunk.length, chunk.kv_len) == (1, 3, 3)
    assert chunk.mask.shape == (3 * 2, 3)


def test_prefill_mask_is_causal_and_repeated_per_query_head_in_the_group() -> None:
    plans = [SeqPlan(2, 2, [1, 2])]  # two cached tokens, two new
    meta = build_metadata(plans, block_size=4, group=2, device=torch.device("cpu"))
    mask = meta.prefills[0].mask.int().tolist()
    # rows are (query, head-in-group): the first query sees 3 keys, the second 4
    assert mask == [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]


def test_decode_tables_are_padded_with_the_null_block() -> None:
    plans = [SeqPlan(2, 1, [5]), SeqPlan(9, 1, [6, 7, 8])]
    meta = build_metadata(plans, block_size=4, group=1, device=torch.device("cpu"))
    assert meta.all_decode
    assert meta.decode is not None
    assert meta.decode.block_table.tolist() == [[5, 0, 0], [6, 7, 8]]


def test_an_empty_step_is_a_bug() -> None:
    with pytest.raises(ValueError, match="no sequences"):
        build_metadata([], block_size=4, group=1, device=torch.device("cpu"))

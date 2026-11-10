"""Paged attention in plain torch, so it runs wherever torch does.

The cache of one layer is a tensor [2, kv_heads, num_blocks, block_size, head_dim]
(K and V). Putting the head dimension first means a gather over blocks lands in
[kv_heads, batch, tokens, head_dim], which is exactly what sdpa wants as batch dims,
so nothing has to be transposed or copied again after the gather.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class SeqPlan:
    """What one sequence does in this step, as plain host data."""

    start: int  # tokens that already have their KV in the cache
    length: int  # new tokens to run now
    block_ids: Sequence[int]  # covers at least start + length tokens


@dataclass(slots=True)
class DecodeBatch:
    """The sequences that run a single token, attended in one batched call."""

    rows: torch.Tensor  # [B] their index along the packed token dimension
    block_table: torch.Tensor  # [B, nb], padded with the null block
    mask: torch.Tensor  # [1, B, 1, nb * block_size], which slots each one may see


@dataclass(slots=True)
class PrefillChunk:
    """One sequence with several new tokens: a prompt, or a piece of one."""

    start: int  # first row along the packed token dimension
    length: int
    kv_len: int  # cached tokens plus the new ones
    blocks: torch.Tensor  # [nb]
    mask: torch.Tensor  # [length * group, kv_len], causal and aligned to the cache


@dataclass(slots=True)
class AttentionMetadata:
    """Everything the layers share in a step, built once instead of once per layer."""

    num_tokens: int
    slot_mapping: torch.Tensor  # [T], block * block_size + offset for every new token
    decode: DecodeBatch | None
    prefills: list[PrefillChunk]
    all_decode: bool  # every row is a decode row, in order


def build_metadata(
    plans: Sequence[SeqPlan], block_size: int, group: int, device: torch.device
) -> AttentionMetadata:
    """group is the number of query heads that share a kv head."""
    if not plans:
        raise ValueError("no sequences in this step")
    slots: list[np.ndarray] = []
    decode_rows: list[int] = []
    decode_tables: list[Sequence[int]] = []
    decode_lens: list[int] = []
    prefills: list[PrefillChunk] = []
    row = 0
    for plan in plans:
        positions = np.arange(plan.start, plan.start + plan.length)
        blocks = np.asarray(plan.block_ids, dtype=np.int64)
        slots.append(
            blocks[positions // block_size] * block_size + positions % block_size
        )
        kv_len = plan.start + plan.length
        needed = math.ceil(kv_len / block_size)
        if plan.length == 1:
            decode_rows.append(row)
            decode_tables.append(plan.block_ids[:needed])
            decode_lens.append(kv_len)
        else:
            query_pos = torch.arange(plan.start, kv_len, device=device)
            key_pos = torch.arange(kv_len, device=device)
            mask = (key_pos[None, :] <= query_pos[:, None]).repeat_interleave(group, 0)
            prefills.append(
                PrefillChunk(
                    start=row,
                    length=plan.length,
                    kv_len=kv_len,
                    blocks=torch.as_tensor(blocks[:needed], device=device),
                    mask=mask,
                )
            )
        row += plan.length

    decode = None
    if decode_rows:
        width = max(len(t) for t in decode_tables)
        table = np.zeros((len(decode_rows), width), dtype=np.int64)
        for i, blocks_i in enumerate(decode_tables):
            table[i, : len(blocks_i)] = blocks_i
        lengths = torch.as_tensor(decode_lens, device=device)
        visible = (
            torch.arange(width * block_size, device=device)[None, :] < lengths[:, None]
        )
        decode = DecodeBatch(
            rows=torch.as_tensor(decode_rows, device=device),
            block_table=torch.as_tensor(table, device=device),
            mask=visible[None, :, None, :],
        )
    return AttentionMetadata(
        num_tokens=row,
        slot_mapping=torch.as_tensor(np.concatenate(slots), device=device),
        decode=decode,
        prefills=prefills,
        all_decode=not prefills,
    )


def write_kv(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Store the new keys and values [T, kv_heads, head_dim] in their slots."""
    kv_heads, num_blocks, block_size, head_dim = kv_cache.shape[1:]
    flat = kv_cache.view(2, kv_heads, num_blocks * block_size, head_dim)
    flat[0].index_copy_(1, slot_mapping, key.transpose(0, 1))
    flat[1].index_copy_(1, slot_mapping, value.transpose(0, 1))


def paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    """Attend query [T, heads, head_dim] over the cache. Returns [T, heads * head_dim].

    The query heads that share a kv head are folded into the query length instead of
    repeating K and V, so the cache is never expanded.
    """
    num_tokens, heads, head_dim = query.shape
    kv_heads = kv_cache.shape[1]
    group = heads // kv_heads
    out = torch.empty(
        num_tokens, heads, head_dim, dtype=query.dtype, device=query.device
    )

    if meta.decode is not None:
        d = meta.decode
        batch = d.rows.shape[0]
        q = query if meta.all_decode else query[d.rows]
        q = q.view(batch, kv_heads, group, head_dim).transpose(0, 1)
        keys = kv_cache[0][:, d.block_table].reshape(kv_heads, batch, -1, head_dim)
        values = kv_cache[1][:, d.block_table].reshape(kv_heads, batch, -1, head_dim)
        attended = F.scaled_dot_product_attention(q, keys, values, d.mask, scale=scale)
        attended = attended.transpose(0, 1).reshape(batch, heads, head_dim)
        if meta.all_decode:
            out = attended
        else:
            out[d.rows] = attended

    for chunk in meta.prefills:
        rows = slice(chunk.start, chunk.start + chunk.length)
        q = query[rows].view(chunk.length, kv_heads, group, head_dim)
        q = q.transpose(0, 1).reshape(kv_heads, chunk.length * group, head_dim)
        keys = kv_cache[0][:, chunk.blocks].reshape(kv_heads, -1, head_dim)
        values = kv_cache[1][:, chunk.blocks].reshape(kv_heads, -1, head_dim)
        attended = F.scaled_dot_product_attention(
            q,
            keys[:, : chunk.kv_len],
            values[:, : chunk.kv_len],
            chunk.mask,
            scale=scale,
        )
        attended = attended.view(kv_heads, chunk.length, group, head_dim)
        out[rows] = attended.transpose(0, 1).reshape(chunk.length, heads, head_dim)

    return out.reshape(num_tokens, heads * head_dim)

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import torch

from vlmserve.model.attention import SeqPlan, build_metadata
from vlmserve.model.decoder import Decoder
from vlmserve.multimodal.encoder_cache import EncoderCache
from vlmserve.request import ImageSegment
from vlmserve.sampler import Sampler
from vlmserve.scheduler import Batch


class VisionEncoder(Protocol):
    """Turns preprocessed images into embeddings in the text model's space."""

    def encode(
        self, inputs: Sequence[Mapping[str, torch.Tensor]]
    ) -> list[torch.Tensor]:
        """One [num_tokens, hidden_size] tensor per input, on the decoder's device."""
        ...


class ModelRunner:
    """Runs one scheduled batch: embed (images included), decode, sample."""

    def __init__(
        self,
        decoder: Decoder,
        vision: VisionEncoder | None,
        block_size: int,
        sampler: Sampler,
        encoder_cache: EncoderCache | None = None,
    ) -> None:
        self.decoder = decoder
        self.vision = vision
        self.block_size = block_size
        self.sampler = sampler
        self.encoder_cache = encoder_cache
        self.group = decoder.config.num_heads // decoder.config.num_kv_heads
        self.images_encoded = 0  # tokens that went through the vision tower

    @torch.inference_mode()
    def execute(self, batch: Batch) -> list[int]:
        """Returns the sampled token of every item that reaches its last token."""
        items = batch.items
        self._prepare_images(batch)

        device = self.decoder.device
        ids = np.concatenate(
            [
                np.asarray(i.request.token_ids[i.start : i.start + i.num_tokens])
                for i in items
            ]
        )
        positions = np.concatenate(
            [np.arange(i.start, i.start + i.num_tokens) for i in items]
        )
        hidden = self.decoder.embed(torch.from_numpy(ids).to(device))
        self._splice_images(hidden, batch)

        plans = [SeqPlan(i.start, i.num_tokens, i.request.block_ids) for i in items]
        meta = build_metadata(plans, self.block_size, self.group, device)
        hidden = self.decoder(hidden, torch.from_numpy(positions).to(device), meta)

        last_rows: list[int] = []
        sampling = []
        row = 0
        for item in items:
            row += item.num_tokens
            if item.samples:
                last_rows.append(row - 1)
                sampling.append(item.request)
        if not sampling:
            return []
        logits = self.decoder.logits(hidden[torch.tensor(last_rows, device=device)])
        tokens = self.sampler(
            logits, [r.params for r in sampling], [r.token_ids for r in sampling]
        )
        return [int(t) for t in tokens.tolist()]

    def _prepare_images(self, batch: Batch) -> None:
        """Make sure every request has the embeddings of the images in its chunk."""
        fresh: dict[str, torch.Tensor] = {}
        if batch.to_encode:
            for seg, embedding in zip(
                batch.to_encode, self._embed(batch.to_encode), strict=True
            ):
                fresh[seg.key] = embedding
        for item in batch.items:
            req = item.request
            for seg in _overlapping(
                req.segments, item.start, item.start + item.num_tokens
            ):
                if seg.key in req.embeddings:
                    continue
                emb = fresh.get(seg.key)
                if emb is None and self.encoder_cache is not None:
                    emb = self.encoder_cache.get(seg.key)
                if emb is None:  # evicted since it was planned, do it again
                    (emb,) = self._embed([seg])
                    fresh[seg.key] = emb
                req.embeddings[seg.key] = emb

    def _embed(self, segments: Sequence[ImageSegment]) -> list[torch.Tensor]:
        if self.vision is None:
            raise RuntimeError("the batch has images but the model has no vision tower")
        embeddings = self.vision.encode([seg.inputs for seg in segments])
        for seg, emb in zip(segments, embeddings, strict=True):
            if emb.shape[0] != seg.length:
                raise RuntimeError(
                    f"image {seg.key} has {seg.length} placeholder tokens "
                    f"but the vision tower produced {emb.shape[0]} embeddings"
                )
            self.images_encoded += seg.length
            if self.encoder_cache is not None:
                self.encoder_cache.put(seg.key, emb)
        return embeddings

    @staticmethod
    def _splice_images(hidden: torch.Tensor, batch: Batch) -> None:
        """Overwrite the placeholder rows with the image embeddings."""
        row = 0
        for item in batch.items:
            low, high = item.start, item.start + item.num_tokens
            for seg in _overlapping(item.request.segments, low, high):
                a, b = max(low, seg.offset), min(high, seg.end)
                source = item.request.embeddings[seg.key]
                hidden[row + a - low : row + b - low] = source[
                    a - seg.offset : b - seg.offset
                ]
            row += item.num_tokens


def _overlapping(
    segments: Sequence[ImageSegment], low: int, high: int
) -> list[ImageSegment]:
    return [s for s in segments if s.end > low and s.offset < high]

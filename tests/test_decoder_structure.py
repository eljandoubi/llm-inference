import pytest
import torch

from tests.models import attention_of, tiny_config
from vlmserve.model.attention import SeqPlan, build_metadata
from vlmserve.model.decoder import Decoder


def test_a_step_without_a_cache_is_refused(device: torch.device) -> None:
    decoder = Decoder(tiny_config(), 32, device, torch.float32)
    meta = build_metadata([SeqPlan(0, 3, [1])], 4, 2, device)
    positions = torch.arange(3, device=device)
    hidden = decoder.embed(torch.tensor([1, 2, 3], device=device))
    with pytest.raises(RuntimeError, match="allocate_kv_cache"):
        decoder(hidden, positions, meta)


def test_a_step_produces_logits_for_every_row(device: torch.device) -> None:
    decoder = Decoder(tiny_config(), 32, device, torch.float32)
    decoder.allocate_kv_cache(num_blocks=4, block_size=4)
    meta = build_metadata([SeqPlan(0, 3, [1]), SeqPlan(0, 2, [2])], 4, 2, device)
    ids = torch.tensor([1, 2, 3, 4, 5], device=device)
    positions = torch.tensor([0, 1, 2, 0, 1], device=device)
    hidden = decoder(decoder.embed(ids), positions, meta)
    assert hidden.shape == (5, 32)
    assert decoder.logits(hidden).shape == (5, 50)


def test_every_layer_gets_its_own_slice_of_one_allocation(device: torch.device) -> None:
    decoder = Decoder(tiny_config(), 32, device, torch.float32)
    cache = decoder.allocate_kv_cache(num_blocks=4, block_size=4)
    assert cache.shape == (2, 2, 2, 4, 4, 8)  # layers, K/V, kv heads, blocks, size, dim
    first, second = attention_of(decoder, 0).kv_cache, attention_of(decoder, 1).kv_cache
    assert first is not None
    assert second is not None
    first.fill_(1.0)
    assert second.abs().sum() == 0
    assert cache[0].min() == 1.0


def test_tied_embeddings_have_no_separate_head(device: torch.device) -> None:
    tied = Decoder(tiny_config(tie_word_embeddings=True), 32, device, torch.float32)
    assert tied.lm_head is None
    hidden = torch.randn(2, 32, device=device)
    torch.testing.assert_close(tied.logits(hidden), hidden @ tied.embed_tokens.weight.T)
    untied = Decoder(tiny_config(), 32, device, torch.float32)
    assert untied.lm_head is not None


def test_qkv_bias_only_when_the_config_says_so(device: torch.device) -> None:
    plain = Decoder(tiny_config(), 32, device, torch.float32)
    biased = Decoder(tiny_config(qkv_bias=True), 32, device, torch.float32)
    assert attention_of(plain, 0).qkv_proj.bias is None
    assert attention_of(biased, 0).qkv_proj.bias is not None

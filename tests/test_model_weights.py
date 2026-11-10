import pytest
import torch

from tests.dense import random_weights
from tests.models import TINY, attention_of, layer_of, tiny_config
from vlmserve.model.decoder import Decoder
from vlmserve.model.weights import load_decoder_weights


def fresh(config_overrides: dict[str, object] | None = None) -> Decoder:
    config = tiny_config(**(config_overrides or {}))
    return Decoder(config, 32, torch.device("cpu"), torch.float32)


def test_q_k_v_and_gate_up_are_fused_in_that_order() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    load_decoder_weights(decoder, weights.items())
    layer = layer_of(decoder, 1)
    qkv = layer.self_attn.qkv_proj.weight
    expected = torch.cat(
        [weights[f"layers.1.self_attn.{n}_proj.weight"] for n in "qkv"]
    )
    torch.testing.assert_close(qkv, expected)
    gate_up = layer.mlp.gate_up_proj.weight
    expected = torch.cat(
        [weights[f"layers.1.mlp.{n}_proj.weight"] for n in ("gate", "up")]
    )
    torch.testing.assert_close(gate_up, expected)


def test_biases_are_fused_too() -> None:
    decoder = fresh({"qkv_bias": True})
    weights = random_weights(decoder.config)
    load_decoder_weights(decoder, weights.items())
    bias = attention_of(decoder, 0).qkv_proj.bias
    expected = torch.cat([weights[f"layers.0.self_attn.{n}_proj.bias"] for n in "qkv"])
    torch.testing.assert_close(bias, expected)


def test_whatever_has_a_one_to_one_name_is_copied_as_is() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    load_decoder_weights(decoder, weights.items())
    torch.testing.assert_close(
        decoder.embed_tokens.weight, weights["embed_tokens.weight"]
    )
    torch.testing.assert_close(decoder.norm.weight, weights["norm.weight"])
    assert decoder.lm_head is not None
    torch.testing.assert_close(decoder.lm_head.weight, weights["lm_head.weight"])


def test_a_stream_of_tensors_works() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    ignored = load_decoder_weights(decoder, iter(weights.items()))
    assert ignored == []


def test_a_missing_tensor_is_named() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    del weights["layers.1.mlp.down_proj.weight"]
    with pytest.raises(ValueError, match=r"1 weights were never loaded.*down_proj"):
        load_decoder_weights(decoder, weights.items())


def test_a_half_filled_fused_weight_counts_as_missing() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    del weights["layers.0.self_attn.k_proj.weight"]
    with pytest.raises(ValueError, match="qkv_proj"):
        load_decoder_weights(decoder, weights.items())


def test_a_wrong_shape_is_refused() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    weights["layers.0.self_attn.v_proj.weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match=r"v_proj.*\(3, 3\)"):
        load_decoder_weights(decoder, weights.items())


def test_tied_models_ignore_the_checkpoints_lm_head() -> None:
    decoder = fresh({"tie_word_embeddings": True})
    weights = random_weights(TINY)  # untied, so it carries an lm_head
    weights["layers.0.self_attn.rotary_emb.inv_freq"] = torch.zeros(4)
    ignored = load_decoder_weights(decoder, weights.items())
    assert sorted(ignored) == [
        "layers.0.self_attn.rotary_emb.inv_freq",
        "lm_head.weight",
    ]


def test_an_untied_model_needs_its_lm_head() -> None:
    decoder = fresh()
    weights = random_weights(decoder.config)
    del weights["lm_head.weight"]
    with pytest.raises(ValueError, match="lm_head"):
        load_decoder_weights(decoder, weights.items())


def test_weights_are_converted_to_the_model_dtype() -> None:
    decoder = Decoder(tiny_config(), 32, torch.device("cpu"), torch.float16)
    weights = random_weights(decoder.config)
    load_decoder_weights(decoder, weights.items())
    assert decoder.norm.weight.dtype == torch.float16
    torch.testing.assert_close(
        decoder.norm.weight.float(), weights["norm.weight"], rtol=1e-2, atol=1e-2
    )

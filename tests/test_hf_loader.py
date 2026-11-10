import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.tinyhf import (
    MULTI,
    SINGLE,
    TEXT_ONLY,
    TinyLlava,
    TinyText,
    hf_continuation,
    image,
)
from vlmserve.config import EngineConfig
from vlmserve.device import resolve_dtype
from vlmserve.engine import Engine
from vlmserve.frontend import TextPart, Turn
from vlmserve.hf.llava import LlavaVision
from vlmserve.hf.loader import LoadedModel, eos_token_ids, load_model, open_model
from vlmserve.request import Request
from vlmserve.sampling import SamplingParams

STEPS = 8
GREEDY = SamplingParams(temperature=0, max_tokens=STEPS)


def engine_of(loaded: LoadedModel, **overrides: Any) -> Engine:
    config = EngineConfig(
        "tiny",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        block_size=4,
        num_kv_blocks=120,
        **overrides,
    )
    frontend = loaded.frontend
    return Engine(
        config,
        loaded.decoder,
        loaded.vision,
        frontend.tokenizer,
        frontend.eos_token_ids,
    )


def start(engine: Engine, loaded: LoadedModel, name: str, turns: list[Turn]) -> Request:
    prompt = loaded.frontend.build_prompt(turns)
    engine.add_request(name, prompt.token_ids, GREEDY, prompt.segments)
    return engine.scheduler.waiting[-1]  # the engine keeps its own Request object


def run(engine: Engine) -> None:
    for _ in range(500):
        if not engine.has_unfinished:
            return
        engine.step()
    raise AssertionError("the engine did not finish")


def loaded_llava(tiny: TinyLlava, device: torch.device) -> LoadedModel:
    source = open_model(str(tiny.path))
    return load_model(source, device, torch.float32, max_model_len=128)


def test_a_llava_checkpoint_is_read_the_way_it_is_configured(
    tiny_llava: TinyLlava,
) -> None:
    source = open_model(str(tiny_llava.path))
    assert source.multimodal
    qwen = tiny_llava.variant.text == "qwen2"
    assert source.decoder.hidden_size == 32
    assert source.decoder.num_kv_heads == 2
    assert source.decoder.qkv_bias is qwen
    assert source.decoder.tie_word_embeddings is qwen
    assert source.context_length == 128


@pytest.mark.parametrize(
    "turns", [SINGLE, MULTI, TEXT_ONLY], ids=["one", "many", "text"]
)
def test_llava_answers_like_transformers(
    tiny_llava: TinyLlava, device: torch.device, turns: list[Turn]
) -> None:
    loaded = loaded_llava(tiny_llava, device)
    engine = engine_of(loaded)
    req = start(engine, loaded, "a", turns)
    run(engine)
    assert req.output_token_ids == hf_continuation(tiny_llava, turns)


def test_requests_in_one_batch_each_answer_like_transformers(
    tiny_llava: TinyLlava, device: torch.device
) -> None:
    """Different images, chunked prefill, and a repeat that comes out of the cache."""
    loaded = loaded_llava(tiny_llava, device)
    engine = engine_of(loaded)
    other = [Turn("user", [image(7), TextPart("what is this?")])]
    cases = {"many": MULTI, "one": SINGLE, "other": other, "text": TEXT_ONLY}
    requests = {n: start(engine, loaded, n, t) for n, t in cases.items()}
    run(engine)
    again = start(engine, loaded, "again", SINGLE)
    run(engine)
    for name, turns in cases.items():
        assert requests[name].output_token_ids == hf_continuation(tiny_llava, turns)
    assert again.output_token_ids == requests["one"].output_token_ids
    assert again.num_cached_tokens > 0


def test_half_precision_loads_and_runs(tiny_llava: TinyLlava) -> None:
    source = open_model(str(tiny_llava.path))
    loaded = load_model(source, torch.device("cpu"), torch.bfloat16, max_model_len=128)
    assert loaded.decoder.dtype == torch.bfloat16
    engine = engine_of(loaded)
    req = start(engine, loaded, "a", SINGLE)
    run(engine)
    assert len(req.output_token_ids) == STEPS


def test_every_device_runs_in_the_dtype_it_picks_for_itself(
    tiny_llava: TinyLlava, device: torch.device
) -> None:
    dtype = resolve_dtype("auto", device)
    loaded = load_model(
        open_model(str(tiny_llava.path)), device, dtype, max_model_len=128
    )
    assert loaded.decoder.dtype == dtype
    vision = loaded.vision
    assert isinstance(vision, LlavaVision)
    assert next(vision.tower.parameters()).dtype == dtype
    engine = engine_of(loaded)
    req = start(engine, loaded, "a", MULTI)
    run(engine)
    assert len(req.output_token_ids) == STEPS


def test_the_frontend_knows_what_ends_an_answer(tiny_llava: TinyLlava) -> None:
    loaded = loaded_llava(tiny_llava, torch.device("cpu"))
    assert loaded.frontend.eos_token_ids == {loaded.frontend.tokenizer.eos_token_id}


def test_a_plain_text_model_loads_without_a_vision_tower(
    tiny_text: TinyText, device: torch.device
) -> None:
    source = open_model(str(tiny_text.path))
    assert not source.multimodal
    loaded = load_model(source, device, torch.float32, max_model_len=128)
    assert loaded.vision is None
    assert loaded.frontend.draft_size is None


def test_a_plain_text_model_answers_like_transformers(
    tiny_text: TinyText, device: torch.device
) -> None:
    source = open_model(str(tiny_text.path))
    loaded = load_model(source, device, torch.float32, max_model_len=128)
    engine = engine_of(loaded)
    turns = [Turn("system", [TextPart("be brief")]), Turn("user", [TextPart("hi")])]
    req = start(engine, loaded, "a", turns)
    run(engine)
    prompt = torch.tensor([req.prompt_token_ids])
    expected = tiny_text.model.generate(prompt, do_sample=False, max_new_tokens=STEPS)
    assert req.output_token_ids == expected[0, prompt.shape[1] :].tolist()


def test_a_context_longer_than_the_model_was_trained_for_is_refused(
    tiny_llava: TinyLlava,
) -> None:
    source = open_model(str(tiny_llava.path))
    with pytest.raises(ValueError, match="trained for 128 tokens"):
        load_model(source, torch.device("cpu"), torch.float32, max_model_len=129)


def test_other_architectures_are_named_in_the_error(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    with pytest.raises(ValueError, match=r"'mistral' is not supported, only llava"):
        open_model(str(tmp_path))


def test_a_text_model_inside_llava_has_to_be_one_we_know(tmp_path: Path) -> None:
    config = {"model_type": "llava", "text_config": {"model_type": "gemma"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="text model 'gemma' is not supported"):
        open_model(str(tmp_path))


def test_a_missing_folder_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        open_model(str(tmp_path / "nope"))


@pytest.mark.parametrize(
    ("generation", "config", "tokenizer", "expected"),
    [
        ({"eos_token_id": [7, 8]}, 10, 9, {7, 8, 9, 10}),
        ({"eos_token_id": 7}, 7, 7, {7}),
        ({}, 2, None, {2}),
        (None, None, 5, {5}),
        (None, None, None, set()),
    ],
)
def test_every_source_of_end_of_sequence_ids_counts(
    tmp_path: Path,
    generation: dict[str, Any] | None,
    config: int | None,
    tokenizer: int | None,
    expected: set[int],
) -> None:
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    got = eos_token_ids(
        tmp_path,
        SimpleNamespace(eos_token_id=tokenizer),
        SimpleNamespace(eos_token_id=config),
    )
    assert got == expected

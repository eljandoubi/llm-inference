import pytest
import torch

from tests.dense import dense_greedy
from tests.models import TinyVision, image_segment
from tests.rig import Rig, joined
from vlmserve.request import FinishReason, ImageSegment
from vlmserve.sampling import SamplingParams

IMG = 49
GREEDY = SamplingParams(temperature=0, max_tokens=8)
PROMPT = [5, 6, 7, 8, 9, 10]


def test_generation_matches_the_dense_reference(device: torch.device) -> None:
    rig = Rig(device)
    rig.engine.add_request("a", PROMPT, SamplingParams(temperature=0, max_tokens=8))
    outputs = rig.run()["a"]
    assert joined(outputs) == rig.text(rig.greedy(PROMPT, 8))
    assert outputs[-1].finished
    assert outputs[-1].finish_reason == FinishReason.LENGTH
    assert outputs[-1].num_generated_tokens == 8
    assert outputs[-1].num_prompt_tokens == 6


def test_text_is_streamed_as_deltas(device: torch.device) -> None:
    rig = Rig(device)
    rig.engine.add_request("a", PROMPT, GREEDY)
    outputs = rig.run()["a"]
    assert len(outputs) == 8  # one per token
    assert [o.finished for o in outputs] == [False] * 7 + [True]
    assert all(len(o.text) == 1 for o in outputs)


def test_an_eos_token_ends_the_request_and_leaves_no_text(device: torch.device) -> None:
    rig = Rig(device)
    expected = rig.greedy(PROMPT, 8)
    eos = expected[2]
    count = expected.index(eos) + 1  # a repeated token would stop it earlier
    rig = Rig(device, eos={eos})
    rig.engine.add_request("a", PROMPT, GREEDY)
    outputs = rig.run()["a"]
    assert outputs[-1].finish_reason == FinishReason.STOP
    assert outputs[-1].num_generated_tokens == count  # the eos counts as generated
    assert joined(outputs) == rig.text(expected[: count - 1])


def test_ignore_eos_keeps_going(device: torch.device) -> None:
    expected = Rig(device).greedy(PROMPT, 8)
    rig = Rig(device, eos={expected[2]})
    rig.engine.add_request(
        "a", PROMPT, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    )
    outputs = rig.run()["a"]
    assert outputs[-1].num_generated_tokens == 8


def test_a_stop_token_id_works_like_eos(device: torch.device) -> None:
    expected = Rig(device).greedy(PROMPT, 8)
    rig = Rig(device)
    stop_id = expected[4]
    params = SamplingParams(temperature=0, max_tokens=8, stop_token_ids=(stop_id,))
    rig.engine.add_request("a", PROMPT, params)
    outputs = rig.run()["a"]
    assert outputs[-1].finish_reason == FinishReason.STOP
    assert outputs[-1].num_generated_tokens == expected.index(stop_id) + 1


def test_a_stop_string_cuts_the_text_in_front_of_it(device: torch.device) -> None:
    rig = Rig(device)
    full = rig.text(rig.greedy(PROMPT, 8))
    stop = full[4:6]
    params = SamplingParams(temperature=0, max_tokens=8, stop=(stop,))
    rig.engine.add_request("a", PROMPT, params)
    outputs = rig.run()["a"]
    assert joined(outputs) == full[: full.find(stop)]
    assert outputs[-1].finish_reason == FinishReason.STOP
    assert outputs[-1].num_generated_tokens < 8 or full.find(stop) + len(stop) == 8


def test_a_stop_string_that_never_shows_up_changes_nothing(
    device: torch.device,
) -> None:
    rig = Rig(device)
    full = rig.text(rig.greedy(PROMPT, 8))
    params = SamplingParams(temperature=0, max_tokens=8, stop=("0123",))
    rig.engine.add_request("a", PROMPT, params)
    assert joined(rig.run()["a"]) == full


def test_held_back_text_is_released_when_the_request_runs_out_of_tokens(
    device: torch.device,
) -> None:
    rig = Rig(device)
    full = rig.text(rig.greedy(PROMPT, 8))
    # the stop string begins with the last character, so that one is held back
    params = SamplingParams(temperature=0, max_tokens=8, stop=(full[-1] + "#",))
    rig.engine.add_request("a", PROMPT, params)
    outputs = rig.run()["a"]
    assert joined(outputs) == full
    assert outputs[-1].finish_reason == FinishReason.LENGTH


def test_requests_in_one_batch_do_not_disturb_each_other(device: torch.device) -> None:
    rig = Rig(device)
    prompts = {"a": [5, 6, 7], "b": [8, 9, 10, 11, 12], "c": [13, 14]}
    for name, prompt in prompts.items():
        rig.engine.add_request(name, prompt, GREEDY)
    seen = rig.run()
    for name, prompt in prompts.items():
        assert joined(seen[name]) == rig.text(rig.greedy(prompt, 8))


def test_the_prefix_cache_serves_a_repeated_prompt(device: torch.device) -> None:
    rig = Rig(device)
    rig.engine.add_request("a", PROMPT * 2, GREEDY)
    first = rig.run()["a"]
    rig.engine.add_request("b", PROMPT * 2, GREEDY)
    second = rig.run()["b"]
    assert first[-1].num_cached_tokens == 0
    assert second[-1].num_cached_tokens == 8  # 12 tokens, two whole blocks of 4
    assert joined(second) == joined(first)
    assert rig.engine.stats.cached_prompt_tokens == 8


def test_prefix_caching_can_be_switched_off(device: torch.device) -> None:
    rig = Rig(device, enable_prefix_caching=False)
    for name in "ab":
        rig.engine.add_request(name, PROMPT * 2, GREEDY)
        assert rig.run()[name][-1].num_cached_tokens == 0


def test_abort_gives_the_blocks_back(device: torch.device) -> None:
    rig = Rig(device)
    free = rig.engine.kv.pool.num_free
    rig.engine.add_request("a", PROMPT, GREEDY)
    rig.engine.step()
    assert rig.engine.kv.pool.num_free < free
    output = rig.engine.abort("a")
    assert output is not None
    assert output.finish_reason == FinishReason.ABORT
    assert rig.engine.kv.pool.num_free == free
    assert not rig.engine.has_unfinished
    assert rig.engine.abort("a") is None


def test_duplicate_ids_and_oversized_prompts_are_refused(device: torch.device) -> None:
    rig = Rig(device)
    rig.engine.add_request("a", PROMPT, GREEDY)
    with pytest.raises(ValueError, match="already running"):
        rig.engine.add_request("a", PROMPT, GREEDY)
    with pytest.raises(ValueError, match="max_model_len"):
        rig.engine.add_request("b", [1] * 64, GREEDY)


def test_max_model_len_ends_a_request_before_max_tokens(device: torch.device) -> None:
    rig = Rig(device, max_model_len=12)
    rig.engine.add_request("a", PROMPT, SamplingParams(temperature=0, max_tokens=50))
    last = rig.run()["a"][-1]
    assert last.finish_reason == FinishReason.LENGTH
    assert last.num_prompt_tokens + last.num_generated_tokens == 12


def test_a_cache_that_cannot_hold_one_sequence_is_refused_up_front(
    device: torch.device,
) -> None:
    with pytest.raises(ValueError, match="KV cache holds"):
        Rig(device, num_kv_blocks=4)


def test_preempted_requests_still_produce_the_same_text(device: torch.device) -> None:
    # 17 usable blocks (68 tokens) for three requests that need 3 * 20 tokens
    rig = Rig(device, num_kv_blocks=18, max_model_len=40)
    prompts = {"a": [5, 6, 7, 8], "b": [9, 10, 11, 12], "c": [13, 14, 15, 16]}
    params = SamplingParams(temperature=0, max_tokens=16)
    for name, prompt in prompts.items():
        rig.engine.add_request(name, prompt, params)
    seen = rig.run()
    for name, prompt in prompts.items():
        assert joined(seen[name]) == rig.text(rig.greedy(prompt, 16))


def image_prompt(seed: int, key: str) -> tuple[list[int], ImageSegment]:
    return [5, 6, 7, 8, IMG, IMG, IMG, IMG, 9, 10, 11, 12, 13], image_segment(
        4, key, seed
    )


def test_an_image_prompt_matches_the_dense_reference(device: torch.device) -> None:
    rig = Rig(device)
    prompt, segment = image_prompt(1, "cat")
    rig.engine.add_request("a", prompt, GREEDY, [segment])
    reference = TinyVision(rig.model_config.hidden_size, torch.device("cpu"))
    expected = dense_greedy(
        rig.model_config,
        rig.weights,
        prompt,
        8,
        image_rows=(4, reference.encode([segment.inputs])[0]),
    )
    assert joined(rig.run()["a"]) == rig.text(expected)


def test_a_repeated_image_skips_the_vision_tower_and_the_prefill(
    device: torch.device,
) -> None:
    rig = Rig(device)
    prompt, segment = image_prompt(1, "cat")
    rig.engine.add_request("a", prompt, GREEDY, [segment])
    first = rig.run()["a"]
    rig.engine.add_request("b", prompt, GREEDY, [image_segment(4, "cat", 1)])
    second = rig.run()["b"]
    assert rig.vision.images == 1
    assert second[-1].num_cached_tokens == 12  # the image blocks came from the cache
    assert joined(second) == joined(first)


def test_another_image_with_the_same_text_does_not_reuse_the_blocks(
    device: torch.device,
) -> None:
    rig = Rig(device, encoder_cache_tokens=0)
    prompt, segment = image_prompt(1, "cat")
    rig.engine.add_request("a", prompt, GREEDY, [segment])
    first = rig.run()["a"]
    rig.engine.add_request("b", prompt, GREEDY, [image_segment(4, "dog", 2)])
    second = rig.run()["b"]
    assert second[-1].num_cached_tokens == 4  # only the text in front of the image
    assert rig.vision.images == 2
    assert joined(second) != joined(first)


def test_the_encoder_cache_helps_when_the_text_differs(device: torch.device) -> None:
    rig = Rig(device)
    prompt, segment = image_prompt(1, "cat")
    rig.engine.add_request("a", prompt, GREEDY, [segment])
    rig.run()
    other = [20, 21, 22, 23, *prompt[4:]]
    rig.engine.add_request("b", other, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    assert rig.vision.images == 1  # no block in common, but the embeddings are reused
    assert rig.engine.encoder_cache is not None
    assert rig.engine.encoder_cache.hits == 1

import torch

from tests.models import image_segment
from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine
from vlmserve.sampling import SamplingParams
from vlmserve.server.metrics import render

CPU = torch.device("cpu")
GREEDY = SamplingParams(temperature=0, max_tokens=6)
IMG = 3  # the placeholder id of the tiny tokenizer, as in test_engine
PROMPT = [5, 6, 7, 8, IMG, IMG, IMG, IMG, 9, 10, 11, 12, 13]


def scrape(text: str) -> dict[str, float]:
    """Parse the exposition text back, checking that every sample is described."""
    helped: set[str] = set()
    typed: dict[str, str] = {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helped.add(line.split()[2])
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            typed[name] = kind
        else:
            name, value = line.split()
            values[name] = float(value)
    assert helped == set(typed) == set(values)
    assert all(kind in ("gauge", "counter") for kind in typed.values())
    assert all(n.endswith("_total") for n, kind in typed.items() if kind == "counter")
    return values


def test_a_fresh_engine_reports_zeroes() -> None:
    rig = Rig(CPU)
    values = scrape(render(AsyncEngine(rig.engine)))
    assert values["vlmserve_engine_healthy"] == 0  # not started yet
    assert values["vlmserve_requests_running"] == 0
    assert values["vlmserve_requests_waiting"] == 0
    assert values["vlmserve_kv_cache_usage_ratio"] == 0
    assert values["vlmserve_generation_tokens_total"] == 0
    assert values["vlmserve_encoder_cache_hits_total"] == 0


def test_without_a_vision_tower_there_are_no_image_metrics() -> None:
    rig = Rig(CPU, vision=False)
    values = scrape(render(AsyncEngine(rig.engine)))
    assert not any("image" in name or "encoder" in name for name in values)
    assert "vlmserve_requests_running" in values


def test_a_finished_request_is_counted() -> None:
    rig = Rig(CPU)
    rig.engine.add_request("a", PROMPT, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    values = scrape(render(AsyncEngine(rig.engine)))
    assert values["vlmserve_requests_finished_total"] == 1
    assert values["vlmserve_prompt_tokens_total"] == len(PROMPT)
    assert values["vlmserve_generation_tokens_total"] == 6
    assert values["vlmserve_image_tokens_encoded_total"] == 4
    assert values["vlmserve_encoder_cache_hits_total"] == 0
    assert values["vlmserve_encoder_cache_tokens"] == 4


def test_a_repeated_prompt_shows_up_as_prefix_cache_hits() -> None:
    rig = Rig(CPU)
    rig.engine.add_request("a", PROMPT, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    rig.engine.add_request("b", PROMPT, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    values = scrape(render(AsyncEngine(rig.engine)))
    assert values["vlmserve_prefix_cache_hit_tokens_total"] == 12
    assert values["vlmserve_image_tokens_encoded_total"] == 4  # the repeat skipped it


def test_the_same_image_after_other_text_hits_the_encoder_cache() -> None:
    rig = Rig(CPU)
    rig.engine.add_request("a", PROMPT, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    other = [20, 21, 22, 23, *PROMPT[4:]]  # no block in common, same picture
    rig.engine.add_request("b", other, GREEDY, [image_segment(4, "cat", 1)])
    rig.run()
    values = scrape(render(AsyncEngine(rig.engine)))
    assert values["vlmserve_encoder_cache_hits_total"] == 1
    assert values["vlmserve_image_tokens_encoded_total"] == 4


def test_a_request_in_flight_shows_in_the_gauges() -> None:
    rig = Rig(CPU, max_num_seqs=1)
    rig.engine.add_request("a", PROMPT, GREEDY)
    rig.engine.add_request("b", PROMPT, GREEDY)
    rig.engine.step()
    values = scrape(render(AsyncEngine(rig.engine)))
    assert values["vlmserve_requests_running"] == 1
    assert values["vlmserve_requests_waiting"] == 1
    assert 0 < values["vlmserve_kv_cache_usage_ratio"] <= 1

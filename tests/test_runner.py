import pytest
import torch

from tests.dense import dense_logits, random_weights
from tests.models import TinyVision, build_decoder, image_segment, tiny_config
from vlmserve.config import EngineConfig
from vlmserve.kv.manager import KVCacheManager
from vlmserve.multimodal.encoder_cache import EncoderCache
from vlmserve.request import ImageSegment, Request
from vlmserve.runner import ModelRunner
from vlmserve.sampler import Sampler
from vlmserve.sampling import SamplingParams
from vlmserve.scheduler import Batch, Scheduler

IMG = 49  # the placeholder token id, inside the tiny vocabulary
GREEDY = SamplingParams(temperature=0)


class World:
    """Scheduler, runner and a tiny decoder, plus the loop the engine will run."""

    def __init__(
        self,
        device: torch.device,
        max_tokens: int = 64,
        block_size: int = 4,
        cache_tokens: int = 0,
        with_vision: bool = True,
    ) -> None:
        self.config = tiny_config()
        self.weights = random_weights(self.config, seed=3)
        self.decoder = build_decoder(self.config, self.weights, device)
        self.decoder.allocate_kv_cache(60, block_size)
        self.cache = EncoderCache(cache_tokens) if cache_tokens else None
        self.vision = TinyVision(self.config.hidden_size, device)
        engine = EngineConfig(
            "tiny",
            max_model_len=128,
            max_num_seqs=4,
            max_num_batched_tokens=max_tokens,
            block_size=block_size,
            num_kv_blocks=60,
        )
        self.scheduler = Scheduler(engine, KVCacheManager(60, block_size), self.cache)
        self.runner = ModelRunner(
            self.decoder,
            self.vision if with_vision else None,
            block_size,
            Sampler(device),
            self.cache,
        )
        self.inputs: list[torch.Tensor] = []  # what each step fed into the decoder
        self.decoder.register_forward_pre_hook(
            lambda module, args: self.inputs.append(args[0].detach().cpu().clone())
        )

    def step(self) -> Batch:
        batch = self.scheduler.schedule()
        sampled = iter(self.runner.execute(batch))
        self.scheduler.advance(batch)
        for item in batch.items:
            if item.samples:
                item.request.append_token(next(sampled))
        return batch

    def embed(self, token_id: int) -> torch.Tensor:
        return self.weights["embed_tokens.weight"][token_id]

    def reference_embeddings(self, segment: ImageSegment) -> torch.Tensor:
        """What the vision tower gives for this image, from a tower of our own."""
        tower = TinyVision(self.config.hidden_size, torch.device("cpu"))
        return tower.encode([segment.inputs])[0]


def with_image(
    rid: str, before: list[int], after: list[int], key: str, seed: int = 0
) -> Request:
    tokens = [*before, *[IMG] * 4, *after]
    return Request(
        rid, tokens, GREEDY, segments=[image_segment(len(before), key, seed)]
    )


def test_the_image_rows_get_the_vision_embeddings(device: torch.device) -> None:
    world = World(device)
    req = with_image("a", [1, 2], [3], "cat")
    world.scheduler.add(req)
    world.step()
    (fed,) = world.inputs
    torch.testing.assert_close(fed[:2], torch.stack([world.embed(1), world.embed(2)]))
    torch.testing.assert_close(
        fed[2:6], world.reference_embeddings(req.segments[0]), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(fed[6], world.embed(3))


def test_a_chunk_boundary_in_the_middle_of_an_image(device: torch.device) -> None:
    world = World(device, max_tokens=4)
    req = with_image("a", [1, 2], [3], "cat")
    world.scheduler.add(req)
    world.step()  # tokens 0-3: the text and half the image
    world.step()  # tokens 4-6: the other half and the last text token
    first, second = world.inputs
    reference = world.reference_embeddings(req.segments[0])
    torch.testing.assert_close(first[2:4], reference[:2], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(second[:2], reference[2:], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(second[2], world.embed(3))


def test_generation_with_an_image_matches_the_dense_reference(
    device: torch.device,
) -> None:
    world = World(device)
    req = with_image("a", [5, 6, 7], [8, 9], "cat", seed=2)
    world.scheduler.add(req)
    for _ in range(6):  # the prompt, then five decode steps
        world.step()
    reference = world.reference_embeddings(req.segments[0])
    tokens = list(req.prompt_token_ids)
    for generated in req.output_token_ids:
        embeds = world.weights["embed_tokens.weight"][torch.tensor(tokens)].clone()
        embeds[3:7] = reference
        logits = dense_logits(world.config, world.weights, tokens, embeds)[-1]
        first, second = logits.topk(2).values.tolist()
        assert first - second > 1e-3, "a near tie could flip, try another seed"
        assert int(logits.argmax()) == generated
        tokens.append(generated)


def test_text_only_requests_work_next_to_images(device: torch.device) -> None:
    world = World(device)
    world.scheduler.add(Request("t", [1, 2, 3, 4], GREEDY))
    world.scheduler.add(with_image("i", [1], [2], "cat"))
    batch = world.step()
    assert [i.request.request_id for i in batch.items] == ["t", "i"]
    assert world.vision.images == 1


def test_one_image_in_two_requests_goes_through_the_tower_once(
    device: torch.device,
) -> None:
    world = World(device)
    world.scheduler.add(with_image("a", [1], [2], "cat"))
    world.scheduler.add(with_image("b", [3, 4], [5], "cat"))
    world.step()
    assert world.vision.images == 1
    assert world.vision.calls == 1


def test_the_cache_spares_the_tower_a_repeated_image(device: torch.device) -> None:
    world = World(device, cache_tokens=100)
    world.scheduler.add(with_image("a", [1], [2], "cat"))
    world.step()
    world.scheduler.add(with_image("b", [7, 8], [9], "cat"))
    world.step()
    assert world.vision.images == 1
    assert world.cache is not None
    assert world.cache.hits == 1
    assert world.runner.images_encoded == 4


def test_without_a_cache_the_tower_runs_again(device: torch.device) -> None:
    world = World(device)
    world.scheduler.add(with_image("a", [1], [2], "cat"))
    world.step()
    world.scheduler.add(with_image("b", [7, 8], [9], "cat"))
    world.step()
    assert world.vision.images == 2


def test_an_evicted_embedding_is_computed_again(device: torch.device) -> None:
    world = World(device, cache_tokens=100)
    world.scheduler.add(with_image("a", [1], [2], "cat"))
    world.step()
    assert world.cache is not None
    world.scheduler.add(with_image("b", [7, 8], [9], "cat"))
    batch = world.scheduler.schedule()  # planned as a cache hit ...
    assert batch.to_encode == []
    world.cache = None  # ... but gone by the time the runner looks
    world.runner.encoder_cache = None
    world.runner.execute(batch)
    assert world.vision.images == 2


def test_embeddings_are_released_after_the_prompt(device: torch.device) -> None:
    world = World(device)
    req = with_image("a", [1], [2], "cat")
    world.scheduler.add(req)
    world.step()
    assert req.embeddings == {}


def test_images_without_a_vision_tower_fail_loudly(device: torch.device) -> None:
    world = World(device, with_vision=False)
    world.scheduler.add(with_image("a", [1], [2], "cat"))
    with pytest.raises(RuntimeError, match="no vision tower"):
        world.step()


def test_a_token_count_the_tower_does_not_match_is_caught(device: torch.device) -> None:
    world = World(device)
    segment = image_segment(1, "cat")
    short = ImageSegment(1, 3, "cat", segment.inputs)  # the tower makes 4
    world.scheduler.add(Request("a", [1, IMG, IMG, IMG, 2], GREEDY, segments=[short]))
    with pytest.raises(RuntimeError, match="3 placeholder tokens"):
        world.step()

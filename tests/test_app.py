from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from fastapi.testclient import TestClient

from tests.frontend import TinyFrontend
from tests.rig import Rig
from vlmserve.async_engine import AsyncEngine
from vlmserve.multimodal.images import ImageLimits
from vlmserve.server.app import create_app


@dataclass
class Served:
    client: TestClient
    rig: Rig
    engine: AsyncEngine
    frontend: TinyFrontend


@pytest.fixture
def served() -> Iterator[Served]:
    rig = Rig(torch.device("cpu"))
    engine = AsyncEngine(rig.engine)
    frontend = TinyFrontend()
    app = create_app(engine, frontend, "tiny-vlm", ImageLimits(max_images=2))
    with TestClient(app) as client:
        yield Served(client, rig, engine, frontend)


def test_health_is_ok_while_the_engine_runs(served: Served) -> None:
    reply = served.client.get("/health")
    assert reply.status_code == 200
    assert reply.json() == {"status": "ok"}


def test_health_turns_into_a_503_when_the_engine_is_gone(served: Served) -> None:
    served.engine.stop()
    reply = served.client.get("/health")
    assert reply.status_code == 503
    assert reply.json() == {"status": "unavailable"}


def test_the_model_list_has_the_served_name(served: Served) -> None:
    body = served.client.get("/v1/models").json()
    assert body["object"] == "list"
    (card,) = body["data"]
    assert card["id"] == "tiny-vlm"
    assert card["max_model_len"] == 64


def test_unknown_routes_are_404(served: Served) -> None:
    assert served.client.get("/nope").status_code == 404

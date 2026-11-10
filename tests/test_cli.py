import base64
import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from tests.tinyhf import SINGLE, TinyLlava, TinyText, hf_continuation, picture
from vlmserve import __version__, cli
from vlmserve.config import EngineConfig

TINY_ENGINE = [
    "--device",
    "cpu",
    "--dtype",
    "float32",
    "--max-model-len",
    "128",
    "--num-kv-blocks",
    "120",
    "--block-size",
    "4",
    "--max-num-seqs",
    "4",
    "--max-num-batched-tokens",
    "64",
]


def png_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def expected_text(tiny: TinyLlava, steps: int) -> str:
    tokens = hf_continuation(tiny, SINGLE, steps)
    return str(tiny.processor.tokenizer.decode(tokens, skip_special_tokens=True))


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    """uvicorn.run stands in for the real server: remembers the app it was given."""
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
    )
    return calls


def test_the_parser_takes_its_defaults_from_the_engine_config() -> None:
    defaults = EngineConfig(model="")
    args = cli.build_parser().parse_args(["serve", "--model", "x"])
    assert args.max_num_seqs == defaults.max_num_seqs
    assert args.block_size == defaults.block_size
    assert args.prefix_caching is True
    assert args.max_model_len is None
    assert (args.host, args.port) == ("127.0.0.1", 8000)  # not open to the network


def test_prefix_caching_can_be_turned_off() -> None:
    args = cli.build_parser().parse_args(
        ["serve", "--model", "x", "--no-prefix-caching"]
    )
    assert args.prefix_caching is False


def test_a_model_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["serve"])
    assert stop.value.code == 2
    assert "--model" in capsys.readouterr().err


def test_serve_builds_an_app_that_answers_like_transformers(
    tiny_llava: TinyLlava, uvicorn_calls: list[tuple[Any, dict[str, Any]]]
) -> None:
    argv = ["serve", "--model", str(tiny_llava.path), *TINY_ENGINE]
    assert cli.main([*argv, "--served-model-name", "tiny-vlm", "--port", "9123"]) == 0
    ((app, kwargs),) = uvicorn_calls
    assert (kwargs["host"], kwargs["port"]) == ("127.0.0.1", 9123)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": png_url(picture(1))}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
        "max_tokens": 6,
        "temperature": 0,
    }
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").json()["data"][0]["id"] == "tiny-vlm"
        reply = client.post("/v1/chat/completions", json=body)
        assert reply.status_code == 200
        message = reply.json()["choices"][0]["message"]["content"]
        assert message == expected_text(tiny_llava, 6)
        assert "vlmserve_requests_finished_total 1" in client.get("/metrics").text


def test_serve_passes_the_image_limits_on(
    tiny_llava: TinyLlava, uvicorn_calls: list[tuple[Any, dict[str, Any]]]
) -> None:
    argv = ["serve", "--model", str(tiny_llava.path), *TINY_ENGINE, "--max-images", "1"]
    assert cli.main(argv) == 0
    ((app, _),) = uvicorn_calls
    assert app.state.limits.max_images == 1


def test_generate_prints_the_answer_and_some_numbers(
    tiny_llava: TinyLlava, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    photo = tmp_path / "photo.png"
    picture(1).save(photo)
    argv = ["generate", "--model", str(tiny_llava.path), *TINY_ENGINE]
    argv += ["--prompt", "what is this?", "--image", str(photo), "--max-tokens", "6"]
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    assert out == expected_text(tiny_llava, 6) + "\n"
    assert "6 generated" in err
    assert "tokens/s" in err


def test_generate_works_for_a_plain_text_model(
    tiny_text: TinyText, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["generate", "--model", str(tiny_text.path), *TINY_ENGINE]
    argv += ["--prompt", "hi", "--system", "be brief", "--max-tokens", "5"]
    assert cli.main(argv) == 0
    out, _ = capsys.readouterr()
    assert out.endswith("\n")
    assert len(out) > 1


def test_generate_stops_at_the_end_of_the_context(
    tiny_llava: TinyLlava, capsys: pytest.CaptureFixture[str]
) -> None:
    # the window is 128 tokens, a request for far more than fits is shortened
    argv = ["generate", "--model", str(tiny_llava.path), *TINY_ENGINE]
    argv += ["--prompt", "hi", "--max-tokens", "100000"]
    assert cli.main(argv) == 0
    assert "generated" in capsys.readouterr().err


def test_devices_lists_the_cpu_and_the_dtype_it_gets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["devices"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("torch ")
    assert "cpu (" in out
    assert f"dtype auto: {torch.float32}" in out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--model", "/nope/nothing"], "not a directory"),
        (["--max-model-len", "129"], "trained for 128 tokens"),
        (
            ["--max-num-seqs", "8", "--max-num-batched-tokens", "4"],
            "at least max_num_seqs",
        ),
        (["--dtype", "int8"], "unknown dtype"),
        (["--device", "tpu"], "unknown device"),
    ],
)
def test_user_mistakes_end_with_a_message_not_a_traceback(
    tiny_llava: TinyLlava,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    argv = ["generate", "--model", str(tiny_llava.path), *TINY_ENGINE]
    argv += ["--prompt", "hi"]
    # later options win, so a mistake replaces the matching sane option
    code = cli.main([*argv, *extra])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("vlmserve: error: ")
    assert message in err


def test_python_m_vlmserve_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vlmserve", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"vlmserve {__version__}"

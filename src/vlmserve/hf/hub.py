from pathlib import Path

from huggingface_hub import snapshot_download

# what a model needs to run: weights, configs, tokenizer and template files.
# No pickles, no gguf, no onnx copies of the same weights.
PATTERNS = ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt", "*.tiktoken"]


def resolve_model_dir(ref: str, revision: str | None = None) -> Path:
    """A local directory as it is, anything else is taken for a Hub repo id.

    Repo ids look like "org/name"; a path that is not there is reported as such
    instead of being sent to the Hub as a very odd repo name.
    """
    path = Path(ref).expanduser()
    if path.is_dir():
        return path
    if ref.startswith((".", "/", "~")) or ref.count("/") > 1:
        raise FileNotFoundError(f"{ref} is not a directory")
    return Path(snapshot_download(ref, revision=revision, allow_patterns=PATTERNS))

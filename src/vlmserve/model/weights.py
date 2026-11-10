import re
from collections import defaultdict
from collections.abc import Iterable

import torch

from vlmserve.model.decoder import Decoder

_QKV = re.compile(r"(layers\.\d+\.self_attn)\.([qkv])_proj\.(weight|bias)$")
_GATE_UP = re.compile(r"(layers\.\d+\.mlp)\.(gate|up)_proj\.weight$")


def _expected_parts(param_name: str) -> set[str]:
    if "qkv_proj" in param_name:
        return {"q", "k", "v"}
    if "gate_up_proj" in param_name:
        return {"gate", "up"}
    return {"whole"}


def load_decoder_weights(
    decoder: Decoder, tensors: Iterable[tuple[str, torch.Tensor]]
) -> list[str]:
    """Copy hf-named tensors into the decoder, fusing q/k/v and gate/up on the way.

    Names are relative to the text model ("layers.0.self_attn.q_proj.weight",
    "embed_tokens.weight", "norm.weight", "lm_head.weight"). Tensors are copied as
    they come, so a checkpoint can be streamed without holding it in memory.
    Returns the names that had no place in the decoder; raises if any parameter of
    the decoder was left without a value.
    """
    cfg = decoder.config
    q_size = cfg.num_heads * cfg.head_dim
    kv_size = cfg.num_kv_heads * cfg.head_dim
    inter = cfg.intermediate_size
    qkv_rows = {
        "q": (0, q_size),
        "k": (q_size, q_size + kv_size),
        "v": (q_size + kv_size, q_size + 2 * kv_size),
    }
    gate_up_rows = {"gate": (0, inter), "up": (inter, 2 * inter)}

    params = dict(decoder.named_parameters())
    filled: dict[str, set[str]] = defaultdict(set)
    ignored: list[str] = []

    with torch.no_grad():
        for name, tensor in tensors:
            rows: tuple[int, int] | None = None
            part = "whole"
            target = name
            if m := _QKV.match(name):
                target, part = f"{m[1]}.qkv_proj.{m[3]}", m[2]
                rows = qkv_rows[part]
            elif m := _GATE_UP.match(name):
                target, part = f"{m[1]}.gate_up_proj.weight", m[2]
                rows = gate_up_rows[part]
            if target not in params:
                ignored.append(name)
                continue
            dest = params[target] if rows is None else params[target][rows[0] : rows[1]]
            if dest.shape != tensor.shape:
                raise ValueError(
                    f"{name}: checkpoint has {tuple(tensor.shape)}, "
                    f"the model expects {tuple(dest.shape)}"
                )
            dest.copy_(tensor)
            filled[target].add(part)

    missing = [n for n in params if filled[n] != _expected_parts(n)]
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise ValueError(f"{len(missing)} weights were never loaded: {shown}")
    return ignored

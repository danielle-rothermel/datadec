from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from datadec.models.convert import (
    CONVERSION_RECORD_FILENAME,
    ConversionRecord,
    UnsupportedCheckpointError,
    convert_hf_olmo_to_native,
    native_config,
    sha256_file,
)

D_MODEL = 8
N_HEADS = 2
N_LAYERS = 2
MLP_RATIO = 4
EMBEDDING_SIZE = 16
REVISION = "a" * 40
SOURCE_CONFIG: dict[str, object] = {
    "model_type": "hf_olmo",
    "activation_type": "swiglu",
    "alibi": False,
    "attention_layer_norm": False,
    "bias_for_layer_norm": False,
    "block_group_size": 1,
    "block_type": "sequential",
    "clip_qkv": None,
    "d_model": D_MODEL,
    "embedding_layer_norm": False,
    "embedding_size": EMBEDDING_SIZE,
    "eos_token_id": 15,
    "include_bias": False,
    "layer_norm_eps": 1e-6,
    "layer_norm_type": "rms",
    "layer_norm_with_affine": True,
    "max_sequence_length": 32,
    "mlp_hidden_size": None,
    "mlp_ratio": MLP_RATIO,
    "multi_query_attention": None,
    "n_heads": N_HEADS,
    "n_kv_heads": None,
    "n_layers": N_LAYERS,
    "norm_after": False,
    "pad_token_id": 1,
    "rope": True,
    "rope_theta": 10000,
    "scale_logits": False,
    "vocab_size": 14,
    "weight_tying": False,
}
TOKENIZER_FILES = {
    "special_tokens_map.json": b'{"eos_token": "<|endoftext|>"}\n',
    "tokenizer.json": b'{"version": "1.0", "model": {}}  \n',
    "tokenizer_config.json": b'{"tokenizer_class": "OLMoTokenizer"}',
}


def _source_tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    hidden = MLP_RATIO * D_MODEL

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.5

    tensors = {
        "model.transformer.wte.weight": rand(EMBEDDING_SIZE, D_MODEL),
        "model.transformer.ln_f.weight": 1 + rand(D_MODEL),
        "model.transformer.ff_out.weight": rand(EMBEDDING_SIZE, D_MODEL),
    }
    for layer in range(N_LAYERS):
        prefix = f"model.transformer.blocks.{layer}."
        tensors |= {
            prefix + "att_proj.weight": rand(3 * D_MODEL, D_MODEL),
            prefix + "attn_out.weight": rand(D_MODEL, D_MODEL),
            prefix + "ff_proj.weight": rand(hidden, D_MODEL),
            prefix + "ff_out.weight": rand(D_MODEL, hidden // 2),
            prefix + "attn_norm.weight": 1 + rand(D_MODEL),
            prefix + "ff_norm.weight": 1 + rand(D_MODEL),
        }
    return tensors


def _write_source(path: Path, config: dict[str, object] | None = None) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config or SOURCE_CONFIG))
    (path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 15, "pad_token_id": 1})
    )
    save_file(_source_tensors(), path / "model.safetensors")
    for name, content in TOKENIZER_FILES.items():
        (path / name).write_bytes(content)
    return path


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1e-6) * weight


def _rope(x: torch.Tensor) -> torch.Tensor:
    head_size = x.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    freqs = torch.outer(torch.arange(x.shape[-2]).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    first, second = x[..., : head_size // 2], x[..., head_size // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return x * emb.cos() + rotated * emb.sin()


def _olmo_reference_logits(
    tensors: dict[str, torch.Tensor], ids: list[int]
) -> torch.Tensor:
    """OLMo sequential-block forward pass, written from the OLMo layout."""
    t = {k.removeprefix("model.transformer."): v for k, v in tensors.items()}
    head_size = D_MODEL // N_HEADS
    length = len(ids)
    x = t["wte.weight"][ids]
    mask = torch.full((length, length), -math.inf).triu(1)
    for layer in range(N_LAYERS):
        block = f"blocks.{layer}."
        qkv = _rms(x, t[block + "attn_norm.weight"]) @ t[block + "att_proj.weight"].T
        q, k, v = (
            part.view(length, N_HEADS, head_size).transpose(0, 1)
            for part in qkv.split(D_MODEL, dim=-1)
        )
        q, k = _rope(q), _rope(k)
        weights = torch.softmax(q @ k.transpose(-1, -2) / head_size**0.5 + mask, -1)
        attention = (weights @ v).transpose(0, 1).reshape(length, D_MODEL)
        x = x + attention @ t[block + "attn_out.weight"].T
        up, gate = (
            _rms(x, t[block + "ff_norm.weight"]) @ t[block + "ff_proj.weight"].T
        ).chunk(2, dim=-1)
        x = x + (torch.nn.functional.silu(gate) * up) @ t[block + "ff_out.weight"].T
    return _rms(x, t["ln_f.weight"]) @ t["ff_out.weight"].T


def test_conversion_matches_the_olmo_forward_pass(tmp_path: Path) -> None:
    from transformers import LlamaForCausalLM

    source = _write_source(tmp_path / "source")
    out = tmp_path / "out"
    convert_hf_olmo_to_native("allenai/x", REVISION, out, source_dir=source)

    model = LlamaForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = [3, 7, 1, 12, 5, 9]
    with torch.inference_mode():
        native = model(input_ids=torch.tensor([ids])).logits[0]
    reference = _olmo_reference_logits(_source_tensors(), ids)

    assert native.shape == (len(ids), EMBEDDING_SIZE)
    torch.testing.assert_close(native, reference, atol=1e-5, rtol=1e-5)


def test_conversion_writes_verbatim_tokenizer_and_record(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    out = tmp_path / "out"

    record = convert_hf_olmo_to_native("allenai/x", REVISION, out, source_dir=source)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            CONVERSION_RECORD_FILENAME,
            "config.json",
            "generation_config.json",
            "model.safetensors",
            *TOKENIZER_FILES,
        ]
    )
    for name, content in TOKENIZER_FILES.items():
        assert (out / name).read_bytes() == content
        assert record.tokenizer_sha256[name] == sha256_file(source / name)
    stored = ConversionRecord.model_validate_json(
        (out / CONVERSION_RECORD_FILENAME).read_text()
    )
    assert stored == record
    assert record.weights_sha256 == sha256_file(out / "model.safetensors")
    assert record.target_architecture == "LlamaForCausalLM"
    assert len(record.tensors) == 3 + 6 * N_LAYERS
    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "llama"
    assert config["vocab_size"] == EMBEDDING_SIZE
    assert config["intermediate_size"] == MLP_RATIO * D_MODEL // 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clip_qkv", 8.0),
        ("layer_norm_type", "default"),
        ("layer_norm_with_affine", False),
        ("weight_tying", True),
        ("attention_layer_norm", True),
        ("n_kv_heads", 1),
    ],
)
def test_unsupported_layouts_are_refused(field: str, value: object) -> None:
    with pytest.raises(UnsupportedCheckpointError, match=field):
        native_config({**SOURCE_CONFIG, field: value})


def test_unmapped_source_tensors_are_refused(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "source")
    tensors = _source_tensors() | {"model.transformer.extra": torch.zeros(1)}
    save_file(tensors, source / "model.safetensors")

    with pytest.raises(UnsupportedCheckpointError, match="unmapped"):
        convert_hf_olmo_to_native(
            "allenai/x", REVISION, tmp_path / "out", source_dir=source
        )


def test_non_empty_out_dir_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale").write_text("x")

    with pytest.raises(FileExistsError):
        convert_hf_olmo_to_native(
            "allenai/x", REVISION, out, source_dir=tmp_path / "missing"
        )

"""Convert ``hf_olmo`` DataDecide checkpoints to a native transformers class.

DataDecide checkpoints declare ``model_type: hf_olmo``, which only loads
with the ``ai2-olmo`` package. Every DataDecide size uses the same OLMo
layout: pre-norm sequential blocks, RMSNorm with a learned scale and no
bias, SwiGLU, RoPE, no linear biases, no QKV clipping, no QK norm, and
untied input and output embeddings. Transformers' native ``olmo`` class
implements OLMo-1's non-parametric LayerNorm and cannot hold the learned
RMSNorm scales, so the exact native equivalent is ``LlamaForCausalLM``.
Fused projections are split; nothing is re-scaled or re-ordered.

Any source config outside the supported layout is refused rather than
approximated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final

from dr_hf import CommitSha
from pydantic import BaseModel, ConfigDict

from datadec.config.checkpoints import load_checkpoint_contract

if TYPE_CHECKING:
    import torch

__all__ = [
    "CONVERSION_RECORD_FILENAME",
    "ConversionRecord",
    "TensorMapping",
    "UnsupportedCheckpointError",
    "convert_hf_olmo_to_native",
    "download_source_checkpoint",
    "sha256_file",
]

CONVERSION_RECORD_FILENAME: Final = "conversion.json"
SOURCE_MODEL_TYPE: Final = "hf_olmo"
TARGET_MODEL_TYPE: Final = "llama"
TARGET_ARCHITECTURE: Final = "LlamaForCausalLM"
_WEIGHTS_FILENAME: Final = "model.safetensors"
_SOURCE_PREFIX: Final = "model.transformer."

# Source config fields whose values define the supported layout. Any other
# value means the checkpoint is not the layout this converter maps exactly.
SUPPORTED_LAYOUT: Final[dict[str, object]] = {
    "activation_type": "swiglu",
    "alibi": False,
    "attention_layer_norm": False,
    "bias_for_layer_norm": False,
    "block_group_size": 1,
    "block_type": "sequential",
    "clip_qkv": None,
    "embedding_layer_norm": False,
    "include_bias": False,
    "layer_norm_type": "rms",
    "layer_norm_with_affine": True,
    "mlp_hidden_size": None,
    "multi_query_attention": None,
    "n_kv_heads": None,
    "norm_after": False,
    "rope": True,
    "scale_logits": False,
    "weight_tying": False,
}


class UnsupportedCheckpointError(ValueError):
    """The source checkpoint is not the supported hf_olmo layout."""


class TensorMapping(BaseModel):
    """One source tensor and the target tensors it becomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    targets: tuple[str, ...]
    transform: str


class ConversionRecord(BaseModel):
    """What a conversion read, what it wrote, and how tensors map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_repo_id: str
    source_revision: CommitSha
    source_model_type: str
    target_model_type: str
    target_architecture: str
    conversion_tool: str
    transformers_version: str
    source_config: dict[str, object]
    target_config: dict[str, object]
    tensors: tuple[TensorMapping, ...]
    tokenizer_sha256: dict[str, str]
    weights_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conversion_tool_name() -> str:
    """``datadec <version> (git <commit>)``; the commit is omitted outside a
    git checkout and marked ``-dirty`` when tracked files are modified."""
    name = f"datadec {version('datadec')} datadec.models.convert"
    commit = _git_commit()
    return f"{name} (git {commit})" if commit else name


def _git_commit() -> str | None:
    package_dir = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def download_source_checkpoint(source_repo_id: str, source_revision: CommitSha) -> Path:
    """Download the conversion inputs of one source commit; return the dir."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    files = list(load_checkpoint_contract().source.files)
    return Path(
        snapshot_download(
            source_repo_id, revision=source_revision, allow_patterns=files
        )
    )


def convert_hf_olmo_to_native(
    source_repo_id: str,
    source_revision: CommitSha,
    out_dir: Path,
    *,
    source_dir: Path | None = None,
) -> ConversionRecord:
    """Convert one hf_olmo checkpoint commit into ``out_dir``.

    ``out_dir`` must be absent or empty. It receives ``config.json``,
    ``generation_config.json``, ``model.safetensors``, the tokenizer files
    copied byte for byte, and ``conversion.json`` (the returned record).
    ``source_dir`` skips the download when the commit is already local.
    """
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    contract = load_checkpoint_contract().source
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"out_dir must be empty: {out_dir}")
    if source_dir is None:
        source_dir = download_source_checkpoint(source_repo_id, source_revision)
    source_config = json.loads((source_dir / "config.json").read_text())
    target_config = native_config(source_config)
    source_tensors = load_file(source_dir / _WEIGHTS_FILENAME)
    target_tensors, mappings = map_tensors(source_tensors, source_config)
    check_target_shapes(target_config, target_tensors)

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / _WEIGHTS_FILENAME
    save_file(
        {name: tensor.contiguous() for name, tensor in target_tensors.items()},
        weights_path,
        metadata={"format": "pt"},
    )
    write_native_configs(target_config, source_dir, out_dir)
    tokenizer_sha256: dict[str, str] = {}
    for filename in contract.tokenizer_files:
        shutil.copyfile(source_dir / filename, out_dir / filename)
        tokenizer_sha256[filename] = sha256_file(out_dir / filename)
        if tokenizer_sha256[filename] != sha256_file(source_dir / filename):
            raise RuntimeError(f"tokenizer copy changed bytes: {filename}")
    record = ConversionRecord(
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_model_type=SOURCE_MODEL_TYPE,
        target_model_type=TARGET_MODEL_TYPE,
        target_architecture=TARGET_ARCHITECTURE,
        conversion_tool=conversion_tool_name(),
        transformers_version=version("transformers"),
        source_config=source_config,
        target_config=target_config,
        tensors=tuple(mappings),
        tokenizer_sha256=tokenizer_sha256,
        weights_sha256=sha256_file(weights_path),
    )
    (out_dir / CONVERSION_RECORD_FILENAME).write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return record


def native_config(source: dict[str, object]) -> dict[str, object]:
    """Map a supported hf_olmo config to LlamaConfig keyword arguments."""
    if source.get("model_type") != SOURCE_MODEL_TYPE:
        raise UnsupportedCheckpointError(
            f"expected model_type {SOURCE_MODEL_TYPE!r}, "
            f"got {source.get('model_type')!r}"
        )
    mismatches = {
        field: source.get(field)
        for field, expected in SUPPORTED_LAYOUT.items()
        if source.get(field) != expected
    }
    if mismatches:
        raise UnsupportedCheckpointError(
            f"unsupported hf_olmo layout fields: {mismatches!r}"
        )
    d_model = _int(source, "d_model")
    n_heads = _int(source, "n_heads")
    if d_model % n_heads:
        raise UnsupportedCheckpointError("d_model must divide by n_heads")
    mlp_hidden = _int(source, "mlp_ratio") * d_model
    embedding_size = source.get("embedding_size") or source["vocab_size"]
    return {
        "vocab_size": embedding_size,
        "hidden_size": d_model,
        "intermediate_size": mlp_hidden // 2,
        "num_hidden_layers": _int(source, "n_layers"),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "hidden_act": "silu",
        "max_position_embeddings": _int(source, "max_sequence_length"),
        "rms_norm_eps": source["layer_norm_eps"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(source["rope_theta"]),  # type: ignore[arg-type]
        },
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "bos_token_id": None,
        "eos_token_id": source["eos_token_id"],
        "pad_token_id": source["pad_token_id"],
    }


def _int(source: dict[str, object], field: str) -> int:
    value = source.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnsupportedCheckpointError(f"{field} must be an int: {value!r}")
    return value


def map_tensors(
    source: dict[str, torch.Tensor], config: dict[str, object]
) -> tuple[dict[str, torch.Tensor], list[TensorMapping]]:
    """Rename and split every source tensor; refuse leftovers or gaps."""
    d_model = _int(config, "d_model")
    n_layers = _int(config, "n_layers")
    hidden = _int(config, "mlp_ratio") * d_model // 2
    target: dict[str, torch.Tensor] = {}
    mappings: list[TensorMapping] = []
    remaining = dict(source)

    def take(name: str) -> torch.Tensor:
        key = _SOURCE_PREFIX + name
        if key not in remaining:
            raise UnsupportedCheckpointError(f"missing source tensor {key!r}")
        return remaining.pop(key)

    def rename(name: str, new: str) -> None:
        target[new] = take(name)
        mappings.append(
            TensorMapping(
                source=_SOURCE_PREFIX + name, targets=(new,), transform="copy"
            )
        )

    def split(name: str, news: tuple[str, ...], size: int) -> None:
        tensor = take(name)
        if tensor.shape[0] != size * len(news):
            raise UnsupportedCheckpointError(
                f"{name} has {tensor.shape[0]} rows, expected {size * len(news)}"
            )
        for index, new in enumerate(news):
            target[new] = tensor[index * size : (index + 1) * size].clone()
        mappings.append(
            TensorMapping(
                source=_SOURCE_PREFIX + name,
                targets=news,
                transform=f"split rows into {len(news)} blocks of {size}",
            )
        )

    rename("wte.weight", "model.embed_tokens.weight")
    for layer in range(n_layers):
        block = f"blocks.{layer}."
        out = f"model.layers.{layer}."
        split(
            block + "att_proj.weight",
            tuple(out + f"self_attn.{p}_proj.weight" for p in ("q", "k", "v")),
            d_model,
        )
        rename(block + "attn_out.weight", out + "self_attn.o_proj.weight")
        # OLMo SwiGLU: ``x, gate = ff_proj(h).chunk(2); silu(gate) * x``, so
        # the first half is Llama's up projection and the second its gate.
        split(
            block + "ff_proj.weight",
            (out + "mlp.up_proj.weight", out + "mlp.gate_proj.weight"),
            hidden,
        )
        rename(block + "ff_out.weight", out + "mlp.down_proj.weight")
        rename(block + "attn_norm.weight", out + "input_layernorm.weight")
        rename(block + "ff_norm.weight", out + "post_attention_layernorm.weight")
    rename("ln_f.weight", "model.norm.weight")
    rename("ff_out.weight", "lm_head.weight")
    if remaining:
        raise UnsupportedCheckpointError(
            f"unmapped source tensors: {sorted(remaining)!r}"
        )
    return target, mappings


def check_target_shapes(
    target_config: dict[str, object], tensors: dict[str, torch.Tensor]
) -> None:
    """Compare names and shapes against a meta-device LlamaForCausalLM."""
    import torch  # noqa: PLC0415
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415

    with torch.device("meta"):
        model = LlamaForCausalLM(LlamaConfig(**target_config))
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    actual = {name: tuple(t.shape) for name, t in tensors.items()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        raise UnsupportedCheckpointError(
            f"target tensors differ from {TARGET_ARCHITECTURE}: "
            f"missing={missing} extra={extra} wrong_shape={wrong}"
        )


def write_native_configs(
    target_config: dict[str, object], source_dir: Path, out_dir: Path
) -> None:
    from transformers import GenerationConfig, LlamaConfig  # noqa: PLC0415

    config = LlamaConfig(**target_config)
    config.architectures = [TARGET_ARCHITECTURE]
    config.dtype = "float32"
    config.save_pretrained(out_dir)
    source_generation = json.loads((source_dir / "generation_config.json").read_text())
    GenerationConfig(
        eos_token_id=source_generation.get("eos_token_id"),
        pad_token_id=source_generation.get("pad_token_id"),
    ).save_pretrained(out_dir)

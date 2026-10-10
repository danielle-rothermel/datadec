from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from dr_hf import CommitSha
from pydantic import BaseModel, ConfigDict

from datadec.config.checkpoints import load_checkpoint_contract

if TYPE_CHECKING:
    import torch

__all__ = [
    "ConversionRecord",
    "TensorMapping",
    "UnsupportedCheckpointError",
    "convert_hf_olmo_to_native",
    "download_source_checkpoint",
    "sha256_file",
]


class UnsupportedCheckpointError(ValueError): ...


class TensorMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    targets: tuple[str, ...]
    transform: str


class ConversionRecord(BaseModel):
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
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    checkpoint_contract = load_checkpoint_contract()
    contract = checkpoint_contract.source
    conversion = checkpoint_contract.conversion
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"out_dir must be empty: {out_dir}")
    if source_dir is None:
        source_dir = download_source_checkpoint(source_repo_id, source_revision)
    source_config = json.loads((source_dir / "config.json").read_text())
    target_config = native_config(source_config)
    source_tensors = load_file(source_dir / conversion.weights_filename)
    target_tensors, mappings = map_tensors(source_tensors, source_config)
    check_target_shapes(target_config, target_tensors)

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / conversion.weights_filename
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
        source_model_type=conversion.source_model_type,
        target_model_type=conversion.target_model_type,
        target_architecture=conversion.target_architecture,
        conversion_tool=conversion_tool_name(),
        transformers_version=version("transformers"),
        source_config=source_config,
        target_config=target_config,
        tensors=tuple(mappings),
        tokenizer_sha256=tokenizer_sha256,
        weights_sha256=sha256_file(weights_path),
    )
    (out_dir / conversion.conversion_record_filename).write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return record


def native_config(source: dict[str, object]) -> dict[str, object]:
    conversion = load_checkpoint_contract().conversion
    if source.get("model_type") != conversion.source_model_type:
        raise UnsupportedCheckpointError(
            f"expected model_type {conversion.source_model_type!r}, "
            f"got {source.get('model_type')!r}"
        )
    layout = conversion.supported_source_layout
    mismatches: dict[str, object] = {
        field: source.get(field)
        for field, expected in layout.required_values.items()
        if source.get(field) != expected
    }
    mismatches |= {
        field: source.get(field)
        for field in layout.required_absent
        if source.get(field) is not None
    }
    if mismatches:
        raise UnsupportedCheckpointError(
            f"unsupported {conversion.source_model_type} layout fields: {mismatches!r}"
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
    conversion = load_checkpoint_contract().conversion
    prefix = conversion.source_tensor_prefix
    n_layers = _int(config, "n_layers")
    target: dict[str, torch.Tensor] = {}
    mappings: list[TensorMapping] = []
    remaining = dict(source)

    def take(name: str) -> torch.Tensor:
        if name not in remaining:
            raise UnsupportedCheckpointError(f"missing source tensor {name!r}")
        return remaining.pop(name)

    def apply(source_name: str, targets: tuple[str, ...], transform: str) -> None:
        tensor = take(source_name)
        if transform == "copy":
            target[targets[0]] = tensor
            mappings.append(
                TensorMapping(source=source_name, targets=targets, transform="copy")
            )
            return
        rows = tensor.shape[0]
        if rows % len(targets):
            raise UnsupportedCheckpointError(
                f"{source_name}: {rows} rows do not split into {len(targets)} blocks"
            )
        size = rows // len(targets)
        for index, name in enumerate(targets):
            target[name] = tensor[index * size : (index + 1) * size].clone()
        mappings.append(
            TensorMapping(
                source=source_name,
                targets=targets,
                transform=f"split rows into {len(targets)} blocks of {size}",
            )
        )

    for rule in conversion.tensor_map:
        layers: tuple[int | None, ...] = (
            tuple(range(n_layers)) if "{layer}" in rule.source else (None,)
        )
        for layer in layers:
            fill = {"layer": layer} if layer is not None else {}
            apply(
                prefix + rule.source.format(**fill),
                tuple(t.format(**fill) for t in rule.targets),
                rule.transform,
            )
    if remaining:
        raise UnsupportedCheckpointError(
            f"unmapped source tensors: {sorted(remaining)!r}"
        )
    return target, mappings


def check_target_shapes(
    target_config: dict[str, object], tensors: dict[str, torch.Tensor]
) -> None:
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
            "target tensors differ from "
            f"{load_checkpoint_contract().conversion.target_architecture}: "
            f"missing={missing} extra={extra} wrong_shape={wrong}"
        )


def write_native_configs(
    target_config: dict[str, object], source_dir: Path, out_dir: Path
) -> None:
    from transformers import GenerationConfig, LlamaConfig  # noqa: PLC0415

    config = LlamaConfig(**target_config)
    config.architectures = [load_checkpoint_contract().conversion.target_architecture]
    config.dtype = "float32"
    config.save_pretrained(out_dir)
    source_generation = json.loads((source_dir / "generation_config.json").read_text())
    GenerationConfig(
        eos_token_id=source_generation.get("eos_token_id"),
        pad_token_id=source_generation.get("pad_token_id"),
    ).save_pretrained(out_dir)

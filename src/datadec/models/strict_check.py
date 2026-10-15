from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from datadec.config.checkpoints import load_checkpoint_contract

__all__ = [
    "REFERENCE_SCRIPT",
    "StrictCheckResult",
    "compare_with_reference",
    "reference_logits",
    "run_strict_check",
]

REFERENCE_SCRIPT: Final = (
    Path(__file__).parents[3] / "scripts" / "models" / "hf_olmo_reference.py"
)


@dataclass(frozen=True, slots=True)
class StrictCheckResult:
    prompts: int
    positions: int
    max_abs_logit_diff: float
    argmax_agreement: float
    reference_environment: dict[str, str]
    native_transformers: str

    def as_json(self) -> dict[str, object]:
        return {
            "kind": "strict_logit_check",
            "prompts": self.prompts,
            "positions": self.positions,
            "max_abs_logit_diff": self.max_abs_logit_diff,
            "argmax_agreement": self.argmax_agreement,
            "device": "cpu",
            "dtype": "float32",
            "reference_environment": self.reference_environment,
            "native_transformers": self.native_transformers,
        }


def reference_logits(
    source_dir: Path, prompts: tuple[str, ...], out_path: Path
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = Path(tmp) / "prompts.json"
        prompts_path.write_text(json.dumps(list(prompts)))
        strict = load_checkpoint_contract().strict_check
        if not REFERENCE_SCRIPT.is_file():
            raise FileNotFoundError(REFERENCE_SCRIPT)
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            strict.python,
            *(
                arg
                for req in strict.reference_environment
                for arg in ("--with", req.requirement)
            ),
            "python",
            "-I",
            str(REFERENCE_SCRIPT),
            str(source_dir),
            str(prompts_path),
            str(out_path),
        ]
        subprocess.run(command, check=True)  # noqa: S603


def compare_with_reference(
    native_dir: Path, prompts: tuple[str, ...], reference_path: Path
) -> StrictCheckResult:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(native_dir, dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(native_dir)
    max_diff = 0.0
    agree = 0
    positions = 0
    with safe_open(str(reference_path), "pt") as reference:
        metadata = dict(reference.metadata() or {})
        with torch.inference_mode():
            for index, prompt in enumerate(prompts):
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                ref_ids = reference.get_tensor(f"input_ids_{index}").tolist()
                if ids != ref_ids:
                    raise ValueError(
                        f"prompt {index}: native tokenizer ids differ from "
                        "the original tokenizer ids"
                    )
                ref_logits = reference.get_tensor(f"logits_{index}")
                logits = model(input_ids=torch.tensor([ids])).logits[0].float()
                if logits.shape != ref_logits.shape:
                    raise ValueError(
                        f"prompt {index}: logits shape {tuple(logits.shape)} "
                        f"!= reference {tuple(ref_logits.shape)}"
                    )
                diff = (logits - ref_logits).abs().max().item()
                max_diff = max(max_diff, float(diff))
                agree += int((logits.argmax(-1) == ref_logits.argmax(-1)).sum().item())
                positions += len(ids)
    return StrictCheckResult(
        prompts=len(prompts),
        positions=positions,
        max_abs_logit_diff=max_diff,
        argmax_agreement=agree / positions,
        reference_environment=metadata,
        native_transformers=transformers.__version__,
    )


def run_strict_check(
    source_dir: Path,
    native_dir: Path,
    prompts: tuple[str, ...] | None = None,
) -> StrictCheckResult:
    if prompts is None:
        prompts = load_checkpoint_contract().strict_check.prompts
    with tempfile.TemporaryDirectory() as tmp:
        reference_path = Path(tmp) / "reference.safetensors"
        reference_logits(source_dir, prompts, reference_path)
        return compare_with_reference(native_dir, prompts, reference_path)

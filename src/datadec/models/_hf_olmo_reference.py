"""Reference logits from an original hf_olmo checkpoint (throwaway env only).

Run as a script in an isolated environment that has ``ai2-olmo``; it must
not import datadec. ``strict_check.reference_logits`` builds the command::

    python -I _hf_olmo_reference.py SOURCE_DIR PROMPTS_JSON OUT_SAFETENSORS

It loads the checkpoint with ``hf_olmo`` on CPU in float32 (overriding
``flash_attention`` and ``init_device`` from the training config, which
name CUDA-only settings), runs each prompt without special tokens, and
writes ``input_ids_<i>`` and ``logits_<i>`` tensors plus version metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(source_dir: str, prompts_path: str, out_path: str) -> None:
    import hf_olmo  # noqa: F401, PLC0415  # ty: ignore[unresolved-import]
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from hf_olmo import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        OLMoConfig,
        OLMoForCausalLM,
    )
    from safetensors.torch import save_file  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    prompts = json.loads(Path(prompts_path).read_text())
    config = OLMoConfig.from_pretrained(source_dir)
    config.flash_attention = False
    config.init_device = "cpu"
    model = OLMoForCausalLM.from_pretrained(
        source_dir, config=config, torch_dtype=torch.float32
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    tensors = {}
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids=input_ids).logits[0].float()
            tensors[f"input_ids_{index}"] = input_ids[0].contiguous()
            tensors[f"logits_{index}"] = logits.contiguous()
    from importlib.metadata import version  # noqa: PLC0415

    save_file(
        tensors,
        out_path,
        metadata={
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "ai2_olmo": version("ai2-olmo"),
            "device": "cpu",
            "dtype": "float32",
        },
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])

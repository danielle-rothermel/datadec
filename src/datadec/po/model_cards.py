"""Plain-language model cards passed to the reflection LM during GEPA runs, and nominal parameter counts.

Families with dedicated cards: DataDecide (from the catalog), Pythia, OLMo 2, OLMoE, OLMo 3 / 3.1 and Qwen3."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[3] / "configs" / "catalog.toml"
OLMES_CFG = Path(__file__).resolve().parents[3] / "configs" / "olmes.toml"

_DD = re.compile(r"^allenai/DataDecide-(?P<recipe>.+)-(?P<size>\d+M|1B)$")


def _catalog():
    return tomllib.loads(CATALOG.read_text())


def _recipe_display(recipe_slug: str) -> str:
    recipe_map = tomllib.loads(OLMES_CFG.read_text())["recipe_map"]
    # model repo slugs use '_' for '.' and 'qc' for 'top' in some names; try a few spellings
    candidates = [recipe_slug, recipe_slug.replace("_", "."), recipe_slug.replace("qc-", "top-")]
    for c in candidates:
        if c in recipe_map:
            return recipe_map[c]
    return recipe_slug


_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)([MmBb])(?![A-Za-z])")


def nominal_params(model: str) -> float:
    """Nominal parameter count parsed from the hub id (the largest size token: OLMoE-1B-7B -> 7e9 total)."""
    return max(v for v, _ in _sizes(model))


def _sizes(model: str) -> list[tuple[float, str]]:
    sizes = [(float(v) * (1e6 if unit in "Mm" else 1e9), v + unit.upper()) for v, unit in _SIZE.findall(model.split("/")[-1])]
    if not sizes:
        raise ValueError(f"no size token in model id {model!r}")
    return sizes


_POST_TRAINING = {
    "SFT": "Post-trained with supervised fine-tuning (the SFT stage, before preference tuning)",
    "DPO": "Post-trained with supervised fine-tuning then DPO preference tuning (the DPO stage)",
    "RLVR1": "Post-trained with SFT, DPO, then reinforcement learning with verifiable rewards (the RLVR stage)",
    "Instruct": "The final instruction-tuned release of the post-training pipeline",
}
_PROMPT_NOTE = "Evaluated in the plain 5-shot base-model prompt format, no chat template; scored by likelihood over answer choices."


def _stage(name: str) -> str | None:
    for suffix in ("-Instruct-SFT", "-Instruct-DPO", "-SFT", "-DPO", "-RLVR1", "-Instruct"):
        if name.endswith(suffix):
            return suffix.rsplit("-", 1)[-1]
    return None


_FAMILY_PREFIXES = ("EleutherAI/pythia-", "allenai/OLMo-2-", "allenai/OLMoE-", "allenai/Olmo-3", "Qwen/Qwen3-")


def _family_card(model: str, revision: str | None) -> dict[str, str] | None:
    """Cards for the non-DataDecide families in the phase-2 model set; None for anything else."""
    if not model.startswith(_FAMILY_PREFIXES):
        return None
    name = model.split("/")[-1]
    stage = _stage(name)
    base = {"name": model, "revision": revision or "main", "nominal_parameters": _size_label(model)}
    stage_note = (_POST_TRAINING[stage] + ". ") if stage else "Base (not instruction-tuned) model. "
    if model.startswith("EleutherAI/pythia-"):
        step = revision[4:] if revision and revision.startswith("step") else None
        return base | {
            "family": "Pythia (EleutherAI) GPT-NeoX decoder-only transformer suite",
            "pretraining_data": "the Pile" + (" (deduplicated)" if "deduped" in name else "") + ", about 300B tokens",
            **({"checkpoint": f"training step {step} of 143000" + (" (final)" if step == "143000" else "")} if step else {}),
            "notes": "Base (not instruction-tuned) model released in 2023. " + _PROMPT_NOTE,
        }
    if model.startswith("allenai/OLMo-2-"):
        return base | {"family": "OLMo 2 (allenai), fully open decoder-only transformer, 2025 release",
                       "notes": stage_note + _PROMPT_NOTE}
    if model.startswith("allenai/OLMoE-"):
        return base | {"family": "OLMoE (allenai), sparse mixture-of-experts transformer, about 1B active of 7B total parameters, 2025 release",
                       "notes": stage_note + _PROMPT_NOTE}
    if model.startswith("allenai/Olmo-3"):
        family = "OLMo 3.1" if model.startswith("allenai/Olmo-3.1") else "OLMo 3"
        return base | {"family": f"{family} (allenai), fully open decoder-only transformer, 2025 release",
                       "notes": stage_note + _PROMPT_NOTE}
    note = "Base (not instruction-tuned) model. " if name.endswith("-Base") else "Instruction-tuned (post-trained) release. "
    return base | {"family": "Qwen3 (Alibaba), decoder-only transformer, 2025 release", "notes": note + _PROMPT_NOTE}


def _size_label(model: str) -> str:
    """The size token as the hub id spells it (largest one): 0.6B, 410M, 7B."""
    return max(_sizes(model))[1]


def model_card(model: str, revision: str | None) -> dict[str, str]:
    m = _DD.match(model)
    if not m:
        card = _family_card(model, revision)
        if card is not None:
            return card
        return {
            "name": model, "revision": revision or "main", "family": "unknown",
            "notes": "General-purpose pretrained model; not part of the DataDecide suite. " + _PROMPT_NOTE,
        }
    cat = _catalog()
    size = m.group("size")
    arch = next((e for e in cat["models"] if e["name"] == size), None)
    tr = cat["training"]
    card = {
        "name": model, "revision": revision or "main",
        "family": "OLMo-style decoder-only transformer from the DataDecide suite (allenai)",
        "nominal_parameters": size,
        "pretraining_data_recipe": _recipe_display(m.group("recipe")),
        "training_regime": f"trained to {tr['token_length_multiplier'] * 5} tokens per parameter (5x Chinchilla), sequence length {tr['max_sequence_length']}",
    }
    if arch:
        card["architecture"] = (f"{arch['n_layers']} layers, d_model {arch['d_model']}, {arch['n_heads']} heads, "
                                f"MLP ratio {arch['mlp_ratio']}, {arch['exact_parameter_count']:,} exact parameters")
    if revision and revision.startswith("step"):
        card["checkpoint"] = f"training step {revision.split('-')[0][4:]} ({'final' if True else ''} seed-default run)"
    card["notes"] = "Base model, no instruction tuning; scored by likelihood over answer choices in a 5-shot prompt."
    return card

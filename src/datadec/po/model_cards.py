"""Plain-language model cards passed to the reflection LM during GEPA runs."""

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


def model_card(model: str, revision: str | None) -> dict[str, str]:
    m = _DD.match(model)
    if not m:
        return {
            "name": model, "revision": revision or "main",
            "family": "Qwen3" if model.startswith("Qwen/") else "unknown",
            "notes": "General-purpose pretrained model; not part of the DataDecide suite. "
                     + ("Instruction-tuned, but evaluated here in the same base-model prompt format." if not model.endswith("-Base") else "Base (not instruction-tuned) model."),
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

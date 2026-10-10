"""Golden test: models.csv attributes derived from (model, revision, dtype, quant), one example per family/variant."""

from __future__ import annotations

import pytest

from datadec.po.model_registry import describe, model_key, precision

COLUMNS = ("model_key", "family", "display", "recipe", "size_label", "params", "seed_label", "variant", "stage_index",
           "checkpoint_step", "checkpoint_fraction", "checkpoint_kind", "precision", "era")
GOLDEN = [
    (("allenai/DataDecide-dclm-baseline-150M", "step37500-seed-default", "float32", None),
     ("allenai/DataDecide-dclm-baseline-150M@step37500-seed-default@fp32-tf32", "datadecide", "DCLM 150M", "dclm",
      "150M", 1.5e8, "default", "base", 0, 37500, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/DataDecide-dolma1_7-20M", "step14584-seed-small-aux-2", "float32", None),
     ("allenai/DataDecide-dolma1_7-20M@step14584-seed-small-aux-2@fp32-tf32", "datadecide", "Dolma 20M (small-aux-2)",
      "dolma", "20M", 2e7, "small-aux-2", "base", 0, 14584, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/DataDecide-c4-1B", "step35000-seed-default", "float32", None),
     ("allenai/DataDecide-c4-1B@step35000-seed-default@fp32-tf32", "datadecide", "C4 1B @mid", "c4", "1B", 1e9,
      "default", "base", 0, 35000, 35000 / 69359, "mid", "fp32-tf32", 2025)),
    (("allenai/DataDecide-dclm-baseline-1B", "step15000-seed-default", "float32", None),
     ("allenai/DataDecide-dclm-baseline-1B@step15000-seed-default@fp32-tf32", "datadecide", "DCLM 1B @peak", "dclm",
      "1B", 1e9, "default", "base", 0, 15000, 15000 / 69359, "peak", "fp32-tf32", 2025)),
    (("allenai/DataDecide-c4-4M", "step2500-seed-default", "float32", None),  # in both lists: mid wins
     ("allenai/DataDecide-c4-4M@step2500-seed-default@fp32-tf32", "datadecide", "C4 4M @mid", "c4", "4M", 4e6,
      "default", "base", 0, 2500, 2500 / 5715, "mid", "fp32-tf32", 2025)),
    (("EleutherAI/pythia-1.4b", "step143000", "float32", None),
     ("EleutherAI/pythia-1.4b@step143000@fp32-tf32", "pythia", "Pythia 1.4B", None, "1.4B", 1.4e9, None, "base", 0,
      143000, 1.0, "final", "fp32-tf32", 2023)),
    (("EleutherAI/pythia-160m", "step131000", "float32", None),
     ("EleutherAI/pythia-160m@step131000@fp32-tf32", "pythia", "Pythia 160M @peak", None, "160M", 1.6e8, None, "base",
      0, 131000, 131000 / 143000, "peak", "fp32-tf32", 2023)),
    (("Qwen/Qwen3-1.7B-Base", None, "float32", None),
     ("Qwen/Qwen3-1.7B-Base@main@fp32-tf32", "qwen3", "Qwen3 1.7B Base", None, "1.7B", 1.7e9, None, "base", 0, None,
      1.0, "final", "fp32-tf32", 2025)),
    (("Qwen/Qwen3-14B", None, "float32", None),
     ("Qwen/Qwen3-14B@main@fp32-tf32", "qwen3", "Qwen3 14B Instruct", None, "14B", 1.4e10, None, "instruct", 3, None,
      1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMo-2-0425-1B", None, "float32", None),
     ("allenai/OLMo-2-0425-1B@main@fp32-tf32", "olmo2", "OLMo-2 1B", None, "1B", 1e9, None, "base", 0, None, 1.0,
      "final", "fp32-tf32", 2025)),
    (("allenai/OLMo-2-0425-1B-SFT", None, "float32", None),
     ("allenai/OLMo-2-0425-1B-SFT@main@fp32-tf32", "olmo2", "OLMo-2 1B SFT", None, "1B", 1e9, None, "sft", 1, None,
      1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMo-2-0425-1B-DPO", None, "float32", None),
     ("allenai/OLMo-2-0425-1B-DPO@main@fp32-tf32", "olmo2", "OLMo-2 1B DPO", None, "1B", 1e9, None, "dpo", 2, None,
      1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMo-2-0425-1B-RLVR1", None, "float32", None),
     ("allenai/OLMo-2-0425-1B-RLVR1@main@fp32-tf32", "olmo2", "OLMo-2 1B RLVR1", None, "1B", 1e9, None, "rlvr", 3,
      None, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMo-2-0425-1B-Instruct", None, "float32", None),
     ("allenai/OLMo-2-0425-1B-Instruct@main@fp32-tf32", "olmo2", "OLMo-2 1B Instruct", None, "1B", 1e9, None,
      "instruct", 3, None, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMoE-1B-7B-0125", None, "float32", None),
     ("allenai/OLMoE-1B-7B-0125@main@fp32-tf32", "olmoe", "OLMoE 1B-7B", None, "1B-7B", 7e9, None, "base", 0, None,
      1.0, "final", "fp32-tf32", 2025)),
    (("allenai/OLMoE-1B-7B-0125-Instruct", None, "float32", None),
     ("allenai/OLMoE-1B-7B-0125-Instruct@main@fp32-tf32", "olmoe", "OLMoE 1B-7B Instruct", None, "1B-7B", 7e9, None,
      "instruct", 3, None, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/Olmo-3-1025-7B", None, "bfloat16", "nf4"),
     ("allenai/Olmo-3-1025-7B@main@nf4", "olmo3", "OLMo-3 7B [nf4]", None, "7B", 7e9, None, "base", 0, None, 1.0,
      "final", "nf4", 2025)),
    (("allenai/Olmo-3-7B-Instruct-SFT", None, "bfloat16", None),
     ("allenai/Olmo-3-7B-Instruct-SFT@main@bf16", "olmo3", "OLMo-3 7B SFT [bf16]", None, "7B", 7e9, None, "sft", 1,
      None, 1.0, "final", "bf16", 2025)),
    (("allenai/Olmo-3-7B-Instruct-DPO", None, "bfloat16", "int8"),
     ("allenai/Olmo-3-7B-Instruct-DPO@main@int8", "olmo3", "OLMo-3 7B DPO [int8]", None, "7B", 7e9, None, "dpo", 2,
      None, 1.0, "final", "int8", 2025)),
    (("allenai/Olmo-3-7B-Instruct", None, "float32", None),
     ("allenai/Olmo-3-7B-Instruct@main@fp32-tf32", "olmo3", "OLMo-3 7B Instruct", None, "7B", 7e9, None, "instruct",
      3, None, 1.0, "final", "fp32-tf32", 2025)),
    (("allenai/Olmo-3-1125-32B", None, "bfloat16", None),
     ("allenai/Olmo-3-1125-32B@main@bf16", "olmo3", "OLMo-3 32B [bf16]", None, "32B", 3.2e10, None, "base", 0, None,
      1.0, "final", "bf16", 2025)),
    (("allenai/Olmo-3.1-32B-Instruct-SFT", None, "bfloat16", None),
     ("allenai/Olmo-3.1-32B-Instruct-SFT@main@bf16", "olmo3.1", "OLMo-3.1 32B SFT [bf16]", None, "32B", 3.2e10, None,
      "sft", 1, None, 1.0, "final", "bf16", 2025)),
    (("allenai/Olmo-3.1-32B-Instruct-DPO", None, "bfloat16", None),
     ("allenai/Olmo-3.1-32B-Instruct-DPO@main@bf16", "olmo3.1", "OLMo-3.1 32B DPO [bf16]", None, "32B", 3.2e10, None,
      "dpo", 2, None, 1.0, "final", "bf16", 2025)),
    (("allenai/Olmo-3.1-32B-Instruct", None, "bfloat16", None),
     ("allenai/Olmo-3.1-32B-Instruct@main@bf16", "olmo3.1", "OLMo-3.1 32B Instruct [bf16]", None, "32B", 3.2e10, None,
      "instruct", 3, None, 1.0, "final", "bf16", 2025)),
]


@pytest.mark.parametrize(("ident", "expected"), GOLDEN, ids=[g[1][2] for g in GOLDEN])
def test_golden_attributes(ident: tuple, expected: tuple) -> None:
    row = describe(*ident).row()
    assert tuple(row[c] for c in COLUMNS) == pytest.approx(expected)
    assert row["model"] == ident[0]


def test_every_family_and_variant_is_pinned() -> None:
    combos = {(e[1], e[7]) for _, e in GOLDEN}
    assert combos >= {("datadecide", "base"), ("pythia", "base"), ("qwen3", "base"), ("qwen3", "instruct"),
                      ("olmo2", "base"), ("olmo2", "sft"), ("olmo2", "dpo"), ("olmo2", "rlvr"), ("olmo2", "instruct"),
                      ("olmoe", "base"), ("olmoe", "instruct"), ("olmo3", "base"), ("olmo3", "sft"), ("olmo3", "dpo"),
                      ("olmo3", "instruct"), ("olmo3.1", "sft"), ("olmo3.1", "dpo"), ("olmo3.1", "instruct")}


def test_model_key_slug() -> None:
    assert model_key("Qwen/Qwen3-4B", None, "bf16") == "Qwen/Qwen3-4B@main@bf16"


@pytest.mark.parametrize(("dtype", "quant", "expected"), [("float32", None, "fp32-tf32"), ("bfloat16", None, "bf16"),
                                                          ("bfloat16", "int8", "int8"), ("bfloat16", "nf4", "nf4")])
def test_precision(dtype: str, quant: str | None, expected: str) -> None:
    assert precision(dtype, quant) == expected


@pytest.mark.parametrize("ident", [
    ("ibm-granite/granite-4.0-micro", None, "float32", None),  # not registered (Granite dropped)
    ("allenai/DataDecide-dclm-baseline-1B", "step20000-seed-default", "float32", None),  # unlisted intermediate step
    ("Qwen/Qwen3-4B", "abc123", "float32", None),  # pinned revision of an unpinned family
    ("Qwen/Qwen3-4B", None, None, None),  # unknown dtype
    ("Qwen/Qwen3-4B", None, "float32", "int8"),  # quantized weights need bfloat16
])
def test_unknown_inputs_raise(ident: tuple) -> None:
    with pytest.raises(ValueError):
        describe(*ident)

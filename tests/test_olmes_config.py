from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    OLMESContract,
    OLMESTableContract,
    load_catalog,
    load_olmes_contract,
    load_source_manifest,
)

EXPECTED_RECIPE_MAP = {
    "c4": "C4",
    "dclm-baseline-25p-dolma1.7-75p": "DCLM-Baseline 25% / Dolma 75%",
    "dclm-baseline-50p-dolma1.7-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dclm-baseline-75p-dolma1.7-25p": "DCLM-Baseline 75% / Dolma 25%",
    "dclm-baseline-top-10p": "DCLM-Baseline (QC 10%)",
    "dclm-baseline-top-20p": "DCLM-Baseline (QC 20%)",
    "dclm-baseline-top-fw-10p": "DCLM-Baseline (QC FW 10%)",
    "dclm-baseline-top-fw-3p": "DCLM-Baseline (QC FW 3%)",
    "dclm-baseline-top-fw2-7p": "DCLM-Baseline (QC 7%, FW2)",
    "dclm-baseline-top-fw3-7p": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline": "DCLM-Baseline",
    "dolma1.6++": "Dolma1.6++",
    "dolma1.7-no-code": "Dolma1.7 (no code)",
    "dolma1.7-no-flan": "Dolma1.7 (no Flan)",
    "dolma1.7-no-math-no-code": "Dolma1.7 (no math, code)",
    "dolma1.7-no-reddit": "Dolma1.7 (no Reddit)",
    "dolma1.7": "Dolma1.7",
    "falcon-with-cc-top-10p": "Falcon+CC (QC 10%)",
    "falcon-with-cc-top-20p": "Falcon+CC (QC 20%)",
    "falcon-with-cc-top-orig-10p": "Falcon+CC (QC Orig 10%)",
    "falcon-with-cc-top-tulu-10p": "Falcon+CC (QC Tulu 10%)",
    "falcon-with-cc": "Falcon+CC",
    "falcon": "Falcon",
    "fineweb-edu": "FineWeb-Edu",
    "fineweb-pro": "FineWeb-Pro",
}

AGGREGATE_METRICS = (
    "correct_choice",
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "acc_per_byte",
    "acc_uncond",
    "no_answer",
    "sum_logits_corr",
    "logits_per_token_corr",
    "logits_per_char_corr",
    "bits_per_byte_corr",
    "logits_per_byte_corr",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
    "uncond_correct_prob",
    "uncond_correct_prob_per_token",
    "uncond_correct_prob_per_char",
    "uncond_total_prob",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
    "primary_metric",
)

DETAILED_TASK_METRICS = (
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "acc_uncond",
    "primary_score",
    "logits_per_byte_corr",
    "logits_per_char_corr",
    "no_answer",
)

DETAILED_INSTANCE_METRICS = (
    "predicted_index_raw",
    "predicted_index_per_token",
    "predicted_index_per_char",
    "predicted_index_per_byte",
    "predicted_index_uncond",
    "correct_choice",
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "acc_per_byte",
    "acc_uncond",
    "no_answer",
    "sum_logits_corr",
    "logits_per_token_corr",
    "logits_per_char_corr",
    "logits_per_byte_corr",
)

DETAILED_CHOICE_METRICS = (
    "sum_logits",
    "num_tokens",
    "num_tokens_all",
    "is_greedy",
    "sum_logits_uncond",
    "logits_per_token",
    "logits_per_char",
    "logits_per_byte",
    "num_chars",
)

EXPECTED_TABLES = {
    "aggregate": {
        "path": "processed/olmes.parquet",
        "primary_key": ("params", "data", "seed", "step", "task"),
        "columns": (
            ("params", "string", False),
            ("data", "string", False),
            ("seed", "string", False),
            ("step", "int64", False),
            ("task", "string", False),
            ("chinchilla", "string", False),
            ("tokens", "int64", False),
            ("compute", "float64", False),
            *(
                (name, logical_type, False)
                for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES[2:]
            ),
            *((name, "float64", True) for name in AGGREGATE_METRICS),
        ),
    },
    "detailed_tasks": {
        "path": "processed/olmes-details/{recipe}/tasks.parquet",
        "primary_key": ("recipe", "params", "seed_value", "step", "task"),
        "columns": (
            ("recipe", "string", False),
            ("data", "string", False),
            ("params", "string", False),
            ("seed_value", "int64", False),
            ("seed", "string", False),
            ("step", "int64", False),
            *(
                (name, logical_type, False)
                for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES
            ),
            ("task", "string", False),
            ("task_hash", "string", False),
            ("model_hash", "string", False),
            ("model_config", "string", False),
            ("task_config", "string", False),
            ("compute_config", "string", False),
            ("processing_time", "float64", False),
            ("current_date", "string", False),
            ("num_instances", "int64", False),
            ("task_idx", "int64", False),
            ("primary_metric", "string", False),
            ("acc_raw", "float64", False),
            ("acc_per_token", "float64", False),
            ("acc_per_char", "float64", False),
            ("acc_uncond", "float64", True),
            ("primary_score", "float64", False),
            ("logits_per_byte_corr", "float64", True),
            ("logits_per_char_corr", "float64", True),
            ("no_answer", "float64", True),
        ),
    },
    "detailed_instances": {
        "path": "processed/olmes-details/{recipe}/instances.parquet",
        "primary_key": (
            "recipe",
            "params",
            "seed_value",
            "step",
            "task",
            "doc_id",
        ),
        "columns": (
            ("recipe", "string", False),
            ("data", "string", False),
            ("params", "string", False),
            ("seed_value", "int64", False),
            ("seed", "string", False),
            ("step", "int64", False),
            ("task", "string", False),
            ("task_hash", "string", False),
            ("model_hash", "string", False),
            ("doc_id", "int64", False),
            ("native_id", "string", True),
            ("native_id_kind", "string", False),
            ("label", "int64", False),
            ("predicted_index_raw", "int64", False),
            ("predicted_index_per_token", "int64", False),
            ("predicted_index_per_char", "int64", False),
            ("predicted_index_per_byte", "int64", True),
            ("predicted_index_uncond", "int64", True),
            ("correct_choice", "int64", False),
            ("acc_raw", "int64", False),
            ("acc_per_token", "int64", False),
            ("acc_per_char", "int64", False),
            ("acc_per_byte", "int64", True),
            ("acc_uncond", "int64", True),
            ("no_answer", "int64", True),
            ("sum_logits_corr", "float64", True),
            ("logits_per_token_corr", "float64", True),
            ("logits_per_char_corr", "float64", True),
            ("logits_per_byte_corr", "float64", True),
        ),
    },
    "detailed_choices": {
        "path": "processed/olmes-details/{recipe}/choices.parquet",
        "primary_key": (
            "recipe",
            "params",
            "seed_value",
            "step",
            "task",
            "doc_id",
            "choice_index",
        ),
        "columns": (
            ("recipe", "string", False),
            ("data", "string", False),
            ("params", "string", False),
            ("seed_value", "int64", False),
            ("seed", "string", False),
            ("step", "int64", False),
            ("task", "string", False),
            ("doc_id", "int64", False),
            ("choice_index", "int64", False),
            ("sum_logits", "float64", False),
            ("num_tokens", "int64", False),
            ("num_tokens_all", "int64", False),
            ("is_greedy", "bool", False),
            ("sum_logits_uncond", "float64", True),
            ("logits_per_token", "float64", False),
            ("logits_per_char", "float64", False),
            ("logits_per_byte", "float64", True),
            ("num_chars", "int64", False),
        ),
    },
}


def test_olmes_contract_pins_identity_and_metric_reconciliation() -> None:
    contract = load_olmes_contract()

    assert contract.recipe_map == EXPECTED_RECIPE_MAP
    assert contract.seed_map == {
        2: "default",
        4: "large aux 2",
        5: "large aux 3",
        14: "small aux 2",
        15: "small aux 3",
    }
    assert contract.identity.native_id_kinds == ("integer", "string", "null")
    assert contract.metrics.aggregate == AGGREGATE_METRICS
    assert contract.metrics.detailed_tasks == DETAILED_TASK_METRICS
    assert contract.metrics.detailed_instances == DETAILED_INSTANCE_METRICS
    assert contract.metrics.detailed_choices == DETAILED_CHOICE_METRICS
    assert contract.metrics.not_reproducible_from_details == ("bits_per_byte_corr",)
    assert contract.metrics.aggregate_primary_metric.model_dump() == {
        "mmlu": "acc_raw",
        "arc_challenge": "acc_uncond",
        "arc_easy": "acc_per_char",
        "boolq": "acc_raw",
        "csqa": "acc_uncond",
        "hellaswag": "acc_per_char",
        "openbookqa": "acc_uncond",
        "piqa": "acc_per_char",
        "socialiqa": "acc_per_char",
        "winogrande": "acc_raw",
    }
    assert contract.metrics.detailed_primary_metric_source == (
        "task_config.primary_metric"
    )
    assert contract.metrics.detailed_primary_metric_column == "primary_score"


@pytest.mark.parametrize(("table_name", "expected"), EXPECTED_TABLES.items())
def test_olmes_contract_pins_table_schemas(
    table_name: str, expected: dict[str, Any]
) -> None:
    table = getattr(load_olmes_contract().tables, table_name)

    assert (table.path or table.path_template) == expected["path"]
    assert table.primary_key == expected["primary_key"]
    assert table.sort_key == expected["primary_key"]
    assert (
        tuple(
            (column.name, column.logical_type, column.nullable)
            for column in table.columns
        )
        == expected["columns"]
    )

    if table_name != "aggregate":
        assert "seed_value" in table.primary_key
        assert "seed" not in table.primary_key


def test_olmes_table_rejects_duplicate_and_missing_key_columns() -> None:
    raw = load_olmes_contract().tables.aggregate.model_dump()
    raw["columns"] = (*raw["columns"], raw["columns"][0])
    with pytest.raises(ValidationError, match="column names must be unique"):
        OLMESTableContract.model_validate(raw)

    raw = load_olmes_contract().tables.aggregate.model_dump()
    raw["sort_key"] = (*raw["sort_key"], "missing")
    with pytest.raises(ValidationError, match="sort key columns are missing"):
        OLMESTableContract.model_validate(raw)


def test_olmes_contract_rejects_mapping_and_metric_drift() -> None:
    raw = load_olmes_contract().model_dump()
    raw["recipe_map"]["c4"] = raw["recipe_map"]["falcon"]
    with pytest.raises(ValidationError, match="recipe mappings must be unique"):
        OLMESContract.model_validate(raw)

    raw = load_olmes_contract().model_dump()
    task_metrics = raw["metrics"]["detailed_tasks"]
    raw["metrics"]["detailed_tasks"] = (
        *task_metrics[:-2],
        task_metrics[-1],
        task_metrics[-2],
    )
    with pytest.raises(ValidationError, match="must match table column order"):
        OLMESContract.model_validate(raw)

    raw = load_olmes_contract().model_dump()
    raw["tables"]["aggregate"]["columns"][-1]["nullable"] = False
    with pytest.raises(ValidationError, match="nullable float64"):
        OLMESContract.model_validate(raw)

    raw = load_olmes_contract().model_dump()
    raw["tables"]["detailed_tasks"]["primary_key"] = (
        "recipe",
        "params",
        "seed",
        "step",
        "task",
    )
    raw["tables"]["detailed_tasks"]["sort_key"] = raw["tables"]["detailed_tasks"][
        "primary_key"
    ]
    with pytest.raises(ValidationError, match="use seed_value keys"):
        OLMESContract.model_validate(raw)


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ({"recipe_map": {"c4": None}}, "exactly cover source detail recipes"),
        ({"recipe_map": {"c4": "not a catalog recipe"}}, "catalog recipes"),
        ({"seed_map": {2: "not a catalog seed"}}, "catalog seeds"),
    ],
)
def test_olmes_contract_rejects_invalid_catalog_or_source_references(
    mapping: dict[str, dict[str | int, str | None]], error: str
) -> None:
    raw = load_olmes_contract().model_dump()
    for field, updates in mapping.items():
        for key, value in updates.items():
            if value is None:
                del raw[field][key]
            else:
                raw[field][key] = value
    contract = OLMESContract.model_validate(raw)

    with pytest.raises(ValueError, match=error):
        contract.validate_references(
            catalog=load_catalog(), source_manifest=load_source_manifest()
        )

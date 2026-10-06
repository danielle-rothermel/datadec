from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    OLMESContract,
    PublishedResultsManifest,
    ScalingLawContract,
    config_file,
    load_catalog,
    load_olmes_contract,
    load_published_results_manifest,
    load_scaling_law_contract,
)
from datadec.data.paths import DataDecidePaths

SOURCE_PRECEDENCE = (
    "results_ladder_5xC_seeds.csv",
    "results_ladder_5xC_small_seed_extras.csv",
    "results_ladder_5xC_small_seeds_extra_real.csv",
)

EXCLUDED_SOURCE_GROUPS = (
    "DCLM-baseline-25p",
    "DCLM-baseline-4M-5xC",
    "DCLM-baseline-50p",
    "DCLM-baseline-75p",
)

SOURCE_GROUP_MAP = {
    "DCLM-baseline": "dclm-baseline",
    "c4": "c4",
    "dclm_ft7percentile_fw2": "dclm-baseline-top-fw2-7p",
    "dclm_ft7percentile_fw3": "dclm-baseline-top-fw3-7p",
    "dclm_fw_top10": "dclm-baseline-top-fw-10p",
    "dclm_fw_top3": "dclm-baseline-top-fw-3p",
    "dolma-v1-6-and-sources-baseline": "dolma1.6++",
    "dolma17": "dolma1.7",
    "dolma17-25p-DCLM-baseline-75p": "dclm-baseline-75p-dolma1.7-25p",
    "dolma17-50p-DCLM-baseline-50p": "dclm-baseline-50p-dolma1.7-50p",
    "dolma17-75p-DCLM-baseline-25p": "dclm-baseline-25p-dolma1.7-75p",
    "falcon": "falcon",
    "falcon_and_cc": "falcon-with-cc",
    "falcon_and_cc_eli5_oh_top10p": "falcon-with-cc-top-10p",
    "falcon_and_cc_eli5_oh_top20p": "falcon-with-cc-top-20p",
    "falcon_and_cc_og_eli5_oh_top10p": "falcon-with-cc-top-orig-10p",
    "falcon_and_cc_tulu_qc_top10": "falcon-with-cc-top-tulu-10p",
    "fineweb_edu_dedup": "fineweb-edu",
    "no_code": "dolma1.7-no-code",
    "no_flan": "dolma1.7-no-flan",
    "no_math_no_code": "dolma1.7-no-math-no-code",
    "no_reddit": "dolma1.7-no-reddit",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": (
        "dclm-baseline-top-10p"
    ),
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": (
        "dclm-baseline-top-20p"
    ),
    "prox_fineweb_pro": "fineweb-pro",
}

EVALUATION_IDENTITY_COLUMNS = (
    ("source_file", "string", False),
    ("recipe", "string", False),
    ("data", "string", False),
    ("params", "string", False),
    ("seed_value", "int64", False),
    ("seed", "string", False),
    ("step", "int64", False),
    ("task", "string", False),
    ("chinchilla", "string", False),
    ("tokens", "int64", False),
    ("compute", "float64", False),
)

CHECKPOINT_LOSS_COLUMNS = (
    "c4_en_validation_cross_entropy",
    "dolma_common_crawl_validation_cross_entropy",
    "pile_validation_cross_entropy",
    "wikitext_103_validation_cross_entropy",
    "train_cross_entropy",
    "throughput_total_tokens",
)

MODEL_PARAMETER_COUNTS = {
    "4M": 3_744_832,
    "6M": 6_010_464,
    "8M": 8_538_240,
    "10M": 9_900_432,
    "14M": 14_380_224,
    "16M": 16_004_560,
    "20M": 19_101_888,
    "60M": 57_078_144,
    "90M": 97_946_640,
    "150M": 151_898_880,
    "300M": 319_980_544,
    "530M": 530_074_944,
    "750M": 681_297_408,
    "1B": 1_176_832_000,
}


def _column_tuples(contract: object) -> tuple[tuple[str, str, bool], ...]:
    columns = getattr(contract, "columns")
    return tuple(
        (column.name, column.logical_type, column.nullable) for column in columns
    )


def test_scaling_law_contract_pins_inputs_aliases_models_and_seed_policy() -> None:
    contract = load_scaling_law_contract()
    catalog = load_catalog()
    olmes = load_olmes_contract()

    assert config_file("scaling_law.toml").is_file()
    assert contract.raw_directory == "raw/scaling-law"
    assert contract.source_precedence == SOURCE_PRECEDENCE
    assert contract.models == tuple(model.name for model in catalog.models)
    assert contract.excluded_source_groups == EXCLUDED_SOURCE_GROUPS
    assert contract.source_group_aliases == {"baseline": "dolma17"}
    assert contract.source_group_map == SOURCE_GROUP_MAP
    assert set(contract.source_group_map.values()) == set(olmes.recipe_map)
    assert {
        olmes.recipe_map[recipe] for recipe in contract.source_group_map.values()
    } == {
        recipe for family in catalog.data_recipe_families.values() for recipe in family
    }
    assert contract.seed_map == {
        2: "default",
        4: "large aux 2",
        5: "large aux 3",
        14: "small aux 2",
        15: "small aux 3",
    }
    assert contract.seed_policy.excluded_legacy_values == (6198,)
    assert contract.seed_policy.missing == "exclude_legacy_input"
    assert contract.seed_policy.unknown_non_null == "error"
    catalog = load_catalog()
    assert catalog.training.flops_per_token_per_parameter == 6
    assert {model.name: model.exact_parameter_count for model in catalog.models} == (
        MODEL_PARAMETER_COUNTS
    )
    with pytest.raises(ValidationError, match="frozen"):
        setattr(contract, "raw_directory", "elsewhere")


def test_scaling_law_evaluations_match_olmes_aggregate_metric_contract() -> None:
    scaling_law = load_scaling_law_contract()
    olmes = load_olmes_contract()
    table = scaling_law.tables.evaluations
    columns = _column_tuples(table)
    aggregate_columns = {
        column.name: (column.logical_type, column.nullable)
        for column in olmes.tables.aggregate.columns
    }

    assert table.path == "processed/scaling-law/evaluations.parquet"
    assert table.primary_key == ("recipe", "params", "seed_value", "step", "task")
    assert table.sort_key == table.primary_key
    assert columns[:11] == EVALUATION_IDENTITY_COLUMNS
    enrichment_columns = tuple(
        (name, logical_type, False)
        for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES[2:]
    )
    assert columns[11 : 11 + len(enrichment_columns)] == enrichment_columns
    metric_columns = columns[-len(olmes.metrics.aggregate) :]
    assert tuple(name for name, _, _ in metric_columns) == olmes.metrics.aggregate
    assert tuple(
        (logical_type, nullable) for _, logical_type, nullable in metric_columns
    ) == tuple(aggregate_columns[name] for name in olmes.metrics.aggregate)
    assert columns[-1] == ("primary_metric", "float64", True)
    assert "source_file" not in table.primary_key


def test_scaling_law_checkpoint_losses_pin_nullable_metrics_and_identity() -> None:
    table = load_scaling_law_contract().tables.checkpoint_losses
    columns = _column_tuples(table)

    assert table.path == "processed/scaling-law/checkpoint-losses.parquet"
    assert table.primary_key == ("recipe", "params", "seed_value", "step")
    assert table.sort_key == table.primary_key
    assert columns[:10] == tuple(
        column for column in EVALUATION_IDENTITY_COLUMNS if column[0] != "task"
    )
    enrichment_columns = tuple(
        (name, logical_type, False)
        for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES[2:]
    )
    assert columns[10 : 10 + len(enrichment_columns)] == enrichment_columns
    assert columns[-len(CHECKPOINT_LOSS_COLUMNS) :] == tuple(
        (name, "float64", True) for name in CHECKPOINT_LOSS_COLUMNS
    )
    assert "source_file" not in table.primary_key


def test_scaling_law_paths_follow_contract_without_creating_directories(
    tmp_path: Path,
) -> None:
    paths = DataDecidePaths(tmp_path)

    assert list(tmp_path.iterdir()) == []
    assert paths.scaling_law_raw_paths() == tuple(
        tmp_path / "raw/scaling-law" / filename for filename in SOURCE_PRECEDENCE
    )
    assert paths.scaling_law_evaluations_path() == (
        tmp_path / "processed/scaling-law/evaluations.parquet"
    )
    assert paths.scaling_law_checkpoint_losses_path() == (
        tmp_path / "processed/scaling-law/checkpoint-losses.parquet"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda raw: raw.update(
                {
                    "source_precedence": raw["source_precedence"]
                    + (raw["source_precedence"][0],)
                }
            ),
            "source precedence must be unique",
        ),
        (
            lambda raw: raw["source_group_map"].update(
                {"c4": raw["source_group_map"]["DCLM-baseline"]}
            ),
            "source group mappings must be unique",
        ),
        (
            lambda raw: raw["source_group_aliases"].update(
                {"legacy": "not-a-canonical-group"}
            ),
            "aliases must reference canonical groups",
        ),
        (
            lambda raw: raw.update(
                {
                    "excluded_source_groups": raw["excluded_source_groups"]
                    + ("DCLM-baseline-25p",)
                }
            ),
            "excluded source groups must be unique",
        ),
        (
            lambda raw: raw["seed_policy"].update({"excluded_legacy_values": (2,)}),
            "excluded legacy seeds must contain only seed 6198",
        ),
        (
            lambda raw: raw["tables"]["evaluations"].update(
                {"primary_key": ("recipe", "params", "seed", "step", "task")}
            ),
            "primary and sort keys must match identity",
        ),
        (
            lambda raw: raw["tables"]["evaluations"]["columns"][4].update(
                {"nullable": True}
            ),
            "identity and provenance columns are invalid",
        ),
        (
            lambda raw: raw["tables"]["checkpoint_losses"]["columns"][-1].update(
                {"nullable": False}
            ),
            "loss metrics must be nullable float64",
        ),
    ],
)
def test_scaling_law_contract_rejects_invalid_local_contracts(
    mutate: Any, error: str
) -> None:
    raw = load_scaling_law_contract().model_dump()
    mutate(raw)

    with pytest.raises(ValidationError, match=error):
        ScalingLawContract.model_validate(raw)


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        ("manifest", "published results manifest"),
        ("models", "exactly match catalog models"),
        ("recipes", "bijectively to OLMES recipes"),
        ("seeds", "exactly match OLMES seeds"),
        ("metrics", "exactly match OLMES aggregate metrics"),
        ("metric_types", "metric types must match OLMES aggregate columns"),
    ],
)
def test_scaling_law_contract_rejects_invalid_external_references(
    reference: str, error: str
) -> None:
    catalog = load_catalog()
    olmes = load_olmes_contract()
    manifest = load_published_results_manifest()
    scaling_raw = load_scaling_law_contract().model_dump()

    if reference == "manifest":
        manifest_raw = manifest.model_dump()
        manifest_raw["files"] = manifest_raw["files"][1:]
        manifest = PublishedResultsManifest.model_validate(manifest_raw)
    elif reference == "models":
        scaling_raw["models"] = scaling_raw["models"][:-1]
    elif reference == "recipes":
        del scaling_raw["source_group_map"]["c4"]
    elif reference == "seeds":
        scaling_raw["seed_map"][2] = "not default"
    elif reference == "metrics":
        columns = scaling_raw["tables"]["evaluations"]["columns"]
        scaling_raw["tables"]["evaluations"]["columns"] = columns[:-2] + tuple(
            reversed(columns[-2:])
        )
    elif reference == "metric_types":
        scaling_raw["tables"]["evaluations"]["columns"][-1]["nullable"] = False

    contract = ScalingLawContract.model_validate(scaling_raw)
    with pytest.raises(ValueError, match=error):
        contract.validate_references(
            catalog=catalog,
            olmes_contract=OLMESContract.model_validate(olmes.model_dump()),
            published_results_manifest=manifest,
        )

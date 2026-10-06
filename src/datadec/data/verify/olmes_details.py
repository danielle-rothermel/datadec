from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from datadec.config import OLMESContract, load_olmes_contract
from datadec.data.artifacts import DataArtifacts

_CHECKPOINT_MEMBER_RE = re.compile(
    r"^(?P<recipe>[^/]+)/(?P<params>[^/]+)/seed-(?P<seed_value>\d+)/step-(?P<step>\d+)\.tar\.gz$"
)
_FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class OlmesVerificationResult:
    overlapping_checkpoint_count: int | None
    parity_row_count: int | None
    reconstructed_task_count: int


def _seed_name_to_value(contract: OLMESContract) -> dict[str, int]:
    return {
        seed_name: seed_value for seed_value, seed_name in contract.seed_map.items()
    }


def detail_checkpoint_keys(
    detail_archive: Path,
    *,
    recipe: str,
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    with tarfile.open(detail_archive, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = _CHECKPOINT_MEMBER_RE.match(member.name)
            if match is None or match.group("recipe") != recipe:
                continue
            keys.add(
                (
                    match.group("params"),
                    int(match.group("seed_value")),
                    int(match.group("step")),
                )
            )
    return keys


def aggregate_checkpoint_keys(
    aggregate_df: pd.DataFrame,
    *,
    data: str,
    contract: OLMESContract,
) -> set[tuple[str, int, int]]:
    seed_name_to_value = _seed_name_to_value(contract)
    filtered = aggregate_df[aggregate_df["data"] == data]
    keys: set[tuple[str, int, int]] = set()
    for row in filtered.itertuples(index=False):
        seed_value = seed_name_to_value.get(str(row.seed))
        if seed_value is None:
            continue
        keys.add((str(row.params), seed_value, int(row.step)))
    return keys


def overlapping_checkpoints(
    *,
    recipe: str,
    detail_archive: Path,
    aggregate_df: pd.DataFrame,
    contract: OLMESContract | None = None,
) -> set[tuple[str, int, int]]:
    contract = contract or load_olmes_contract()
    if recipe not in contract.recipe_map:
        raise ValueError(f"unknown OLMES detail recipe: {recipe!r}")
    detail_keys = detail_checkpoint_keys(detail_archive, recipe=recipe)
    aggregate_keys = aggregate_checkpoint_keys(
        aggregate_df,
        data=contract.recipe_map[recipe],
        contract=contract,
    )
    return detail_keys & aggregate_keys


def verify_cross_source_parity(
    *,
    recipe: str,
    tasks_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    overlapping: set[tuple[str, int, int]],
    contract: OLMESContract | None = None,
    rtol: float = _FLOAT_TOLERANCE,
    atol: float = _FLOAT_TOLERANCE,
) -> int:
    contract = contract or load_olmes_contract()
    overlap = overlapping
    if not overlap:
        return 0

    seed_name_to_value = _seed_name_to_value(contract)
    data = contract.recipe_map[recipe]
    aggregate = aggregate_df[aggregate_df["data"] == data].copy()
    aggregate["seed_value"] = aggregate["seed"].map(seed_name_to_value)
    aggregate = aggregate.dropna(subset=["seed_value"])
    aggregate["seed_value"] = aggregate["seed_value"].astype("int64")

    detail = tasks_df[tasks_df["recipe"] == recipe].copy()
    detail["checkpoint_key"] = list(
        zip(detail["params"], detail["seed_value"], detail["step"], strict=True)
    )
    aggregate["checkpoint_key"] = list(
        zip(
            aggregate["params"], aggregate["seed_value"], aggregate["step"], strict=True
        )
    )

    detail = detail[detail["checkpoint_key"].isin(overlap)]
    aggregate = aggregate[aggregate["checkpoint_key"].isin(overlap)]
    aggregate_values = aggregate[
        ["params", "seed_value", "step", "task", "primary_metric"]
    ].rename(columns={"primary_metric": "aggregate_primary_metric"})
    merged = detail.merge(
        aggregate_values,
        on=["params", "seed_value", "step", "task"],
        how="inner",
    )
    if merged.empty:
        return 0

    mismatches = ~np.isclose(
        merged["primary_score"].astype(float),
        merged["aggregate_primary_metric"].astype(float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    if mismatches.any():
        sample = merged.loc[mismatches, ["params", "seed_value", "step", "task"]].head()
        raise AssertionError(
            "aggregate/detail primary metric mismatch on overlapping checkpoints: "
            f"{sample.to_dict(orient='records')}"
        )
    return len(merged)


def verify_reconstructed_task_metrics(
    *,
    tasks_df: pd.DataFrame,
    instances_df: pd.DataFrame,
    contract: OLMESContract | None = None,
    rtol: float = _FLOAT_TOLERANCE,
    atol: float = _FLOAT_TOLERANCE,
) -> int:
    contract = contract or load_olmes_contract()
    group_cols = ["recipe", "params", "seed_value", "step", "task"]
    reproducible_metrics = [
        metric
        for metric in contract.metrics.detailed_tasks
        if metric in contract.metrics.detailed_instances
        and metric not in contract.metrics.not_reproducible_from_details
        and metric != "primary_score"
        and metric in tasks_df.columns
        and metric in instances_df.columns
    ]
    assert "bits_per_byte_corr" in contract.metrics.not_reproducible_from_details
    assert "bits_per_byte_corr" not in reproducible_metrics

    reconstructed = instances_df.groupby(group_cols, as_index=False)[
        reproducible_metrics
    ].mean(numeric_only=True)
    task_metrics = tasks_df[group_cols + reproducible_metrics]
    merged = task_metrics.merge(
        reconstructed, on=group_cols, suffixes=("_task", "_reconstructed")
    )

    for metric in reproducible_metrics:
        task_col = f"{metric}_task"
        reconstructed_col = f"{metric}_reconstructed"
        if task_col not in merged or reconstructed_col not in merged:
            continue
        task_values = merged[task_col].astype(float)
        reconstructed_values = merged[reconstructed_col].astype(float)
        both_null = task_values.isna() & reconstructed_values.isna()
        comparable = ~both_null
        if not comparable.any():
            continue
        if not np.allclose(
            task_values[comparable],
            reconstructed_values[comparable],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            raise AssertionError(
                f"reconstructed task metric mismatch for {metric!r} "
                f"on {merged.loc[comparable, group_cols].head().to_dict(orient='records')}"
            )

    primary_lookup = tasks_df[group_cols + ["primary_metric", "primary_score"]]
    primary_instances = instances_df.merge(primary_lookup, on=group_cols, how="inner")
    primary_rows: list[float] = []
    primary_expected: list[float] = []
    for _, group in primary_instances.groupby(group_cols, sort=False):
        primary_metric = str(group["primary_metric"].iloc[0])
        if primary_metric not in group.columns:
            continue
        primary_rows.append(float(group[primary_metric].mean()))
        primary_expected.append(float(group["primary_score"].iloc[0]))
    if primary_rows and not np.allclose(
        np.asarray(primary_expected, dtype=np.float64),
        np.asarray(primary_rows, dtype=np.float64),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        raise AssertionError(
            "reconstructed primary_score mismatch from instance metrics"
        )

    return len(reconstructed)


def verify_detail_counts(
    *,
    tasks_df: pd.DataFrame,
    instances_df: pd.DataFrame,
    choices_df: pd.DataFrame,
) -> None:
    task_key = ["recipe", "params", "seed_value", "step", "task"]
    instance_key = [*task_key, "doc_id"]
    choice_key = [*task_key, "doc_id", "choice_index"]

    if tasks_df[task_key].duplicated().any():
        raise AssertionError("duplicate task primary keys in detailed tasks output")
    if instances_df[instance_key].duplicated().any():
        raise AssertionError(
            "duplicate instance primary keys in detailed instances output"
        )
    if choices_df[choice_key].duplicated().any():
        raise AssertionError("duplicate choice primary keys in detailed choices output")

    expected_instances = int(tasks_df["num_instances"].sum())
    if len(instances_df) != expected_instances:
        raise AssertionError(
            f"instance row count mismatch: expected={expected_instances}, "
            f"actual={len(instances_df)}"
        )


def verify_olmes_details(
    *,
    recipe: str,
    paths: DataArtifacts,
    detail_archive: Path | None,
    aggregate_path: Path | None,
    contract: OLMESContract | None = None,
) -> OlmesVerificationResult:
    contract = contract or load_olmes_contract()
    tasks_df = pd.read_parquet(paths.olmes_details_tasks_path(recipe))
    instances_df = pd.read_parquet(paths.olmes_details_instances_path(recipe))
    choices_df = pd.read_parquet(paths.olmes_details_choices_path(recipe))
    verify_detail_counts(
        tasks_df=tasks_df,
        instances_df=instances_df,
        choices_df=choices_df,
    )

    reconstructed_rows = verify_reconstructed_task_metrics(
        tasks_df=tasks_df,
        instances_df=instances_df,
        contract=contract,
    )
    if aggregate_path is None:
        return OlmesVerificationResult(
            overlapping_checkpoint_count=None,
            parity_row_count=None,
            reconstructed_task_count=reconstructed_rows,
        )
    if detail_archive is None:
        raise ValueError(
            "detail_archive is required for aggregate/detail cross-source parity"
        )

    aggregate_df = pd.read_parquet(aggregate_path)
    overlap = overlapping_checkpoints(
        recipe=recipe,
        detail_archive=detail_archive,
        aggregate_df=aggregate_df,
        contract=contract,
    )
    parity_rows = verify_cross_source_parity(
        recipe=recipe,
        tasks_df=tasks_df,
        aggregate_df=aggregate_df,
        overlapping=overlap,
        contract=contract,
    )
    return OlmesVerificationResult(len(overlap), parity_rows, reconstructed_rows)


__all__ = [
    "OlmesVerificationResult",
    "aggregate_checkpoint_keys",
    "detail_checkpoint_keys",
    "overlapping_checkpoints",
    "verify_cross_source_parity",
    "verify_detail_counts",
    "verify_olmes_details",
    "verify_reconstructed_task_metrics",
]

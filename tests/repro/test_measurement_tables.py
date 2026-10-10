from io import StringIO

import pandas as pd
import pyarrow.parquet as pq
import pytest
from rich.console import Console

from repro.datasets import read_claim_evidence, write_claim_evidence
from repro.evidence import collect_claim_evidence
from repro.measurement_runner import write_measurement_tables
from repro.numerical_tables import load_linked_tables, print_linked_table
from repro.results import ClaimEvidence
from repro.scaling_measurements import released_prediction_errors
from repro.table_evidence import EvidenceTable, TableEvidence


def test_persisted_table_names_are_pinned():
    assert {table.value for table in EvidenceTable} == {
        "budget_accuracy.parquet",
        "recipe_scores.parquet",
        "score_summaries.parquet",
        "noise_spread.parquet",
        "scaling_observations.parquet",
        "scaling_errors.parquet",
        "scaling_summaries.parquet",
        "curves.csv",
        "proxy_comparisons.parquet",
        "matched_compute.csv",
        "recipe_crossovers.csv",
    }


def test_table_only_evidence_round_trips_with_pinned_schema(tmp_path):
    result = ClaimEvidence(
        "DD-0098", (), (), (TableEvidence(EvidenceTable.NOISE_SPREAD, (2, 7)),)
    )
    write_claim_evidence((result,), tmp_path / "claims.parquet")
    assert read_claim_evidence(tmp_path / "claims.parquet") == (result,)
    assert result.has_measurements
    assert pq.read_table(tmp_path / "claims.parquet").to_pylist() == [
        {
            "claim_id": "DD-0098",
            "tasks": [],
            "related_tables": [],
            "table_links": [{"table": "noise_spread.parquet", "row_ids": [2, 7]}],
        }
    ]


def test_link_loader_keeps_explicit_parquet_ids_and_rejects_missing_rows(tmp_path):
    kind = EvidenceTable.NOISE_SPREAD
    frame = pd.DataFrame(
        {"noise": [0.1, 0.2, 0.3]}, index=pd.Index([2, 7, 9], name="row_id")
    )
    write_measurement_tables({kind: frame}, tmp_path)
    evidence = (ClaimEvidence("DD-0098", (), (), (TableEvidence(kind, (7,)),)),)
    loaded = load_linked_tables(tmp_path, evidence, (kind,))[kind]
    assert loaded.index.tolist() == [7]
    assert loaded.loc[7, "noise"] == 0.2
    with pytest.raises(ValueError, match="manifest"):
        load_linked_tables(tmp_path, evidence, ())
    frame.drop(index=7).to_parquet(tmp_path / kind.value, index=True)
    with pytest.raises(ValueError, match="missing"):
        load_linked_tables(tmp_path, evidence, (kind,))


def test_table_links_filter_unavailable_values_and_change_counts(sweep, config):
    values = config.model_dump()
    values["measurement_claims"] = {
        "DD-0098": [{"table": "noise_spread.parquet", "tasks": ["mmlu"]}]
    }
    config = type(config).model_validate(values)
    table = pd.DataFrame(
        {"task": ["mmlu", "mmlu", "arc_easy"], "available": [True, False, True]}
    )
    tables = {EvidenceTable.NOISE_SPREAD: table}
    results = {r.claim_id: r for r in collect_claim_evidence(sweep, config, tables)}
    assert results["DD-0098"].has_measurements
    assert results["DD-0098"].table_links[0].row_ids == (0,)
    table.loc[0, "available"] = False
    results = {r.claim_id: r for r in collect_claim_evidence(sweep, config, tables)}
    assert not results["DD-0098"].has_measurements


def test_released_errors_ignore_precomputed_errors_and_use_source_aliases(config):
    values = config.model_dump()
    values["measurements"]["scaling_setups"] = ["3_param"]
    config = type(config).model_validate(values)
    rows = pd.DataFrame(
        [
            dict(
                task="task",
                mix=mix,
                metric="primary_metric",
                setup="3_param",
                stacked_pred=pred,
                stacked_y=observed,
                rel_error_stacked=999,
                source_file="source.csv",
                source_unit="unit",
            )
            for mix, pred, observed in [("DCLM-baseline", 3.0, 2.0), ("c4", 2.0, 0.0)]
        ]
    )
    errors, summaries = released_prediction_errors(rows, config)
    assert errors.recipe.tolist() == ["DCLM-Baseline", "C4"]
    assert errors.absolute_error.tolist() == [1.0, 2.0]
    assert errors.loc[0, "relative_error"] == 0.5
    assert pd.isna(errors.loc[1, "relative_error"])
    assert summaries.loc[0, "mean_relative_error"] == 0.5
    assert summaries.loc[0, "relative_error_count"] == 1
    assert errors.source_file.tolist() == ["source.csv"] * 2
    with pytest.raises(ValueError, match="duplicate"):
        released_prediction_errors(pd.concat([rows, rows]), config)


def test_numerical_tables_show_limits_and_all_values_without_truncation():
    kind = EvidenceTable.SCALING_SUMMARIES
    row = dict(
        task="task",
        metric="primary_metric",
        setup="3_param-helper_points-step2=0.5",
        recipe_count=25,
        mean_absolute_error=0.12,
        mean_relative_error=0.34,
        relative_error_count=25,
        decision_accuracy=0.8,
    )
    frame = pd.DataFrame([row, row]).rename_axis("row_id")
    output = StringIO()
    print_linked_table(
        TableEvidence(kind, (0, 1)), frame, Console(file=output, width=80), max_rows=1
    )
    rendered = output.getvalue()
    assert "Rows shown: 1 / 2" in rendered
    assert "--max-rows 0" in rendered
    assert "0.34" in rendered and "0.12" in rendered and "0.8" in rendered
    assert "…" not in rendered

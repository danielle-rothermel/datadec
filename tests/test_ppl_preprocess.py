from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from datadec.config import (
    CHECKPOINT_ENRICHMENT_TYPES,
    PPL_IDENTITY_COLUMNS,
    PPL_METRIC_COLUMNS,
    PPL_OUTPUT_COLUMNS,
)
from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.ppl import (
    flatten_perplexity_rows,
    group_perplexity_rows,
    preprocess_ppl,
)

WIKITEXT_RAW = "eval/wikitext_103-validation/Perplexity"
PILE_RAW = "eval/pile-validation/Perplexity"
CHECKPOINT_DTYPES = {
    field: (
        pd.StringDtype()
        if logical_type == "string"
        else np.dtype("bool" if logical_type == "bool" else logical_type)
    )
    for field, logical_type in CHECKPOINT_ENRICHMENT_TYPES
}


def _raw_record(
    *,
    params: object = "4M",
    data: object = "C4",
    seed: object = "default",
    step: object = 100,
    wikitext: object = "2.5",
) -> dict[str, object]:
    return {
        "params": params,
        "data": data,
        "seed": seed,
        "step": step,
        WIKITEXT_RAW: wikitext,
    }


def test_exact_output_schema_mapping_types_and_enum_values() -> None:
    record = _raw_record(step=1250.0)
    record[PILE_RAW] = None
    record["unknown/raw-column"] = "ignored"

    output = flatten_perplexity_rows(group_perplexity_rows(pd.DataFrame([record])))

    assert tuple(output.columns) == PPL_OUTPUT_COLUMNS
    assert output.loc[0, ["params", "data", "seed"]].tolist() == [
        ModelSizeName.M4.value,
        DataRecipeName.C4.value,
        Seed.DEFAULT.value,
    ]
    assert output.loc[0, "step"] == 1250
    assert output.loc[0, "wikitext_103_valppl"] == 2.5
    assert pd.isna(output.loc[0, "pile_valppl"])
    assert output.dtypes.to_dict() == {
        "params": pd.StringDtype(),
        "data": pd.StringDtype(),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        **CHECKPOINT_DTYPES,
        **{field: np.dtype("float64") for field in PPL_METRIC_COLUMNS},
    }


def test_shuffled_input_flattens_in_deterministic_run_and_step_order() -> None:
    records = [
        _raw_record(params="4M", data="C4", seed="default", step=200),
        _raw_record(params="10M", data="C4", seed="small aux 2", step=300),
        _raw_record(params="4M", data="C4", seed="default", step=100),
        _raw_record(params="4M", data="Dolma1.7", seed="default", step=50),
    ]
    original = flatten_perplexity_rows(group_perplexity_rows(pd.DataFrame(records)))
    shuffled = flatten_perplexity_rows(
        group_perplexity_rows(
            pd.DataFrame(records).sample(frac=1, random_state=17).reset_index(drop=True)
        )
    )

    pd.testing.assert_frame_equal(shuffled, original)
    assert list(
        original.loc[:, ["params", "data", "seed", "step"]].itertuples(
            index=False, name=None
        )
    ) == [
        ("10M", "C4", "small aux 2", 300),
        ("4M", "C4", "default", 100),
        ("4M", "C4", "default", 200),
        ("4M", "Dolma1.7", "default", 50),
    ]


@pytest.mark.parametrize(
    "step",
    [None, np.nan, np.inf, -np.inf, True, "malformed", 1.5],
    ids=["null", "nan", "positive-inf", "negative-inf", "bool", "text", "fraction"],
)
def test_invalid_steps_are_rejected_with_row_and_value_context(step: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        group_perplexity_rows(pd.DataFrame([_raw_record(step=step)]))

    message = str(exc_info.value)
    assert "invalid PPL step at row 0" in message
    assert repr(step) in message


def test_duplicate_normalized_full_key_is_rejected() -> None:
    ppl_df = pd.DataFrame(
        [
            _raw_record(step=1250),
            _raw_record(step=1250.0),
        ]
    )

    with pytest.raises(ValueError, match="duplicate PPL checkpoint at row 1") as exc:
        group_perplexity_rows(ppl_df)

    assert "params='4M', data='C4', seed='default', step=1250" in str(exc.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("params", "5M"),
        ("data", "unknown recipe"),
        ("seed", "seed 7"),
    ],
)
def test_unknown_enum_values_use_current_enum_validation(
    field: str, value: str
) -> None:
    record = _raw_record()
    record[field] = value

    with pytest.raises(ValueError, match=value):
        group_perplexity_rows(pd.DataFrame([record]))


def test_empty_input_writes_exact_typed_schema(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("ppl_raw")
    input_path.parent.mkdir(parents=True)
    pd.DataFrame().to_parquet(input_path, index=False)

    result = preprocess_ppl(paths)
    output = pd.read_parquet(paths.get_path("ppl_processed"))

    assert result.checkpoint_count == 0
    assert result.training_run_count == 0
    assert tuple(output.columns) == PPL_OUTPUT_COLUMNS
    assert output.dtypes.to_dict() == {
        "params": pd.StringDtype(),
        "data": pd.StringDtype(),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        **CHECKPOINT_DTYPES,
        **{field: np.dtype("float64") for field in PPL_METRIC_COLUMNS},
    }


def test_preprocess_projects_sorts_and_counts_without_grouping_helpers(
    tmp_path: Path,
) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("ppl_raw")
    input_path.parent.mkdir(parents=True)
    records = [
        _raw_record(params="4M", data="C4", seed="default", step=200),
        _raw_record(
            params="10M",
            data="C4",
            seed="small aux 2",
            step=300,
            wikitext=None,
        ),
        _raw_record(params="4M", data="C4", seed="default", step=100),
    ]
    records[0][PILE_RAW] = "3.5"
    records[0]["unknown/raw-column"] = "ignored"
    pd.DataFrame(records).to_parquet(input_path, index=False)

    with (
        patch(
            "datadec.data.preprocess.ppl.group_perplexity_rows",
            side_effect=AssertionError("preprocessing must use DuckDB"),
        ),
        patch(
            "datadec.data.preprocess.ppl.flatten_perplexity_rows",
            side_effect=AssertionError("preprocessing must use DuckDB"),
        ),
    ):
        result = preprocess_ppl(paths)

    output_path = paths.get_path("ppl_processed")
    output = pd.read_parquet(output_path)
    assert result.checkpoint_count == 3
    assert result.training_run_count == 2
    assert tuple(output.columns) == PPL_OUTPUT_COLUMNS
    assert output.dtypes.to_dict() == {
        "params": pd.StringDtype(),
        "data": pd.StringDtype(),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        **CHECKPOINT_DTYPES,
        **{field: np.dtype("float64") for field in PPL_METRIC_COLUMNS},
    }
    assert list(
        output.loc[:, PPL_IDENTITY_COLUMNS].itertuples(index=False, name=None)
    ) == [
        ("10M", "C4", "small aux 2", 300),
        ("4M", "C4", "default", 100),
        ("4M", "C4", "default", 200),
    ]
    assert pd.isna(output.loc[0, "wikitext_103_valppl"])
    assert output.loc[2, "pile_valppl"] == 3.5
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()


def test_preprocess_rejects_duplicate_normalized_key_with_row_context(
    tmp_path: Path,
) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("ppl_raw")
    input_path.parent.mkdir(parents=True)
    pd.DataFrame([_raw_record(step=1250), _raw_record(step=1250.0)]).to_parquet(
        input_path, index=False
    )

    with pytest.raises(ValueError, match="duplicate PPL checkpoint at row 1") as exc:
        preprocess_ppl(paths)

    assert "params='4M', data='C4', seed='default', step=1250" in str(exc.value)


def test_preprocess_rejects_unknown_enum_with_row_context(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("ppl_raw")
    input_path.parent.mkdir(parents=True)
    pd.DataFrame([_raw_record(data="unknown recipe")]).to_parquet(
        input_path, index=False
    )

    with pytest.raises(
        ValueError,
        match="unknown PPL data at row 0: 'unknown recipe'",
    ):
        preprocess_ppl(paths)


def test_typed_ingest_calls_the_shared_perplexity_grouping_helper(
    tmp_path: Path,
) -> None:
    ingest_module = importlib.import_module("datadec.data.ingest.ingest")
    ppl_df = pd.DataFrame([_raw_record()])
    dwn_df = pd.DataFrame()

    with (
        patch.object(ingest_module, "_ensure_raw_parquets_exist"),
        patch.object(
            ingest_module.pd,
            "read_parquet",
            side_effect=[ppl_df, dwn_df],
        ),
        patch.object(ingest_module, "load_model_registry", return_value={}),
        patch.object(
            ingest_module,
            "group_perplexity_rows",
            wraps=group_perplexity_rows,
        ) as shared_group,
        patch.object(ingest_module, "_group_task_rows", return_value={}),
        patch.object(ingest_module, "_build_checkpoints", return_value=[]),
    ):
        assert ingest_module.ingest_from_hf(DataDecidePaths(tmp_path)) == []

    shared_group.assert_called_once_with(ppl_df)


def test_typed_ingest_downloads_missing_raw_sources_directly(tmp_path: Path) -> None:
    ingest_module = importlib.import_module("datadec.data.ingest.ingest")
    paths = DataDecidePaths(tmp_path)

    with patch.object(ingest_module, "download_sources") as download_sources:
        ingest_module._ensure_raw_parquets_exist(paths, verbose=True)

    download_sources.assert_called_once_with(
        paths,
        ppl=True,
        olmes=True,
        verbose=True,
    )

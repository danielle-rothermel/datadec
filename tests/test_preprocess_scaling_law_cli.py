from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.scaling_law import RAW_COLUMNS

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_scaling_law.py"
SCRIPT_MODULE = "datadec_preprocess_scaling_law_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app
runner = CliRunner()


def _write_fixture(data_dir: Path) -> tuple[Path, ...]:
    raw_dir = data_dir / "raw/scaling-law"
    raw_dir.mkdir(parents=True)
    filenames = (
        "results_ladder_5xC_seeds.csv",
        "results_ladder_5xC_small_seed_extras.csv",
        "results_ladder_5xC_small_seeds_extra_real.csv",
    )
    row = {
        "group": "c4",
        "model": "4M",
        "task": "boolq",
        "chinchilla": "5xC",
        "step": "0",
        "tokens": "",
        "compute": "",
        "metrics": "{'acc_raw': 0.5}",
        "seed": "2",
    }
    paths = tuple(raw_dir / filename for filename in filenames)
    for index, path in enumerate(paths):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            if index == 0:
                writer.writerow(row)
    return paths


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = DataArtifacts(tmp_path)
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script,
            "preprocess_scaling_law",
            return_value=SimpleNamespace(
                evaluations_output_path=tmp_path / "evaluations.parquet",
                checkpoint_losses_output_path=tmp_path / "losses.parquet",
            ),
        ) as preprocess,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(paths, verbose=True)
    unit = publish.call_args.args[0]
    assert tuple(file.local_path for file in unit.files) == (
        tmp_path / "evaluations.parquet",
        tmp_path / "losses.parquet",
    )
    assert publish.call_args.kwargs == {}
    assert not hasattr(unit, "cleanup_paths")
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_data_dir_prints_stable_evidence_without_network_calls(
    tmp_path: Path,
) -> None:
    input_paths = _write_fixture(tmp_path)
    evaluations_path = tmp_path / "processed/scaling-law/evaluations.parquet"
    checkpoints_path = tmp_path / "processed/scaling-law/checkpoint-losses.parquet"

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("huggingface_hub.HfApi.upload_file") as upload_file,
        patch(
            "datadec.data.preprocess.scaling_law.perf_counter",
            side_effect=(100.0, 100.125),
        ),
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "--no-upload"])

    assert result.exit_code == 0
    assert evaluations_path.is_file()
    assert checkpoints_path.is_file()
    assert pd.read_parquet(evaluations_path).loc[0, "primary_metric"] == 0.5
    assert result.output == (
        "".join(f"scaling-law input: {path}\n" for path in input_paths)
        + f"scaling-law evaluations output: {evaluations_path}\n"
        + f"scaling-law checkpoint losses output: {checkpoints_path}\n"
        + "scaling-law input rows: 1\n"
        + "scaling-law clean rows: 1\n"
        + "scaling-law excluded policy rows: 0\n"
        + "scaling-law superseded rows: 0\n"
        + "scaling-law evaluations: 1\n"
        + "scaling-law checkpoints: 1\n"
        + "scaling-law elapsed seconds: 0.125\n"
    )
    download_sources.assert_not_called()
    upload_file.assert_not_called()


def test_cli_publishes_atomic_scaling_unit_without_cleanup_ownership(
    tmp_path: Path,
) -> None:
    paths = DataArtifacts(tmp_path)
    result_item = SimpleNamespace(
        evaluations_output_path=tmp_path / "custom-evaluations.parquet",
        checkpoint_losses_output_path=tmp_path / "custom-losses.parquet",
    )
    with (
        patch.object(script, "DataArtifacts", return_value=paths),
        patch.object(script, "preprocess_scaling_law", return_value=result_item),
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    unit = publish.call_args.args[0]
    assert tuple(file.local_path for file in unit.files) == (
        result_item.evaluations_output_path,
        result_item.checkpoint_losses_output_path,
    )
    assert not hasattr(unit, "cleanup_paths")
    assert publish.call_args.kwargs == {}

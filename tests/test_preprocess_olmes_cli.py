from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from datadec.data.model_utils import checkpoint_enrichment

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_olmes.py"
SCRIPT_MODULE = "datadec_preprocess_olmes_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app

runner = CliRunner()


def _raw_row() -> dict[str, object]:
    enrichment = checkpoint_enrichment("4M", 1250)
    return {
        "params": "4M",
        "data": "C4",
        "seed": "default",
        "step": 1250.0,
        "task": "arc_challenge",
        "chinchilla": "1x",
        "tokens": enrichment["tokens"],
        "compute": enrichment["compute"],
        "metrics": {"acc_uncond": 0.42},
    }


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script,
            "preprocess_olmes",
            return_value=SimpleNamespace(output_path=tmp_path / "olmes.parquet"),
        ) as preprocess,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(
        paths,
        input_path=None,
        output_path=None,
        verbose=True,
    )
    unit = publish.call_args.args[0]
    assert unit.files[0].local_path == tmp_path / "olmes.parquet"
    assert unit.cleanup_paths == ()
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_data_dir_reads_raw_and_writes_only_processed_olmes(tmp_path: Path) -> None:
    input_path = tmp_path / "raw/olmes.parquet"
    output_path = tmp_path / "processed/olmes.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_raw_row()]).to_parquet(input_path, index=False)

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("datadec.data.ingest.ingest.load_model_registry") as model_registry,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "--no-upload"])

    assert result.exit_code == 0
    assert output_path.is_file()
    assert not (tmp_path / "raw/ppl.parquet").exists()
    output = pd.read_parquet(output_path)
    assert output.loc[0, ["params", "data", "seed", "step", "task"]].tolist() == [
        "4M",
        "C4",
        "default",
        1250,
        "arc_challenge",
    ]
    assert result.output == (
        f"olmes input: {input_path}\n"
        f"olmes output: {output_path}\n"
        "olmes rows: 1\n"
        "olmes training runs: 1\n"
    )
    download_sources.assert_not_called()
    model_registry.assert_not_called()


def test_cli_input_and_output_overrides(tmp_path: Path) -> None:
    input_path = tmp_path / "custom-raw.parquet"
    output_path = tmp_path / "custom-processed.parquet"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_raw_row()]).to_parquet(input_path, index=False)

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(
            app,
            [
                "--data-dir",
                str(tmp_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
        )

    assert result.exit_code == 0
    assert output_path.is_file()
    assert result.output == (
        f"olmes input: {input_path}\n"
        f"olmes output: {output_path}\n"
        "olmes rows: 1\n"
        "olmes training runs: 1\n"
    )
    download_sources.assert_not_called()
    unit = publish.call_args.args[0]
    assert unit.files[0].local_path == output_path
    assert unit.cleanup_paths == ()

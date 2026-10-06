from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_ppl.py"
SCRIPT_MODULE = "datadec_preprocess_ppl_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app

runner = CliRunner()


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataArtifacts", return_value=paths) as path_type,
        patch.object(
            script,
            "preprocess_ppl",
            return_value=SimpleNamespace(output_path=tmp_path / "ppl.parquet"),
        ) as preprocess,
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(paths, verbose=True)
    unit = publish.call_args.args[0]
    assert unit.files[0].local_path == tmp_path / "ppl.parquet"
    assert unit.cleanup_paths == ()
    assert publish.call_args.kwargs == {"keep_sources": False}
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_data_dir_reads_raw_and_writes_only_processed_ppl(tmp_path: Path) -> None:
    input_path = tmp_path / "raw/ppl.parquet"
    output_path = tmp_path / "processed/ppl.parquet"
    input_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "params": "4M",
                "data": "C4",
                "seed": "default",
                "step": 1250.0,
                "eval/wikitext_103-validation/Perplexity": "2.5",
            }
        ]
    ).to_parquet(input_path, index=False)

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("datadec.data.ingest.ingest.load_model_registry") as model_registry,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "--no-upload"])

    assert result.exit_code == 0
    assert output_path.is_file()
    assert not (tmp_path / "raw/olmes.parquet").exists()
    output = pd.read_parquet(output_path)
    assert output.loc[0, ["params", "data", "seed", "step"]].tolist() == [
        "4M",
        "C4",
        "default",
        1250,
    ]
    assert result.output == (
        f"ppl input: {input_path}\n"
        f"ppl output: {output_path}\n"
        "ppl checkpoints: 1\n"
        "ppl training runs: 1\n"
    )
    download_sources.assert_not_called()
    model_registry.assert_not_called()


def test_cli_no_upload_skips_publication(tmp_path: Path) -> None:
    with (
        patch.object(
            script,
            "preprocess_ppl",
            return_value=SimpleNamespace(output_path=tmp_path / "ppl.parquet"),
        ),
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path), "--no-upload"])

    assert result.exit_code == 0
    publish.assert_not_called()


def test_cli_preprocess_failure_suppresses_publication(tmp_path: Path) -> None:
    with (
        patch.object(
            script,
            "preprocess_ppl",
            side_effect=ValueError("invalid PPL input"),
        ),
        patch.object(script, "publish_unit") as publish,
    ):
        result = runner.invoke(app, ["--data-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    publish.assert_not_called()

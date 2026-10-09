import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from repro.config import DEFAULT_CONFIG_PATH, EvaluationConfig


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/repro/evaluate_claims.py"


@pytest.fixture
def script():
    spec = importlib.util.spec_from_file_location("evaluate_claims_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_overrides_config_run_settings(script, tmp_path):
    output = tmp_path / "results"
    args = script.parse_arguments(
        [
            "--config",
            str(ROOT / DEFAULT_CONFIG_PATH),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
            "--matched-compute-tolerance",
            "0.1",
        ]
    )
    config = script.resolve_config(args)
    assert config.run.data_dir == tmp_path / "data"
    assert config.run.output_dir == output
    assert config.run.matched_compute_tolerance == 0.1
    assert config.run.claim_inventory.is_absolute()
    defaults = script.resolve_config(script.parse_arguments([]))
    assert defaults.run.data_dir == ROOT / "data"
    assert defaults.run.output_dir == ROOT / "outputs/repro/ranking"
    assert defaults.run.matched_compute_tolerance == 0.05


def test_cli_rejects_invalid_tolerance(script):
    with pytest.raises(SystemExit):
        script.parse_arguments(["--matched-compute-tolerance", "nan"])


def test_unknown_claim_id_fails_before_running(script, config):
    values = config.model_dump()
    values["run"]["claim_inventory"] = ROOT / config.run.claim_inventory
    values["claims"]["typo"] = values["claims"].pop("DD-0014")
    with pytest.raises(ValueError, match="unknown claim IDs"):
        script.validate_run_claims(EvaluationConfig.model_validate(values))


def test_main_runs_custom_toml_and_writes_effective_config(tmp_path, config):
    # A small complete experiment exercises config loading, aggregation, all
    # runner helpers, and report serialization through the actual entry point.
    source = f'''sweep_metric_group = "all"
default_related_tables = ["curves.csv"]

[run]
data_dir = "data"
output_dir = "outputs/repro/ranking"
claim_inventory = "{config.run.claim_inventory.as_posix()}"
paper_dir = "{config.run.paper_dir.as_posix()}"
matched_compute_tolerance = 0.05

[target]
size = "1B"
step = 10
metric = "primary_metric"
seeds = {json.dumps(config.target.seeds)}

[predictors]
default_seeds = {json.dumps(config.predictors.default_seeds)}
[predictors.seeds_by_size]
1B = {json.dumps(config.target.seeds)}

[aggregation]
benchmark_group = "benchmarks"
macro_task = "olmes"
subject_task = "mmlu"
subject_prefix = "mmlu_"
subject_count = 1

[task_groups]
benchmarks = ["mmlu"]
[metric_groups]
all = ["primary_metric"]
primary = ["primary_metric"]

[claims.DD-0014]
tasks = ["mmlu"]
metric_groups = ["primary"]
'''
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(source)
    from datadec.config import load_olmes_contract
    from repro.config import load_evaluation_config

    config = load_evaluation_config(config_path)
    rows = []
    for size, step, compute, seeds in (
        ("4M", 5, 1.0, config.predictors.default_seeds),
        ("1B", 10, 100.0, config.target.seeds),
    ):
        for seed in seeds:
            for index, recipe in enumerate(load_olmes_contract().recipe_map.values()):
                rows.append(
                    dict(
                        params=size,
                        step=step,
                        compute=compute,
                        total_steps=10,
                        seed=seed,
                        data=recipe,
                        task="mmlu_0",
                        **dict.fromkeys(config.metrics, float(index)),
                    )
                )
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(data_dir / "processed/olmes.parquet", index=False)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output),
            "--matched-compute-tolerance",
            "0.02",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["configuration"]["target"]["step"] == 10
    assert metadata["configuration"]["run"]["matched_compute_tolerance"] == 0.02
    assert metadata["input"]["sha256"]
    assert "Saved analysis report to" in result.stderr
    assert "scripts/repro/claims.py --run-dir" in result.stderr
    assert set(path.name for path in output.iterdir()) == {
        "rankings.parquet",
        "curves.csv",
        "proxy_comparisons.parquet",
        "recipe_crossovers.csv",
        "matched_compute.csv",
        "checkpoints.csv",
        "claim_evidence.parquet",
        "run.json",
    }
    # Reporting must use persisted measurements and config, with no OLMES read
    # or analysis rerun. Only the paper and claim inventory remain necessary.
    (data_dir / "processed/olmes.parquet").unlink()
    config_path.unlink()
    viewed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/repro/claims.py"),
            "--run-dir",
            str(output),
            "--width",
            "120",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    from repro.datasets import read_claim_evidence, read_rankings

    claim = read_claim_evidence(output / "claim_evidence.parquet")[0]
    assert (
        read_rankings(output / "rankings.parquet")[
            claim.tasks[0].best_per_seed_evidence_id
        ].statistics.decision_accuracy
        == 1.0
    )
    from repro.claims import load_claims, read_quotes

    quotes = read_quotes(
        load_claims(ROOT / config.run.claim_inventory), ROOT / config.run.paper_dir
    )
    assert "DD-0014" in viewed.stdout
    assert "With measurements: 1 | Without measurements: 63" in viewed.stdout
    assert "1.000000" in viewed.stdout
    assert "Linked ranking rows: 0" in viewed.stdout
    # Rich wraps passages for the terminal; the loaded source remains exact.
    from repro.reporting import load_claim_report

    report = load_claim_report(
        ROOT / config.run.claim_inventory, ROOT / config.run.paper_dir, output
    )
    selected = next(c for c in report.claims if c.claim_id == "DD-0014")
    assert selected.quotes[0].text == quotes["DD-0014"][0]
    assert (
        len([line for line in viewed.stdout.splitlines() if line.startswith("DD-")])
        == 64
    )
    # Automatic discovery uses the same saved report; viewing is read-only.
    latest = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/repro/claims.py"),
            "--reports-dir",
            str(tmp_path),
            "--width",
            "120",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert latest.stdout == viewed.stdout
    assert not (output / "claims.md").exists()

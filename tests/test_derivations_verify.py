from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from datadec.data.artifacts import DataArtifacts
from datadec.data.model_utils import (
    checkpoint_enrichment,
    create_model_schedules,
)
from datadec.data.selection import DatasetSelection
from datadec.data.verify.derivations import verify_preprocessed_derivations

SELECTION = DatasetSelection(
    ppl=True,
    olmes=True,
    olmes_details=("fixture",),
    scaling_law=True,
    published_results=(),
    published_figures=False,
    all_data=False,
)


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_scaling_raw(
    paths: DataArtifacts,
    *,
    compute: float,
    tokens: int,
) -> None:
    for path in paths.scaling_law_raw_paths():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("model", "step", "tokens", "compute"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "model": "1B",
                    "step": 1,
                    "tokens": tokens,
                    "compute": compute,
                }
            )


def _verification_paths(
    tmp_path: Path,
    *,
    raw_scaling_uses_nominal_compute: bool,
) -> DataArtifacts:
    paths = DataArtifacts(tmp_path)
    schedule = next(
        schedule for schedule in create_model_schedules() if schedule.params == "1B"
    )
    tokens = schedule.tokens_at_step(1)
    exact_compute = schedule.compute_at_step(1)
    nominal_compute = float(
        tokens
        * schedule.nominal_parameter_count
        * schedule.flops_per_token_per_parameter
    )
    ppl_row = {
        "params": "1B",
        "step": 1,
        **checkpoint_enrichment("1B", 1),
    }
    schedule_row = {
        "params": "1B",
        "step": 1,
        **checkpoint_enrichment("1B", 1),
    }
    _write_parquet(paths.get_path("ppl_processed"), [ppl_row])
    _write_parquet(paths.get_path("olmes_processed"), [schedule_row])
    _write_parquet(paths.get_path("dwn_raw"), [schedule_row])
    _write_parquet(paths.scaling_law_evaluations_path(), [schedule_row])
    _write_parquet(paths.scaling_law_checkpoint_losses_path(), [schedule_row])
    _write_scaling_raw(
        paths,
        compute=(
            nominal_compute if raw_scaling_uses_nominal_compute else exact_compute
        ),
        tokens=tokens,
    )
    _write_parquet(
        paths.olmes_details_tasks_path("fixture"),
        [
            {
                "params": "1B",
                "step": 1,
                **checkpoint_enrichment("1B", 1),
                "model_config": json.dumps(
                    {
                        "max_length": 2048,
                        "model": "fixture-1B-5xC-2",
                        "revision": "step1-unsharded-hf",
                    }
                ),
            }
        ],
    )
    return paths


def test_verification_accepts_exact_schedule_evidence(tmp_path: Path) -> None:
    paths = _verification_paths(
        tmp_path,
        raw_scaling_uses_nominal_compute=False,
    )

    result = verify_preprocessed_derivations(paths, SELECTION)

    assert result.contradiction_count == 0
    assert result.raw_scaling_law is not None
    assert result.raw_scaling_law.token_evidence_count == 3
    assert result.raw_scaling_law.compute_evidence_count == 3
    assert result.lr_raw_evidence_count == 0


def test_verification_identifies_nominal_raw_compute_semantics(
    tmp_path: Path,
) -> None:
    paths = _verification_paths(
        tmp_path,
        raw_scaling_uses_nominal_compute=True,
    )

    result = verify_preprocessed_derivations(paths, SELECTION)

    assert result.contradiction_count == 3
    assert result.raw_scaling_law is not None
    assert result.raw_scaling_law.exact_compute_mismatch_count == 3
    assert result.raw_scaling_law.nominal_compute_mismatch_count == 0


def test_verification_rejects_null_required_checkpoint_derivations(
    tmp_path: Path,
) -> None:
    paths = _verification_paths(
        tmp_path,
        raw_scaling_uses_nominal_compute=False,
    )
    ppl = pd.read_parquet(paths.get_path("ppl_processed"))
    ppl.loc[0, "tokens"] = None
    ppl.loc[0, "compute"] = None
    ppl.loc[0, "lr_max"] = None
    ppl.loc[0, "lr_at_step"] = None
    ppl.to_parquet(paths.get_path("ppl_processed"), index=False)

    result = verify_preprocessed_derivations(paths, SELECTION)
    verification = result.processed_outputs[0]

    assert verification.token_evidence_count == 0
    assert verification.token_mismatch_count == 1
    assert verification.compute_evidence_count == 0
    assert verification.exact_compute_mismatch_count == 1
    assert verification.model_detail_mismatch_count == 1
    assert verification.lr_mismatch_count == 1


def test_base_selection_does_not_require_or_discover_detail_outputs(
    tmp_path: Path,
) -> None:
    paths = _verification_paths(
        tmp_path,
        raw_scaling_uses_nominal_compute=False,
    )
    unrelated_detail = paths.olmes_details_tasks_path("unrelated")
    unrelated_detail.parent.mkdir(parents=True)
    unrelated_detail.write_text(
        "not a parquet file",
        encoding="utf-8",
    )
    selection = DatasetSelection(
        ppl=True,
        olmes=True,
        olmes_details=(),
        scaling_law=True,
        published_results=(),
        published_figures=False,
        all_data=False,
    )

    result = verify_preprocessed_derivations(paths, selection)

    assert result.contradiction_count == 0
    assert len(result.processed_outputs) == 4
    assert result.detail_tasks == ()

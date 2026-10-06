from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import srsly
from dr_ds import coerce_int

from datadec.data.download import download_sources
from datadec.data.ingest.checkpoint import EvalCheckpoint
from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed, Task
from datadec.data.ingest.metrics import PerplexityMetrics, TaskEvalMetrics
from datadec.data.ingest.registries.model_details import (
    ModelRegistry,
    load_model_registry,
)
from datadec.data.ingest.run import TrainingRun
from datadec.data.artifacts import DataArtifacts
from datadec.data.preprocess.ppl import group_perplexity_rows

type RunKey = tuple[ModelSizeName, DataRecipeName, Seed]
type TaskRowsByKey = dict[RunKey, dict[int, dict[Task, TaskEvalMetrics]]]


def ingest_from_hf(
    paths: DataArtifacts | None = None,
    *,
    verbose: bool = False,
) -> list[TrainingRun]:
    paths = paths or DataArtifacts()
    _ensure_raw_parquets_exist(paths, verbose=verbose)

    ppl_df = pd.read_parquet(paths.get_path("ppl_raw"))
    dwn_df = pd.read_parquet(paths.get_path("dwn_raw"))

    model_registry = load_model_registry(dwn_df)

    if verbose:
        print(f">> ppl rows: {len(ppl_df)}")
        print(f">> dwn rows: {len(dwn_df)}")

    ppl_by_key = group_perplexity_rows(ppl_df)
    task_by_key = _group_task_rows(dwn_df)

    run_keys = sorted(set(ppl_by_key.keys()) | set(task_by_key.keys()))
    runs: list[TrainingRun] = []
    for run_key in run_keys:
        params, data, seed = run_key
        checkpoints = _build_checkpoints(
            ppl_rows=ppl_by_key.get(run_key, {}),
            task_rows=task_by_key.get(run_key, {}),
        )
        if len(checkpoints) == 0:
            continue
        runs.append(
            TrainingRun(
                params=params,
                data=data,
                seed=seed,
                model_details=model_registry[params],
                checkpoints=checkpoints,
            )
        )
    if verbose:
        print(f">> built {len(runs)} TrainingRuns")
    return runs


def _ensure_raw_parquets_exist(paths: DataArtifacts, *, verbose: bool) -> None:
    missing_types = [
        raw_type
        for raw_type in ("ppl", "dwn")
        if not paths.get_path(f"{raw_type}_raw").exists()
    ]
    if len(missing_types) == 0:
        return
    if verbose:
        print(f">> missing raw parquets for {missing_types}; downloading")
    download_sources(paths, ppl=True, olmes=True, verbose=verbose)


def _group_task_rows(dwn_df: pd.DataFrame) -> TaskRowsByKey:
    grouped: TaskRowsByKey = defaultdict(lambda: defaultdict(dict))
    records = dwn_df.to_dict(orient="records")
    for record in records:
        run_key = _run_key_from_record(record)
        step = coerce_int(record["step"])
        if step is None:
            continue
        task = Task(record["task"])
        metrics_payload = _parse_metrics_payload(record["metrics"])
        grouped[run_key][step][task] = TaskEvalMetrics.model_validate(metrics_payload)
    return grouped


def _run_key_from_record(record: dict[str, Any]) -> RunKey:
    return (
        ModelSizeName(record["params"]),
        DataRecipeName(record["data"]),
        Seed(record["seed"]),
    )


def _parse_metrics_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        return orjson.loads(raw)
    text = str(raw).replace("'", '"')
    return orjson.loads(text)


def _build_checkpoints(
    *,
    ppl_rows: dict[int, PerplexityMetrics],
    task_rows: dict[int, dict[Task, TaskEvalMetrics]],
) -> list[EvalCheckpoint]:
    steps = sorted(set(ppl_rows.keys()) | set(task_rows.keys()))
    checkpoints: list[EvalCheckpoint] = []
    for step in steps:
        perplexity = ppl_rows.get(step)
        task_evals = task_rows.get(step, {})
        if perplexity is None and len(task_evals) == 0:
            continue
        checkpoints.append(
            EvalCheckpoint(
                step=step,
                perplexity=perplexity,
                task_evals=task_evals,
            )
        )
    return checkpoints


DEFAULT_CACHE_FILENAME: str = "typed_runs.jsonl"

_CHECKPOINT_COMPUTED_FIELDS: set[str] = {
    "tokens",
    "compute",
    "lr_at_step",
    "cumulative_lr",
    "mmlu_average",
}


def cache_path(paths: DataArtifacts | None = None) -> Path:
    paths = paths or DataArtifacts()
    return paths.data_dir / DEFAULT_CACHE_FILENAME


def cache_to_jsonl(runs: list[TrainingRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exclude_spec: Any = {
        "model_details": True,
        "checkpoints": {"__all__": _CHECKPOINT_COMPUTED_FIELDS},
    }
    payloads = [run.model_dump(mode="json", exclude=exclude_spec) for run in runs]
    srsly.write_jsonl(path, payloads)


def load_from_cache(
    path: Path,
    *,
    model_registry: ModelRegistry,
) -> list[TrainingRun]:
    runs: list[TrainingRun] = []
    for payload in srsly.read_jsonl(path):
        params = ModelSizeName(payload["params"])
        payload["model_details"] = model_registry[params].model_dump()
        runs.append(TrainingRun.model_validate(payload))
    return runs

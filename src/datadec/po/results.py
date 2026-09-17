"""Load a finished sweep into flat tables for analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datadec.po.formats import AXES
from datadec.po.metrics import GENERATION_METRICS, likelihood_metrics, primary_likelihood_metric
from datadec.po.sweep import baselines_dir

ITEM_METRICS = (
    "acc_raw", "acc_per_token", "acc_per_char", "acc_per_byte", "acc_uncond", "correct_choice",
    "predicted_index_raw", "predicted_index_per_token", "predicted_index_per_char",
    "predicted_index_per_byte", "predicted_index_uncond", "sum_logits_corr",
    "logits_per_token_corr", "logits_per_char_corr", "bits_per_byte_corr",
)
CHOICE_FIELDS = ("sum_logits", "sum_logits_uncond", "num_tokens", "num_tokens_all", "num_chars", "is_greedy")


def _po_columns(task_config: dict, model_config: dict) -> dict:
    po = task_config.get("metadata", {}).get("po", {})
    primary = task_config["primary_metric"]
    cols = {
        "sweep": po.get("sweep"), "formulation": po.get("formulation"),
        "format_id": po.get("format_id"), "instruction_id": po.get("instruction_id"),
        "subset_seed": po.get("subset_seed"), "subset_n": po.get("subset_n"),
        "model": model_config.get("model"), "revision": model_config.get("revision"),
        "num_shots": task_config.get("num_shots"),
        "primary_metric": primary, "primary_likelihood_metric": primary_likelihood_metric(primary),
    }
    for axis in AXES:
        cols[f"fmt_{axis}"] = (po.get("format") or {}).get(axis)
    return cols


def _metrics_paths(sweep_dir: Path) -> list[Path]:
    """Task metrics files of a sweep plus its moved-out baselines, one per (task_hash, model_hash)."""
    seen: set[tuple[str, str]] = set()
    paths = []
    for root in (baselines_dir(sweep_dir), sweep_dir):
        for metrics_path in sorted(root.rglob("task-*-metrics.json")) if root.exists() else []:
            m = json.loads(metrics_path.read_text())
            key = (m["task_hash"], m["model_hash"])
            if key in seen:
                continue
            seen.add(key)
            paths.append(metrics_path)
    return paths


def load_sweep(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    """Flatten a sweep (and the baselines moved out of it) into task, item, and choice tables."""
    sweep_dir = Path(sweep_dir)
    task_rows, item_rows, choice_rows = [], [], []
    for metrics_path in _metrics_paths(sweep_dir):
        m = json.loads(metrics_path.read_text())
        base = _po_columns(m["task_config"], m["model_config"]) | {
            "task_idx": m["task_idx"], "task_hash": m["task_hash"], "model_hash": m["model_hash"],
            "run_dir": str(metrics_path.parent),
        }
        task_rows.append(base | {"num_instances": m["num_instances"], "processing_time": m["processing_time"]}
                         | {k: v for k, v in m["metrics"].items() if isinstance(v, (int, float))}
                         | {"primary": m["metrics"]["primary_score"]})
        pred_path = metrics_path.with_name(metrics_path.name.replace("-metrics.json", "-predictions.jsonl"))
        for line in pred_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            key = {"task_idx": m["task_idx"], "native_id": p["native_id"], "doc_id": p["doc_id"], "label": p.get("label")}
            outputs = p.get("model_output", [])
            if base["primary_likelihood_metric"] is None:  # generation formulation: one generation, no choice scores
                gen = {k: p["metrics"].get(k) for k in GENERATION_METRICS}
                gen["continuation"] = outputs[0].get("continuation") if outputs else None
                item_rows.append(base | key | {k: None for k in ITEM_METRICS} | likelihood_metrics([], None) | gen
                                 | {"primary": p["metrics"].get(base["primary_metric"]), "primary_likelihood": None})
                continue
            lik = likelihood_metrics(outputs, p.get("label"))
            item_rows.append(base | key | {k: p["metrics"].get(k) for k in ITEM_METRICS} | lik
                             | {"primary": p["metrics"].get(base["primary_metric"]),
                                "primary_likelihood": lik[base["primary_likelihood_metric"]]})
            for ci, out in enumerate(outputs):
                choice_rows.append(base | key | {"choice_index": ci} | {k: out.get(k) for k in CHOICE_FIELDS})
    return {
        "tasks": pd.DataFrame(task_rows),
        "items": pd.DataFrame(item_rows),
        "choices": pd.DataFrame(choice_rows),
    }


def write_analysis(sweep_dir: Path) -> Path:
    tables = load_sweep(sweep_dir)
    out = Path(sweep_dir) / "analysis"
    out.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return out

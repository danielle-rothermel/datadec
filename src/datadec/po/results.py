"""Load a finished sweep into flat tables for analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from datadec.po.formats import AXES

ITEM_METRICS = (
    "acc_raw", "acc_per_token", "acc_per_char", "acc_per_byte", "acc_uncond", "correct_choice",
    "predicted_index_raw", "predicted_index_per_token", "predicted_index_per_char",
    "predicted_index_per_byte", "predicted_index_uncond", "sum_logits_corr",
    "logits_per_token_corr", "logits_per_char_corr", "bits_per_byte_corr",
)
CHOICE_FIELDS = ("sum_logits", "sum_logits_uncond", "num_tokens", "num_tokens_all", "num_chars", "is_greedy")


def likelihood_metrics(model_output: list[dict], gold: int) -> dict:
    """DataDecide-style continuous metrics from per-choice summed log-likelihoods.

    correct_prob is the softmax over choices of sum_logits at the gold index (chance 1/k);
    margin is correct_prob minus the best wrong choice's probability.
    """
    ll = [o["sum_logits"] for o in model_output]
    if not ll or gold is None:
        return {"n_choices": len(ll), "correct_prob": None, "margin": None, "correct_logprob": None}
    m = max(ll)
    weights = [math.exp(x - m) for x in ll]
    z = sum(weights)
    probs = [w / z for w in weights]
    others = [p for i, p in enumerate(probs) if i != gold]
    return {
        "n_choices": len(ll),
        "correct_prob": probs[gold],
        "margin": probs[gold] - (max(others) if others else 0.0),
        "correct_logprob": ll[gold],
    }


def _po_columns(task_config: dict, model_config: dict) -> dict:
    po = task_config.get("metadata", {}).get("po", {})
    cols = {
        "sweep": po.get("sweep"), "formulation": po.get("formulation"),
        "format_id": po.get("format_id"), "instruction_id": po.get("instruction_id"),
        "subset_seed": po.get("subset_seed"), "subset_n": po.get("subset_n"),
        "model": model_config.get("model"), "revision": model_config.get("revision"),
        "num_shots": task_config.get("num_shots"),
    }
    for axis in AXES:
        cols[f"fmt_{axis}"] = (po.get("format") or {}).get(axis)
    return cols


def load_sweep(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    sweep_dir = Path(sweep_dir)
    task_rows, item_rows, choice_rows = [], [], []
    for metrics_path in sorted(sweep_dir.rglob("task-*-metrics.json")):
        m = json.loads(metrics_path.read_text())
        base = _po_columns(m["task_config"], m["model_config"]) | {
            "task_idx": m["task_idx"], "task_hash": m["task_hash"], "model_hash": m["model_hash"],
            "run_dir": str(metrics_path.parent),
        }
        task_rows.append(base | {"num_instances": m["num_instances"], "processing_time": m["processing_time"]}
                         | {k: v for k, v in m["metrics"].items() if isinstance(v, (int, float))})
        pred_path = metrics_path.with_name(metrics_path.name.replace("-metrics.json", "-predictions.jsonl"))
        for line in pred_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            key = {"task_idx": m["task_idx"], "native_id": p["native_id"], "doc_id": p["doc_id"], "label": p.get("label")}
            item_rows.append(base | key | {k: p["metrics"].get(k) for k in ITEM_METRICS}
                             | likelihood_metrics(p.get("model_output", []), p.get("label")))
            for ci, out in enumerate(p.get("model_output", [])):
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

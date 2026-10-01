"""Seed selection and GEPA job writing: best/worst/empty groups, per-model batch sizes, dtype."""

import json

import pandas as pd

from datadec.po.formats import CANONICAL, format_id
from datadec.po.gepa_jobs import add_empty_seed, rank_seeds, select_seeds, write_jobs
from datadec.po.subsets import ItemSubset

FID = format_id(CANONICAL)


def _items():
    rows = []
    for instr, share in (("a", 0.30), ("b", 0.20), ("c", 0.25), ("none", 0.22)):
        for i in range(3):
            rows.append({"model": "m", "revision": None, "formulation": "rc", "format_id": FID, "instruction_id": instr,
                         "native_id": f"x{i}", "primary": 1.0 if share > 0.24 else 0.0, "primary_likelihood": share + 0.001 * i})
    return pd.DataFrame(rows)


def test_rank_select_and_empty_seed():
    ranked = rank_seeds(_items())
    assert list(ranked.sort_values("rank")["instruction_id"]) == ["a", "c", "none", "b"]
    sel = select_seeds(ranked, 1)
    assert set(sel["group"]) == {"best", "worst"} and set(sel["instruction_id"]) == {"a", "b"}
    sel = add_empty_seed(sel, ranked)
    empty = sel[sel["group"] == "empty"]
    assert len(empty) == 1 and empty["instruction_id"].iloc[0] == "none" and abs(empty["score"].iloc[0] - 0.221) < 1e-9


def test_write_jobs_records_dtype_and_per_model_batch(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "sweep.json").write_text(json.dumps({"formats": [dict(CANONICAL)], "instructions": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}], "task": "arc_easy"}))
    ranked = rank_seeds(_items())
    sel = add_empty_seed(select_seeds(ranked, 1), ranked)
    train = ItemSubset(dataset_path="allenai/ai2_arc", dataset_name="ARC-Easy", split="test", seed=0, ids=("x0",))
    val = ItemSubset(dataset_path="allenai/ai2_arc", dataset_name="ARC-Easy", split="test", seed=0, ids=("y0",))
    paths = write_jobs(name="t", sweep_dir=sweep, selected=sel, train_subset=train, val_subset=val, max_metric_calls=10,
                       max_proposals=10, device="cuda", batch_size=4, batch_sizes={"m": 64}, dtype="float32", root=tmp_path / "gepa")
    jobs = [json.loads(p.read_text()) for p in paths]
    assert {j["group"] for j in jobs} == {"best", "worst", "empty"}
    assert all(j["device"] == "cuda" and j["dtype"] == "float32" and j["batch_size"] == 64 and j["max_proposals"] == 10 for j in jobs)
    assert [j["seed_instruction"]["text"] for j in jobs if j["group"] == "empty"] == [""]

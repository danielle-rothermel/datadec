"""Tidy tables: role mapping, sweep selection, pooling, contrast generation (synthetic two-sweep fixture), figure specs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from datadec.po.figures import FigureSpec, load_spec, render
from datadec.po.model_registry import describe
from datadec.po.tidy import (
    Role,
    bootstrap_ci,
    build_tables,
    natural_contrasts,
    role_for_subset,
    select_sweeps,
    write_tables,
    read_tables,
)

SPECS = Path(__file__).resolve().parents[2] / "configs" / "po" / "figures"
DCLM = ("allenai/DataDecide-dclm-baseline-150M", "step37500-seed-default")
DOLMA = ("allenai/DataDecide-dolma1_7-150M", "step37500-seed-default")
TRAIN, VAL, TEST1 = ("arc_easy-train-part-train-n3-seed0", "arc_easy-validation-n2-seed0",
                     "arc_easy-test-part-test1-n2-seed0")


@pytest.mark.parametrize(("label", "role"), [
    ("arc_easy-train-part-train-n1495-seed0", Role.TRAIN),
    ("arc_easy-train-part-traindev-n748-seed0", Role.TRAIN_DEV),
    ("arc_easy-validation-n570-seed0", Role.TEST),
    ("arc_easy-test-part-test1-n1188-seed0", Role.TEST),
    ("csqa-validation-part-test-n610-seed0", Role.TEST),
    ("csqa-train-part-testaug-n611-seed0", Role.TEST),
    ("hellaswag-validation-part-test-n2000-seed0", Role.TEST),
])
def test_role_mapping(label: str, role: Role) -> None:
    assert role_for_subset(label) == role


def test_unknown_subset_label_raises() -> None:
    with pytest.raises(ValueError):
        role_for_subset("arc_easy-test-n100-seed0")


# ---------------------------------------------------------------- synthetic sweeps


def _write_sweep(root: Path, name: str, model: tuple[str, str], subsets: dict[str, list[tuple[str, int, int]]],
                 *, num_tasks: int | None = None) -> None:
    """subsets: label -> [(native_id, correct, n_choices)]; correct items put more mass on the gold choice."""
    d = root / name
    run = d / "olmes" / "m" / "r" / "20261001T000000Z-tasks-chunk000"
    run.mkdir(parents=True)
    (d / "sweep.json").write_text(json.dumps({"model": model[0], "revision": model[1], "dtype": "float32",
                                              "quant": None, "task": "arc_easy",
                                              "num_tasks": num_tasks or len(subsets)}))
    for idx, (label, items) in enumerate(subsets.items()):
        po = {"sweep": name, "task": "arc_easy", "subset": label, "formulation": "rc", "format_id": "canon",
              "instruction_id": "none", "num_shots": 5}
        metrics = {"task_idx": idx, "task_hash": f"{name}-{idx}", "model_hash": name, "num_instances": len(items),
                   "processing_time": 1.0, "metrics": {"primary_score": 0.0},
                   "task_config": {"primary_metric": "acc_per_char", "num_shots": 5, "metadata": {"po": po}},
                   "model_config": {"model": model[0], "revision": model[1]}}
        (run / f"task-{idx:03d}-arc_easy-metrics.json").write_text(json.dumps(metrics))
        lines = []
        for doc_id, (nid, correct, n_choices) in enumerate(items):
            outputs = [{"sum_logits": -1.0 if (i == 0) == bool(correct) else -2.0, "num_chars": 1, "num_tokens": 1,
                        "sum_logits_uncond": -3.0, "num_tokens_all": 1, "is_greedy": False}
                       for i in range(n_choices)]
            lines.append(json.dumps({"native_id": nid, "doc_id": doc_id, "label": 0, "model_output": outputs,
                                     "metrics": {"acc_per_char": correct}}))
        (run / f"task-{idx:03d}-arc_easy-predictions.jsonl").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("sweeps")
    _write_sweep(root, "baselines-dd150m", DCLM, {
        TRAIN: [("t1", 1, 4), ("t2", 0, 4), ("t3", 1, 4)],
        VAL: [("v1", 1, 4), ("v2", 1, 4)],
        TEST1: [("x1", 0, 4), ("x2", 1, 2)],
    })
    _write_sweep(root, "baselines-dd-dolma-150m", DOLMA, {
        TRAIN: [("t1", 0, 4), ("t2", 0, 4), ("t3", 1, 4)],
        VAL: [("v1", 0, 4), ("v2", 1, 4)],
        TEST1: [("x1", 0, 4)],  # x2 missing: the sweep is one task short
    }, num_tasks=4)
    _write_sweep(root, "g5-granite4-micro-base", DCLM, {TRAIN: [("t1", 1, 4)]})
    _write_sweep(root, "quant-smoke-dd300m-bf16", DCLM, {TRAIN: [("t1", 1, 4)]})
    _write_sweep(root, "canon-dd150m", DCLM, {TRAIN: [("t1", 1, 4)]})
    return build_tables(root, workers=1)


def test_selection_excludes_by_prefix(built) -> None:
    sel = built.selection
    assert [p.name for p in sel.included] == ["baselines-dd-dolma-150m", "baselines-dd150m"]
    assert sel.excluded == {"canon-": 1, "g5-": 1, "quant-smoke": 1}


def test_selection_strips_task_prefix(tmp_path: Path) -> None:
    for name in ("t-csqa-g5-granite4-micro-base", "t-piqa-retry-mid-dd-dclm-1b", "t-csqa-quant-smoke-x"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sweep.json").write_text("{}")
    sel = select_sweeps(tmp_path)
    assert [p.name for p in sel.included] == ["t-piqa-retry-mid-dd-dclm-1b"]
    assert sel.excluded == {"g5-": 1, "quant-smoke": 1}


def test_tasks_table(built) -> None:
    tasks = built.tables["tasks"]
    assert len(tasks) == 6
    dclm = tasks[tasks["sweep"] == "baselines-dd150m"].set_index("subset")
    assert dclm.loc[TRAIN, "role"] == "train" and dclm.loc[TEST1, "role"] == "test"
    assert dclm.loc[TRAIN, "accuracy"] == pytest.approx(2 / 3)
    assert dclm.loc[TEST1, "chance_accuracy"] == pytest.approx((1 / 4 + 1 / 2) / 2)
    assert dclm["complete"].all()
    assert not tasks[tasks["sweep"] == "baselines-dd-dolma-150m"]["complete"].any()


def test_summary_pools_test_subsets(built) -> None:
    s = built.tables["summary"].set_index(["model_key", "role"])
    key = describe(*DCLM, "float32", None).model_key
    test = s.loc[(key, "test")]
    assert test["n_items"] == 4  # validation (2) + test1 (2) pooled
    assert test["accuracy"] == pytest.approx(3 / 4)
    assert test["chance_accuracy"] == pytest.approx((3 * 0.25 + 0.5) / 4)
    assert test["margin_accuracy"] == pytest.approx(test["accuracy"] - test["chance_accuracy"])
    assert test["accuracy_lo"] <= test["accuracy"] <= test["accuracy_hi"]
    assert test["likelihood_lo"] <= test["likelihood"] <= test["likelihood_hi"]
    assert test["n_sweeps"] == 1
    assert set(built.tables["summary"]["instruction_id"]) == {"none"}


def test_recipe_contrast_pairs_items(built) -> None:
    c = built.tables["contrasts"]
    assert set(c["contrast_type"]) == {"recipe"}  # same size and seed: no size, seed or checkpoint contrasts
    a, b = describe(*DCLM, "float32", None).model_key, describe(*DOLMA, "float32", None).model_key
    row = c[(c["role"] == "test") & (c["metric"] == "accuracy")].iloc[0]
    assert (row["a_key"], row["b_key"]) == (a, b)
    assert row["n_items"] == 3  # v1, v2, x1 paired; x2 missing for dolma
    assert row["delta"] == pytest.approx(((0 - 1) + (1 - 1) + (0 - 0)) / 3)
    assert row["lo"] <= row["delta"] <= row["hi"]
    assert row["n_runs"] == 1 and row["method"] == "paired-item-bootstrap"
    assert "gepa" not in set(c["contrast_type"])


def test_models_table(built) -> None:
    m = built.tables["models"]
    assert len(m) == 2 and set(m["recipe"]) == {"dclm", "dolma"}


def test_bootstrap_is_reproducible() -> None:
    x = np.random.default_rng(3).random((50, 2))
    lo1, hi1 = bootstrap_ci(x)
    lo2, hi2 = bootstrap_ci(x[:, :1])
    assert lo1[0] == lo2[0] and hi1[0] == hi2[0]
    assert (lo1 <= x.mean(0)).all() and (x.mean(0) <= hi1).all()


def test_natural_contrasts() -> None:
    idents = [
        ("allenai/DataDecide-dclm-baseline-150M", "step37500-seed-default", "float32", None),
        ("allenai/DataDecide-dclm-baseline-300M", "step45000-seed-default", "float32", None),
        ("allenai/DataDecide-dolma1_7-150M", "step37500-seed-default", "float32", None),
        ("allenai/DataDecide-dclm-baseline-150M", "step37500-seed-small-aux-2", "float32", None),
        ("allenai/DataDecide-dclm-baseline-150M", "step18750-seed-default", "float32", None),
        ("Qwen/Qwen3-1.7B-Base", None, "float32", None),
        ("Qwen/Qwen3-1.7B", None, "float32", None),
        ("allenai/Olmo-3-1025-7B", None, "float32", None),
        ("allenai/Olmo-3-1025-7B", None, "bfloat16", "nf4"),
        ("allenai/Olmo-3-7B-Instruct-SFT", None, "float32", None),
        ("allenai/Olmo-3-7B-Instruct-DPO", None, "float32", None),
        ("allenai/Olmo-3-7B-Instruct", None, "float32", None),
    ]
    models = pd.DataFrame([describe(*i).row() for i in idents])
    k = {i: describe(*i).model_key for i in idents}
    got = set(natural_contrasts(models))
    dd150, dd300, dolma, aux, mid, qb, qi, o3, o3nf4, sft, dpo, inst = (k[i] for i in idents)
    assert got == {
        ("size-adjacent", dd150, dd300),
        ("recipe", dd150, dolma),
        ("seed", dd150, aux),
        ("checkpoint", dd150, mid),
        ("base-vs-instruct", qb, qi),
        ("base-vs-instruct", o3, inst),
        ("stage", o3, sft), ("stage", sft, dpo), ("stage", dpo, inst),
        ("precision", o3, o3nf4),
    }


# ---------------------------------------------------------------- figure specs


@pytest.mark.parametrize("path", sorted(SPECS.glob("*.json")), ids=lambda p: p.stem)
def test_shipped_specs_validate(path: Path) -> None:
    spec = load_spec(path)
    assert all(p.role == "test" for p in spec.panels)
    assert all(set(p.metrics) in ({"accuracy", "likelihood"}, {"margin_accuracy", "margin_likelihood"})
               for p in spec.panels)


def _spec(**over) -> dict:
    panel = {"title": "p", "task": "arc_easy", "role": "test", "x": {"field": "params", "log": True},
             "metrics": ["accuracy", "likelihood"],
             "series": [{"label": "DCLM", "filter": {"recipe": "dclm"}, "color": "dclm", "band": None, "marker": "o"}],
             "chance_line": True, "arrows": {"contrast_type": "gepa"}}
    spec = {"title": "t", "rows": 1, "cols": 1, "metric_mode": "value", "panels": [panel]}
    for key, val in over.items():
        if key.startswith("panel_"):
            panel[key.removeprefix("panel_")] = val
        else:
            spec[key] = val
    return spec


@pytest.mark.parametrize("over", [
    {"panel_series": [{"label": "x", "filter": {}, "color": "#123456"}]},  # colours come from the palette
    {"panel_metrics": ["accuracy", "perplexity"]},
    {"panel_tasks": ["arc_easy"]},  # both task and tasks
    {"rows": 1, "cols": 1, "panels": []} | {"panels": [_spec()["panels"][0]] * 2},
    {"metric_mode": "delta"},  # delta series need a contrast_type filter (and no arrows)
    {"panel_extra": 1},
])
def test_invalid_specs_raise(over: dict) -> None:
    with pytest.raises(ValidationError):
        FigureSpec.model_validate(_spec(**over))


def test_render_writes_png_and_csv_and_arrows_only_with_gepa_rows(built, tmp_path: Path) -> None:
    write_tables(built, tmp_path / "tidy")
    tables = read_tables(tmp_path / "tidy")
    spec = FigureSpec.model_validate(_spec())
    png, csv = render(spec, tables, tmp_path / "fig.png")
    assert png.exists() and csv.exists()
    plotted = pd.read_csv(csv)
    assert len(plotted) == 2 and "arrow_delta" not in plotted  # one point per metric, no gepa rows
    key = describe(*DCLM, "float32", None).model_key
    gepa = pd.DataFrame([{"contrast_type": "gepa", "a_key": key, "b_key": f"gepa:cell:{s}", "task": "arc_easy",
                          "role": "test", "metric": m, "delta": 0.1, "lo": 0.0, "hi": 0.2, "n_items": 4, "n_runs": 6,
                          "method": "hierarchical-bootstrap"} for m in ("accuracy", "likelihood") for s in ("both", "ape")])
    tables["contrasts"] = pd.concat([tables["contrasts"], gepa], ignore_index=True)
    _, csv = render(spec, tables, tmp_path / "fig-gepa.png")
    plotted = pd.read_csv(csv)
    assert plotted["arrow_delta"].tolist() == [0.1, 0.1]
    assert set(plotted["arrow_key"]) == {"gepa:cell:both"}

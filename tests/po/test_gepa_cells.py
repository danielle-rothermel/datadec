"""GEPA cells: job and pool-item contracts, the eval follow-up, aggregation, and the APE prompt cap."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datadec.po.ape import (
    PromptCap,
    build_seed_meta_prompt,
    enforce_cap,
    olmes_demos,
    render_demos,
    truncate_to_cap,
    within_cap,
)
from datadec.po.formats import CANONICAL
from datadec.po.gepa_cells import (
    aggregate,
    cell_id,
    default_dtype,
    eval_class,
    gepa_class,
    gepa_est_minutes,
    hierarchical_bootstrap,
    model_key,
    write_cell,
    write_eval_item,
)
from datadec.po.subsets import OLMES_REPO

REPO = Path(__file__).resolve().parents[2]
DD = "allenai/DataDecide-dclm-baseline-150M"
DD_REV = "step37500-seed-default"
CELL = "allenai-DataDecide-dclm-baseline-150M_step37500-seed-default_fp32-tf32__arc_easy"
TASKS = ("arc_easy", "arc_challenge", "openbookqa", "csqa", "socialiqa", "hellaswag", "piqa", "winogrande")


def _write(tmp_path, model=DD, revision=DD_REV, task="arc_easy", **kw):
    root, pool = tmp_path / "root", tmp_path / "root" / "pool" / "p"
    paths = write_cell(model=model, revision=revision, task=task, pool_dir=pool, root=root, repo=REPO,
                       olmes_repo=Path("/x/olmes"), **kw)
    return root, pool, paths


def test_ids_and_classes():
    key = model_key(DD, DD_REV, "float32")
    assert key == "allenai/DataDecide-dclm-baseline-150M@step37500-seed-default@fp32-tf32"
    assert cell_id(key, "arc_easy") == CELL
    assert model_key("Qwen/Qwen3-8B", None, "bfloat16") == "Qwen/Qwen3-8B@main@bf16"
    assert [gepa_class(m) for m in (DD, "Qwen/Qwen3-4B", "allenai/Olmo-3-1025-7B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B")] == \
        ["gepa", "gepa", "gepa-big", "gepa-big", "gepa-big"]
    assert [gepa_est_minutes(m) for m in (DD, "EleutherAI/pythia-2.8b", "Qwen/Qwen3-4B", "allenai/Olmo-3-1025-7B")] == \
        [180, 180, 360, 360]
    assert [eval_class(m) for m in (DD, "Qwen/Qwen3-14B", "allenai/Olmo-3.1-32B-Instruct")] == ["small", "big", "32b"]
    assert default_dtype("allenai/Olmo-3-1125-32B") == "bfloat16" and default_dtype(DD) == "float32"


def test_job_files_match_contract(tmp_path):
    root, _, _ = _write(tmp_path)
    jobs = sorted((root / "gepa" / CELL).glob("*.job.json"))
    assert [p.name for p in jobs] == sorted(f"{CELL}__{s}__s{k}.job.json" for s in ("empty", "ape") for k in (0, 1, 2))
    job = json.loads((root / "gepa" / CELL / f"{CELL}__empty__s1.job.json").read_text())
    rid = f"{CELL}__empty__s1"
    golden = {
        "job_id": rid, "cell_id": CELL, "start": "empty", "seed": 1, "model": DD, "revision": DD_REV, "device": "cuda",
        "dtype": "float32", "batch_size": 64, "task": "arc_easy", "formulation": "rc", "prompt_format": dict(CANONICAL),
        "format_id": "daa93775", "num_shots": 5, "score_metric": "primary_likelihood",
        "seed_instruction": {"id": "none", "text": None}, "split": "train", "val_split": "train",
        "subset_paths": {"train": str(REPO / "configs/po/subsets/arc_easy-train-part-train-n1495-seed0.json"),
                         "train_dev": str(REPO / "configs/po/subsets/arc_easy-train-part-traindev-n748-seed0.json")},
        "max_metric_calls": 40 * 748, "max_proposals": 12, "reflection_minibatch_size": 3,
        "reflection_model": "openai/gpt-5.6-sol", "reflection_reasoning": "medium", "reflection_token_limit": 32000,
        "prompt_cap": {"words": 150, "chars": 1000, "shortening_turns": 2},
        "run_dir": str(root / "gepa" / CELL / rid),
    }
    assert {k: job[k] for k in golden} == golden
    assert set(job) == set(golden) | {"train_ids", "val_ids", "model_card"}
    assert len(job["train_ids"]) == 1495 and len(job["val_ids"]) == 748 and not set(job["train_ids"]) & set(job["val_ids"])
    assert job["model_card"]["name"] == DD
    ape = json.loads((root / "gepa" / CELL / f"{CELL}__ape__s0.job.json").read_text())
    seed = json.loads((REPO / "configs/po/instructions/ape-seed-arc_easy.json").read_text())
    assert ape["seed_instruction"] == {"id": "ape-arc_easy", "text": seed["text"]}


def test_row_index_ids_are_strings(tmp_path):
    root, _, _ = _write(tmp_path, model="Qwen/Qwen3-1.7B-Base", revision=None, task="piqa")
    job = json.loads(next((root / "gepa").glob("*/*__empty__s0.job.json")).read_text())
    assert job["revision"] is None and job["batch_size"] == 16
    assert all(isinstance(i, str) for i in job["train_ids"] + job["val_ids"]) and len(job["val_ids"]) == 919


def test_gepa_items_match_pool_contract(tmp_path):
    root, pool, paths = _write(tmp_path)
    assert [p.name for p in paths] == [f"{i:04d}-gepa-{CELL}__{s}__s{k}.json"
                                       for i, (s, k) in enumerate(((s, k) for s in ("empty", "ape") for k in (0, 1, 2)), 1)]
    item = json.loads(paths[0].read_text())
    rid = f"{CELL}__empty__s0"
    run_dir = str(root / "gepa" / CELL / rid)
    assert set(item) == {"schema", "name", "kind", "class", "est_minutes", "cwd", "command", "followup", "followup_cwd",
                         "env_files", "complete", "meta", "created_utc", "attempts"}
    assert item["schema"] == "po-pool-item/1" and item["name"] == f"gepa-{rid}" and item["kind"] == "gepa"
    assert item["class"] == "gepa" and item["est_minutes"] == 180 and item["cwd"] == "/x/olmes"
    assert item["command"] == ["uv", "run", "local/gepa_arc.py", "--job", str(root / "gepa" / CELL / f"{rid}.job.json")]
    assert item["followup"] == ["uv", "run", "--directory", str(REPO), "python", "scripts/po_gepa_cells.py", "eval-item",
                                "--run-dir", run_dir, "--pool", str(pool)]
    assert item["followup_cwd"] == str(REPO)
    assert item["env_files"] == ["/scratch/ddr8143/.config/secrets/openrouter.env"]
    assert item["complete"] == {"type": "file", "path": f"{run_dir}/result.json"}
    assert item["attempts"] == [] and item["meta"]["task"] == "arc_easy" and item["meta"]["quant"] is None
    # idempotent: a second write adds nothing and keeps the jobs
    assert _write(tmp_path)[2] == []
    big = _write(tmp_path / "b", model="allenai/Olmo-3-1125-32B", revision=None, task="csqa")[2]
    item = json.loads(big[0].read_text())
    assert item["class"] == "gepa-big" and item["est_minutes"] == 360 and item["meta"]["dtype"] == "bfloat16"


def test_mps_jobs_without_pool_keep_cuda_eval_sweep(tmp_path):
    root, pool = tmp_path / "root", tmp_path / "root" / "pool" / "p"
    assert write_cell(model=DD, revision=DD_REV, task="arc_easy", pool_dir=None, root=root, repo=REPO, device="mps") == []
    jobs = sorted((root / "gepa" / CELL).glob("*.job.json"))
    assert len(jobs) == 6 and not pool.exists()
    assert {json.loads(p.read_text())["device"] for p in jobs} == {"mps"}
    rid = f"{CELL}__empty__s0"
    (root / "gepa" / CELL / rid).mkdir()
    (root / "gepa" / CELL / rid / "result.json").write_text(json.dumps({"best_candidate": {"system_prompt": "Be exact."}}))
    sweep_dir, paths = write_eval_item(root / "gepa" / CELL / rid, pool, REPO)
    assert json.loads((sweep_dir / "sweep.json").read_text())["device"] == "cuda" and len(paths) == 1


def test_eval_item_writes_sweep_pairs_and_continues_sequence(tmp_path):
    root, pool, _ = _write(tmp_path)
    (pool / "done").mkdir()
    (pool / "done" / "0042-other.json").write_text("{}")
    rid = f"{CELL}__ape__s2"
    run_dir = root / "gepa" / CELL / rid
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"best_candidate": {"system_prompt": "Pick the best answer."}}))
    sweep_dir, paths = write_eval_item(run_dir, pool, REPO)
    assert sweep_dir == root / "sweeps" / f"ge-{rid}"
    manifest = json.loads((sweep_dir / "sweep.json").read_text())
    assert [p["instruction"] for p in manifest["pairs"]] == [{"id": "none", "text": None}, {"id": f"gepa-{rid}", "text": "Pick the best answer."}]
    assert all(p["format"] == dict(CANONICAL) for p in manifest["pairs"]) and manifest["dtype"] == "float32"
    assert [Path(p).name for p in manifest["subset_paths"]] == ["arc_easy-validation-n570-seed0.json", "arc_easy-test-part-test1-n1188-seed0.json"]
    assert len((sweep_dir / "tasks.jsonl").read_text().splitlines()) == 4
    assert [p.name for p in paths] == [f"0043-ge-{rid}.json"]
    item = json.loads(paths[0].read_text())
    assert item["kind"] == "eval" and item["class"] == "small" and item["cwd"] == str(REPO)
    assert item["command"] == ["uv", "run", "python", "scripts/po_run_sweep.py", "--sweep-dir", str(sweep_dir)]
    assert item["complete"] == {"type": "sweep", "sweep_dir": str(sweep_dir)} and item["followup"] is None
    assert write_eval_item(run_dir, pool, REPO) == (sweep_dir, [])  # rerun: sweep reused, no duplicate item


def _fake_run(gepa_root, sweeps, rid, start, best, gepa_text):
    (gepa_root / CELL).mkdir(parents=True, exist_ok=True)
    job = {"job_id": rid, "cell_id": CELL, "model": DD, "revision": DD_REV, "dtype": "float32", "task": "arc_easy",
           "start": start, "seed": 0, "device": "mps" if start == "ape" else "cuda"}
    (gepa_root / CELL / f"{rid}.job.json").write_text(json.dumps(job))
    (gepa_root / CELL / rid).mkdir()
    (gepa_root / CELL / rid / "result.json").write_text(json.dumps(
        {"best_candidate": {"system_prompt": best}, "seed_val_score": 0.3, "best_val_score": 0.4, "num_candidates": 5}))
    (sweeps / f"ge-{rid}").mkdir(parents=True)
    (sweeps / f"ge-{rid}" / "sweep.json").write_text(json.dumps(
        {"instructions": [{"id": "none", "text": None}, {"id": f"gepa-{rid}", "text": gepa_text}],
         "subsets": [{"label": "s1", "n": 2}, {"label": "s2", "n": 2}]}))


def _items(rid, gepa_acc, gepa_lik, include_gepa=True):
    base_acc, base_lik = [1, 0, 1, 0], [0.2, 0.3, 0.4, 0.5]
    rows = []
    for instr, acc, lik in (("none", base_acc, base_lik), (f"gepa-{rid}", gepa_acc, gepa_lik)):
        if instr != "none" and not include_gepa:
            continue
        for i, (a, likv) in enumerate(zip(acc, lik)):
            # two subsets share native ids 0 and 1: pairing must be by (subset, native_id)
            rows.append({"instruction_id": instr, "subset": "s1" if i < 2 else "s2", "native_id": i % 2, "primary": a, "primary_likelihood": likv})
    return pd.DataFrame(rows)


def test_aggregate_known_deltas_same_text_and_idempotent(tmp_path):
    gepa_root, sweeps, tidy = tmp_path / "gepa", tmp_path / "sweeps", tmp_path / "tidy"
    a, b = f"{CELL}__ape__s0", f"{CELL}__empty__s0"
    _fake_run(gepa_root, sweeps, a, "ape", "Answer well.", "Answer well.")
    _fake_run(gepa_root, sweeps, b, "empty", "", None)
    frames = {f"ge-{a}": _items(a, [1, 1, 1, 0], [0.3, 0.4, 0.5, 0.6]), f"ge-{b}": _items(b, [], [], include_gepa=False)}
    tidy.mkdir()
    key = model_key(DD, DD_REV, "float32")
    pd.DataFrame([
        {"contrast_type": "recipe", "a_key": "x", "b_key": "y", "task": "arc_easy", "role": "test", "metric": "accuracy",
         "delta": 0.1, "lo": 0.0, "hi": 0.2, "n_items": 10, "n_runs": 1, "method": "item-bootstrap"},
        {"contrast_type": "gepa", "a_key": key, "b_key": f"gepa:{CELL}:both", "task": "arc_easy", "role": "test",
         "metric": "accuracy", "delta": 9.0, "lo": 9.0, "hi": 9.0, "n_items": 1, "n_runs": 1, "method": "hierarchical-bootstrap"},
        {"contrast_type": "gepa", "a_key": "other", "b_key": "gepa:other__csqa:both", "task": "csqa", "role": "test",
         "metric": "accuracy", "delta": 0.5, "lo": 0.4, "hi": 0.6, "n_items": 5, "n_runs": 6, "method": "hierarchical-bootstrap"},
    ]).to_parquet(tidy / "contrasts.parquet", index=False)

    for _ in range(2):  # second pass must replace, not duplicate
        assert aggregate(tidy, gepa_root, sweeps, load_items=lambda d: frames[d.name])[2] == []
    runs = pd.read_parquet(tidy / "gepa_runs.parquet").set_index("run_id")
    assert len(runs) == 2 and (tidy / "gepa_runs.csv").exists() and (tidy / "contrasts.csv").exists()
    assert runs.loc[a, "accuracy_delta"] == pytest.approx(0.25) and runs.loc[a, "likelihood_delta"] == pytest.approx(0.1)
    assert not runs.loc[a, "same_text"] and runs.loc[b, "same_text"]
    assert (runs.loc[a, "device"], runs.loc[b, "device"]) == ("mps", "cuda")
    assert runs.loc[b, "accuracy_delta"] == 0 and runs.loc[b, "accuracy_lo"] == 0 and runs.loc[b, "n_items"] == 4

    c = pd.read_parquet(tidy / "contrasts.parquet")
    assert len(c) == 2 + 3 * 2  # recipe row, other cell's gepa row, 3 start groups x 2 metrics
    assert set(c[c.contrast_type == "recipe"].b_key) == {"y"} and "gepa:other__csqa:both" in set(c.b_key)
    g = c[c.b_key.str.startswith(f"gepa:{CELL}:")].set_index(["b_key", "metric"])
    both = g.loc[(f"gepa:{CELL}:both", "accuracy")]
    assert both.delta == pytest.approx(0.125) and both.n_runs == 2 and both.a_key == key and both.lo <= both.delta <= both.hi
    assert g.loc[(f"gepa:{CELL}:both", "likelihood")].delta == pytest.approx(0.05)
    empty = g.loc[(f"gepa:{CELL}:empty", "accuracy")]
    assert empty.delta == 0 and empty.lo == 0 and empty.hi == 0 and empty.n_runs == 1
    ape_lik = g.loc[(f"gepa:{CELL}:ape", "likelihood")]
    assert ape_lik.lo == pytest.approx(0.1) and ape_lik.hi == pytest.approx(0.1)
    assert set(g.reset_index()["method"]) == {"hierarchical-bootstrap"} and set(g.reset_index()["role"]) == {"test"}


def test_aggregate_skips_runs_whose_eval_sweep_is_partial(tmp_path):
    gepa_root, sweeps, tidy = tmp_path / "gepa", tmp_path / "sweeps", tmp_path / "tidy"
    done, partial, empty, unswept = (f"{CELL}__ape__s{k}" for k in range(4))
    for rid in (done, partial, empty):
        _fake_run(gepa_root, sweeps, rid, "ape", "Answer well.", "Answer well.")
    _fake_run(gepa_root, sweeps, unswept, "ape", "Answer well.", "Answer well.")
    (sweeps / f"ge-{unswept}" / "sweep.json").unlink()  # follow-up not run yet
    frames = {f"ge-{done}": _items(done, [1, 1, 1, 0], [0.3, 0.4, 0.5, 0.6]),
              f"ge-{partial}": _items(partial, [1, 1, 1, 0], [0.3, 0.4, 0.5, 0.6]).iloc[:-1],  # last gepa item missing
              f"ge-{empty}": pd.DataFrame()}
    contrasts, runs, incomplete = aggregate(tidy, gepa_root, sweeps, load_items=lambda d: frames[d.name])
    assert list(runs["run_id"]) == [done] and sorted(incomplete) == sorted([partial, empty, unswept])
    assert set(contrasts["n_runs"]) == {1}


def test_hierarchical_bootstrap_spans_run_spread():
    lo, hi = hierarchical_bootstrap([np.zeros(50), np.ones(50)])
    assert lo == 0.0 and hi == 1.0
    lo, hi = hierarchical_bootstrap([np.full(10, 0.2)] * 3)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


CAP = PromptCap(words=10, chars=60, shortening_turns=2)
LONG = "One two three. Four five six seven. Eight nine ten eleven twelve."


def test_truncate_to_cap_sentence_boundary_then_tokens():
    assert truncate_to_cap(LONG, CAP) == "One two three. Four five six seven."
    assert truncate_to_cap("a b c d e f g h i j k l m", CAP) == "a b c d e f g h i j"
    assert truncate_to_cap("Short. Fine.", CAP) == "Short. Fine."
    chars_bound = PromptCap(words=150, chars=20)
    assert truncate_to_cap("Abc def. Ghi jkl mno pqr.", chars_bound) == "Abc def."
    assert within_cap("x " * 150) and not within_cap("x " * 151) and not within_cap("y" * 1001)


def test_enforce_cap_turns_then_truncation():
    calls = []

    def shorten(text, turn):
        calls.append(turn)
        return text  # the proposer never manages to shorten

    r = enforce_cap(LONG, shorten, CAP)
    assert calls == [1, 2] and r.shortening_turns == 2 and r.truncated
    assert r.text == "One two three. Four five six seven." and (r.raw_words, r.raw_chars) == (12, len(LONG))
    assert (r.final_words, r.final_chars) == (7, len(r.text))
    r = enforce_cap(LONG, lambda t, n: "Shorter now.", CAP)
    assert r.shortening_turns == 1 and not r.truncated and r.text == "Shorter now."
    r = enforce_cap("  Already short.  ", lambda t, n: pytest.fail("no turn expected"), CAP)
    assert r.shortening_turns == 0 and r.text == "Already short." and not r.truncated


def test_render_demos_task_formats():
    piqa = [{"id": None, "question": "Open a jar", "choices": {"text": ["twist the lid", "kick it"], "label": ["A", "B"]},
             "answerKey": "A", "descriptor": "Goal"}]
    assert render_demos(piqa, "rc") == "Goal: Open a jar\nAnswer: twist the lid"
    wino = [{"id": None, "question": "Tom gave Bob a gift because _ was kind.", "cloze": True,
             "choices": {"text": ["Tom", "Bob"], "label": ["A", "B"]}, "answerKey": "A"}]
    assert render_demos(wino, "rc") == "Tom gave Bob a gift because Tom was kind."


@pytest.mark.skipif(not (OLMES_REPO / "oe_eval" / "tasks" / "fewshot_sources.py").exists(), reason="needs the OLMES fork checkout")
@pytest.mark.parametrize("task", TASKS)
def test_olmes_demos_every_task(task):
    demos = olmes_demos(5, task)
    assert len(demos) == 5
    prompt = build_seed_meta_prompt(task, demos)
    assert render_demos(demos, "rc") in prompt and "at most 150 words" in prompt
    if task == "piqa":
        assert prompt.count("\nGoal: ") + prompt.count(":\n\nGoal: ") >= 5
    if task == "winogrande":
        assert "_" not in render_demos(demos, "rc")

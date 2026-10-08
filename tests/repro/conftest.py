import pandas as pd
import pytest

from repro.config import load_evaluation_config


@pytest.fixture
def config():
    return load_evaluation_config()


@pytest.fixture
def raw(config):
    rows = []
    tasks = [task for task in config.benchmarks if task != "mmlu"] + [
        f"mmlu_{i}" for i in range(57)
    ]
    for size, step, compute in (("small", 5, 1.0), ("large", 10, 2.0)):
        for task in tasks:
            for seed in ("s1", "s2", "s3"):
                for recipe in ("a", "b"):
                    value = (0.0 if task.startswith("mmlu_") else 1.0) + (
                        0.1 if recipe == "b" else 0.0
                    )
                    rows.append(
                        dict(
                            params=size,
                            step=step,
                            compute=compute,
                            total_steps=10,
                            task=task,
                            data=recipe,
                            seed=seed,
                            **dict.fromkeys(config.metrics, value),
                        )
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def sweep(config):
    rows = []
    for task in config.benchmarks + ("olmes",):
        for step, ratio in ((1, 0.00001), (2, 0.0001), (3, 0.1)):
            for metric in config.metrics:
                rows.append(
                    dict(
                        predictor_size="150M",
                        predictor_step=step,
                        task=task,
                        metric=metric,
                        available=True,
                        reason="",
                        compute_ratio=ratio,
                        compute=ratio * 100,
                        schedule_complete=False,
                        decision_accuracy_std=0.0,
                        seed_accuracies=[
                            0.81 if metric == "correct_prob_per_char" else 0.6
                        ]
                        * 3,
                        decision_accuracy=0.81
                        if metric == "correct_prob_per_char"
                        else 0.6,
                    )
                )
    result = pd.DataFrame(rows)
    result.index.name = "evidence_id"
    return result

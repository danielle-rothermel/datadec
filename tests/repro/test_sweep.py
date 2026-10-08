import pandas as pd

from repro.sweep import sweep_rankings


def test_paper_sweep_preserves_missing_seed_evidence(config):
    from datadec.config import load_olmes_contract

    rows = []
    recipes = tuple(sorted(load_olmes_contract().recipe_map.values()))
    for size, step, compute, seeds in (
        ("4M", 5, 1.0, config.predictors.default_seeds),
        ("1B", 10, 100.0, config.target.seeds),
    ):
        for task in config.tasks:
            for seed in seeds:
                for index, recipe in enumerate(recipes):
                    rows.append(
                        dict(
                            params=size,
                            step=step,
                            compute=compute,
                            total_steps=10,
                            task=task,
                            seed=seed,
                            data=recipe,
                            **dict.fromkeys(config.metrics, float(index)),
                        )
                    )
    evaluations = pd.DataFrame(rows)
    # One predictor score is absent, while the target remains complete.
    evaluations = (
        evaluations.loc[
            ~(
                (evaluations["params"] == "4M")
                & (evaluations["task"] == "boolq")
                & (evaluations["seed"] == config.predictors.default_seeds[-1])
                & (evaluations["data"] == recipes[0])
            )
        ]
        .set_index(["params", "step", "task"], drop=False)
        .sort_index()
    )
    result = sweep_rankings(
        evaluations,
        config.model_copy(
            update={"target": config.target.model_copy(update={"step": 10})}
        ),
    )
    assert len(result) == 2 * len(config.tasks) * len(config.metrics)
    missing = result.query("predictor_size == '4M' and task == 'boolq'")
    assert not missing["available"].any()
    assert missing["reason"].str.contains("non-finite").all()
    available = result.loc[result["available"]]
    assert available["decision_accuracy"].eq(1).all()
    assert available["recipe_count"].eq(25).all()
    assert available["pair_count"].eq(300).all()

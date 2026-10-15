from __future__ import annotations

import pytest

from datadec.models import DataDecideCheckpoint


def test_default_seed_naming() -> None:
    checkpoint = DataDecideCheckpoint(
        recipe="dclm-baseline", size="150M", seed_ordinal=0, step=38157
    )

    assert checkpoint.source_repo_id == "allenai/DataDecide-dclm-baseline-150M"
    assert checkpoint.source_branch == "step38157-seed-default"
    assert checkpoint.target_repo_id == "drotherm/DataDecide-dclm-baseline-150M"
    assert checkpoint.target_branch == "step38157-seed0"
    assert checkpoint.final_tag == "final-seed0"
    assert checkpoint.recipe_display_name == "DCLM-Baseline"
    assert (checkpoint.seed.name, checkpoint.seed.value) == ("default", 6198)


@pytest.mark.parametrize(
    ("size", "ordinal", "name", "value", "slug"),
    [
        ("150M", 1, "small aux 2", 14, "small-aux-2"),
        ("150M", 2, "small aux 3", 15, "small-aux-3"),
        ("1B", 1, "large aux 2", 4, "large-aux-2"),
        ("1B", 2, "large aux 3", 5, "large-aux-3"),
    ],
)
def test_aux_seed_literals_depend_on_size_group(
    size: str, ordinal: int, name: str, value: int, slug: str
) -> None:
    checkpoint = DataDecideCheckpoint(
        recipe="c4", size=size, seed_ordinal=ordinal, step=1250
    )

    assert (checkpoint.seed.name, checkpoint.seed.value) == (name, value)
    assert checkpoint.source_branch == f"step1250-seed-{slug}"
    assert checkpoint.target_branch == f"step1250-seed{ordinal}"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recipe": "nope"}, "Unknown DataDecide recipe"),
        ({"size": "7B"}, "Unknown DataDecide size"),
        ({"seed_ordinal": 3}, "seed_ordinal"),
        ({"step": -1}, "step"),
    ],
)
def test_invalid_checkpoints_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "recipe": "c4",
        "size": "150M",
        "seed_ordinal": 0,
        "step": 0,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        DataDecideCheckpoint(**values)  # type: ignore[arg-type]

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from datadec.config import (
    PublishingContract,
    config_file,
    load_olmes_contract,
    load_publishing_contract,
    load_scaling_law_contract,
    load_source_manifest,
)


def test_publishing_contract_pins_target_paths_and_messages() -> None:
    contract = load_publishing_contract()

    assert config_file("publishing.toml").is_file()
    assert contract.target.repo_id == "drotherm/dd_parsed"
    assert contract.target.revision == "main"
    assert contract.ppl.remote_path == "ppl.parquet"
    assert contract.ppl.commit_message == "Publish PPL results"
    assert contract.olmes.remote_path == "olmes.parquet"
    assert contract.olmes.commit_message == "Publish aggregate OLMES results"
    assert (
        contract.scaling_law.evaluations_remote_path
        == "scaling-law/evaluations.parquet"
    )
    assert (
        contract.scaling_law.checkpoint_losses_remote_path
        == "scaling-law/checkpoint-losses.parquet"
    )
    assert contract.scaling_law.commit_message == "Publish scaling-law results"
    assert contract.olmes_details.tasks_remote_path == "olmes-details/tasks.parquet"
    assert contract.olmes_details.remote_path_templates() == (
        "olmes-details/instances/{recipe}.parquet",
        "olmes-details/choices/{recipe}.parquet",
    )
    assert (
        contract.olmes_details.commit_message_template
        == "Publish OLMES details for {recipe}"
    )
    assert contract.published_results.remote_root == "published-results"
    assert (
        contract.published_results.commit_message_template
        == "Publish published results for {unit}"
    )
    with pytest.raises(ValidationError, match="frozen"):
        setattr(contract.target, "revision", "other")


def test_publishing_contract_expands_unique_paths_for_every_detail_recipe() -> None:
    contract = load_publishing_contract()
    recipes = load_source_manifest().olmes_details.recipes

    assert len(recipes) == 25
    detail_paths = {
        template.format(recipe=recipe)
        for recipe in recipes
        for template in contract.olmes_details.remote_path_templates()
    }
    assert len(detail_paths) == 50
    assert "olmes-details/choices/fineweb-pro.parquet" in detail_paths


def _validated_contract(raw: dict[str, Any]) -> PublishingContract:
    contract = PublishingContract.model_validate(raw)
    return contract.validate_references(
        olmes_contract=load_olmes_contract(),
        scaling_law_contract=load_scaling_law_contract(),
        source_manifest=load_source_manifest(),
    )


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda raw: raw["ppl"].update({"remote_path": "/ppl.parquet"}),
            "normalized relative POSIX path",
        ),
        (
            lambda raw: raw["scaling_law"].update(
                {"checkpoint_losses_remote_path": "scaling-law/evaluations.parquet"}
            ),
            "remote paths must be unique",
        ),
        (
            lambda raw: raw["olmes_details"].update(
                {"tasks_remote_path": "olmes-details/{recipe}/tasks.parquet"}
            ),
            "must correspond to local table",
        ),
        (
            lambda raw: raw["olmes_details"].update(
                {
                    "choices_remote_path_template": (
                        "olmes-details/instances/{recipe}.parquet"
                    )
                }
            ),
            "remote path templates must be unique",
        ),
        (
            lambda raw: raw["published_results"].update(
                {"commit_message_template": "Publish {path}"}
            ),
            "must contain exactly {unit}",
        ),
        (
            lambda raw: raw["olmes"].update({"remote_path": "aggregate.parquet"}),
            "must correspond to its local table path",
        ),
    ],
)
def test_publishing_contract_rejects_invalid_paths_templates_and_references(
    mutate: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    raw = load_publishing_contract().model_dump()
    mutate(raw)

    with pytest.raises((ValidationError, ValueError), match=error):
        _validated_contract(raw)

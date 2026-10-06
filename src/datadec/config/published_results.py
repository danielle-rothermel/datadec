from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from datadec.config._load import ConfigModel, load_toml

type PublishedResultCategory = Literal[
    "scaling_law", "published_results", "published_figures"
]
type PublishedResultSchema = Literal[
    "transformed",
    "prediction_model_scale",
    "processed_ladder",
    "cheap_decisions",
    "new_eval_decision_accuracy",
    "new_eval_means",
    "target_pairs",
]
type PublishedResultUnit = Literal[
    "cheap-decisions",
    "new-eval-intermediates",
    "outputs2",
    "per-task-arc-challenge",
    "per-task-arc-easy",
    "per-task-boolq",
    "per-task-csqa",
    "per-task-hellaswag",
    "per-task-mmlu",
    "per-task-openbookqa",
    "per-task-piqa",
    "per-task-socialiqa",
    "per-task-winogrande",
    "processed-data-current",
    "processed-data-pre-extra-real",
]
type PublishedResultLogicalType = Literal["string", "int64", "float64", "bool"]


@dataclass(frozen=True, slots=True)
class PublishedResultColumn:
    name: str
    logical_type: PublishedResultLogicalType
    nullable: bool = False


@dataclass(frozen=True, slots=True)
class PublishedResultTableSchema:
    columns: tuple[PublishedResultColumn, ...]


def _columns(
    *definitions: tuple[str, PublishedResultLogicalType, bool],
) -> tuple[PublishedResultColumn, ...]:
    return tuple(PublishedResultColumn(*definition) for definition in definitions)


PUBLISHED_RESULT_SCHEMAS: dict[PublishedResultSchema, PublishedResultTableSchema] = {
    "transformed": PublishedResultTableSchema(
        _columns(
            ("model", "string", False),
            ("group", "string", False),
            ("seed", "int64", False),
            ("metric", "string", False),
            ("models", "string", False),
            ("compute_latest", "float64", False),
            ("token_latest", "float64", False),
            ("raw_values", "string", False),
            ("value", "float64", False),
        )
    ),
    "prediction_model_scale": PublishedResultTableSchema(
        _columns(
            ("binary_accuracy", "float64", True),
            ("magnitude_correlation", "float64", True),
            ("pearson_correlation", "float64", True),
            ("weighted_pearson_correlation", "float64", True),
            ("NDCG", "float64", True),
            ("correct_count", "float64", True),
            ("incorrect_count", "float64", True),
            ("abstain_count", "float64", True),
            ("total_count", "float64", True),
            ("primary_abstain", "float64", True),
            ("mix1_better", "float64", True),
            ("mix2_better", "float64", True),
            ("actual_mix1_better", "float64", True),
            ("actual_mix2_better", "float64", True),
            ("mix_pairs_incorrect", "string", False),
            ("mix_pairs_correct", "string", False),
            ("metric", "string", False),
            ("model", "string", False),
            ("seed", "int64", False),
            ("compute_limit", "float64", True),
            ("compute_latest", "float64", True),
            ("proportion", "float64", False),
            ("tokens", "float64", False),
            ("three_way_accuracy", "float64", True),
            ("compute", "float64", True),
            ("proportion_target", "float64", True),
        )
    ),
    "processed_ladder": PublishedResultTableSchema(
        _columns(
            ("model", "string", False),
            ("group", "string", False),
            ("task", "string", False),
            ("step", "int64", False),
            ("seed", "int64", False),
            ("chinchilla", "string", False),
            ("tokens", "int64", False),
            ("compute", "float64", False),
            ("metrics", "string", False),
        )
    ),
    "cheap_decisions": PublishedResultTableSchema(
        _columns(
            ("task", "string", False),
            ("mix", "string", False),
            ("metric", "string", False),
            ("setup", "string", False),
            ("step_1_y", "float64", False),
            ("step_2_y", "float64", False),
            ("stacked_y", "float64", False),
            ("step_1_pred", "float64", False),
            ("step_2_pred", "float64", False),
            ("stacked_pred", "float64", False),
            ("abs_error_step_1", "float64", False),
            ("abs_error_step_2", "float64", False),
            ("abs_error_stacked", "float64", False),
            ("rel_error_stacked", "float64", False),
        )
    ),
    "new_eval_decision_accuracy": PublishedResultTableSchema(
        _columns(
            ("size", "string", False),
            ("task", "string", False),
            ("target_ranking", "string", False),
            ("logits_per_byte_corr", "float64", False),
            ("logits_per_char_corr", "float64", False),
            ("primary_score", "float64", False),
        )
    ),
    "new_eval_means": PublishedResultTableSchema(
        _columns(
            ("size", "string", False),
            ("task", "string", False),
            ("primary_score", "float64", False),
            ("logits_per_byte_corr", "float64", False),
            ("logits_per_char_corr", "float64", False),
        )
    ),
    "target_pairs": PublishedResultTableSchema(
        _columns(
            ("pair_index", "int64", False),
            ("model_1", "string", False),
            ("model_2", "string", False),
        )
    ),
}


_SCHEMA_FILENAMES: dict[PublishedResultSchema, frozenset[str]] = {
    "transformed": frozenset({"1_metric_transformed.csv", "1_primary_transformed.csv"}),
    "prediction_model_scale": frozenset({"2_prediction_model_scale.csv"}),
    "processed_ladder": frozenset(
        {
            "results_ladder_5xC_seeds_cleaned_correct_params.csv",
            "results_ladder_5xC_seeds_cleaned_correct_params_pre_extra_real.csv",
            "results_ladder_5xC_seeds_dirty_correct_params.csv",
            "results_ladder_5xC_seeds_dirty_correct_params_pre_extra_real.csv",
        }
    ),
    "cheap_decisions": frozenset({"cheap_decisions_stacked_rc_pred_all.csv"}),
    "new_eval_decision_accuracy": frozenset({"davidh_new_evals_decision_accuracy.csv"}),
    "new_eval_means": frozenset({"davidh_new_evals_means_df.csv"}),
    "target_pairs": frozenset({"0_target_pairs.json"}),
}


def _published_result_unit_for_path(path: PurePosixPath) -> PublishedResultUnit:
    if path.as_posix() == "cheap_decisions_stacked_rc_pred_all.csv":
        return "cheap-decisions"
    if path.parts[0] == "new_eval_intermediates":
        return "new-eval-intermediates"
    if path.parts[0] == "outputs2":
        return "outputs2"
    if len(path.parts) >= 2 and path.parts[0] == "per_task_out":
        task_units: dict[str, PublishedResultUnit] = {
            "arc_challenge_out": "per-task-arc-challenge",
            "arc_easy_out": "per-task-arc-easy",
            "boolq_out": "per-task-boolq",
            "csqa_out": "per-task-csqa",
            "hellaswag_out": "per-task-hellaswag",
            "mmlu_out": "per-task-mmlu",
            "openbookqa_out": "per-task-openbookqa",
            "piqa_out": "per-task-piqa",
            "socialiqa_out": "per-task-socialiqa",
            "winogrande_out": "per-task-winogrande",
        }
        try:
            return task_units[path.parts[1]]
        except KeyError:
            pass
    if path.parts[0] == "processed_data":
        if path.name.endswith("_pre_extra_real.csv"):
            return "processed-data-pre-extra-real"
        return "processed-data-current"
    raise ValueError("structured published result path has no publication unit")


def _published_result_schema_for_path(path: PurePosixPath) -> PublishedResultSchema:
    if path.as_posix() == "cheap_decisions_stacked_rc_pred_all.csv":
        return "cheap_decisions"
    if path.parts[0] == "new_eval_intermediates":
        schemas: dict[str, PublishedResultSchema] = {
            "davidh_new_evals_decision_accuracy.csv": "new_eval_decision_accuracy",
            "davidh_new_evals_means_df.csv": "new_eval_means",
        }
        try:
            return schemas[path.name]
        except KeyError:
            pass
    if (
        path.parts[0] == "processed_data"
        and path.name in _SCHEMA_FILENAMES["processed_ladder"]
    ):
        return "processed_ladder"
    if path.parts[0] == "outputs2" or (
        len(path.parts) >= 2 and path.parts[0] == "per_task_out"
    ):
        for schema in ("target_pairs", "transformed", "prediction_model_scale"):
            if path.name in _SCHEMA_FILENAMES[schema]:
                return schema
    raise ValueError("structured published result path has no schema family")


class PublishedResultFile(ConfigModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str
    path: str
    expected_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    category: PublishedResultCategory
    publication_unit: PublishedResultUnit | None = None
    schema_: PublishedResultSchema | None = Field(default=None, alias="schema")

    @property
    def schema(self) -> PublishedResultSchema | None:
        return self.schema_

    @model_validator(mode="after")
    def validate_path_and_category(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            not path.parts
            or path.is_absolute()
            or path.as_posix() != self.path
            or ".." in path.parts
        ):
            raise ValueError("published result paths must be normalized relative paths")
        suffix = path.suffix.lower()
        is_raw = path.parts[0] == "raw_data"
        if is_raw != (self.category == "scaling_law"):
            raise ValueError("only raw_data files may use the scaling_law category")
        expected_extensions = {
            "scaling_law": {".csv"},
            "published_results": {".csv", ".json"},
            "published_figures": {".pdf", ".png"},
        }
        if suffix not in expected_extensions[self.category]:
            raise ValueError(
                f"invalid extension for published result category {self.category}"
            )
        is_structured = self.category == "published_results"
        if is_structured != (
            self.publication_unit is not None and self.schema is not None
        ):
            raise ValueError(
                "structured published results require publication_unit and schema; "
                "other categories must omit both"
            )
        if is_structured:
            try:
                expected_schema = _published_result_schema_for_path(path)
            except ValueError:
                raise ValueError(
                    "published result schema does not match its source path"
                ) from None
            if self.schema != expected_schema:
                raise ValueError(
                    "published result schema does not match its source path"
                )
            try:
                expected_unit = _published_result_unit_for_path(path)
            except ValueError:
                raise ValueError(
                    "published result publication_unit does not match its source path"
                ) from None
            if self.publication_unit != expected_unit:
                raise ValueError(
                    "published result publication_unit does not match its source path"
                )
        return self

    def parquet_relative_path(self) -> PurePosixPath:
        if self.category != "published_results":
            raise ValueError("only structured published results have Parquet outputs")
        return PurePosixPath(self.path).with_suffix(".parquet")


class PublishedResultsManifest(ConfigModel):
    folder_url: Literal[
        "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
    ]
    files: tuple[PublishedResultFile, ...]

    @model_validator(mode="after")
    def validate_unique_files(self) -> Self:
        ids = [file.id for file in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("published result Google Drive file IDs must be unique")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("published result paths must be unique")
        outputs = [
            file.parquet_relative_path().as_posix()
            for file in self.files
            if file.category == "published_results"
        ]
        if len(outputs) != len(set(outputs)):
            raise ValueError("published result Parquet output paths must be unique")
        return self


@cache
def load_published_results_manifest() -> PublishedResultsManifest:
    return PublishedResultsManifest.model_validate(load_toml("published_results.toml"))

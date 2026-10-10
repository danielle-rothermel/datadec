"""Validated experiment settings and claim selectors loaded from TOML."""

from importlib.resources import files
from pathlib import Path
import tomllib
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from repro.table_evidence import EvidenceTable


DEFAULT_CONFIG_PATH = Path("configs/repro_evaluations/magnusson2025-datadecide.toml")


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSettings(_ConfigModel):
    data_dir: Path
    output_dir: Path
    claim_inventory: Path
    paper_dir: Path
    matched_compute_tolerance: float = Field(ge=0, le=1)


class TargetSettings(_ConfigModel):
    size: str
    step: int = Field(ge=0)
    metric: str
    seeds: tuple[str, ...] = Field(min_length=1)


class PredictorSettings(_ConfigModel):
    default_seeds: tuple[str, ...] = Field(min_length=1)
    seeds_by_size: dict[str, tuple[str, ...]]

    def seeds_for(self, size: str) -> tuple[str, ...]:
        return self.seeds_by_size.get(size, self.default_seeds)


class AggregationSettings(_ConfigModel):
    benchmark_group: str
    macro_task: str
    subject_task: str
    subject_prefix: str
    subject_count: int = Field(gt=0)


class EvidenceSelection(_ConfigModel):
    tasks: tuple[str, ...] = ()
    task_groups: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    metric_groups: tuple[str, ...] = ()
    min_compute_ratio: float = Field(default=0, ge=0, le=1)
    max_compute_ratio: float = Field(default=1, ge=0, le=1)
    predictor_size: str | None = None
    related_tables: tuple[EvidenceTable, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.min_compute_ratio > self.max_compute_ratio:
            raise ValueError("minimum compute ratio cannot exceed maximum")
        return self


class ScoreCheckpoint(_ConfigModel):
    size: str
    step: int = Field(ge=0)


class MeasurementSettings(_ConfigModel):
    score_metrics: tuple[str, ...] = ()
    score_checkpoints: tuple[ScoreCheckpoint, ...] = ()
    noise_ddof: int = Field(default=0, ge=0, le=1)
    budget_ratios: tuple[float, ...] = ()
    budget_metrics: tuple[str, ...] = ()
    scaling_metrics: tuple[str, ...] = ()
    scaling_setups: tuple[str, ...] = ()
    loss_metric: str = "train_cross_entropy"

    @model_validator(mode="after")
    def validate_budgets(self) -> Self:
        if any(not 0 < ratio <= 1 for ratio in self.budget_ratios):
            raise ValueError("budget ratios must be positive and at most one")
        if self.scaling_metrics and not self.scaling_setups:
            raise ValueError("scaling metrics require explicit released setups")
        return self


class TableSelection(_ConfigModel):
    table: EvidenceTable
    tasks: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    steps: tuple[int, ...] = ()
    setups: tuple[str, ...] = ()


def _resolve_selection(
    names: tuple[str, ...],
    groups: tuple[str, ...],
    available: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    unknown = set(groups) - available.keys()
    if unknown:
        raise ValueError(f"unknown selection groups: {sorted(unknown)}")
    return tuple(
        dict.fromkeys(
            (*names, *(name for group in groups for name in available[group]))
        )
    )


class EvaluationConfig(_ConfigModel):
    run: RunSettings
    target: TargetSettings
    predictors: PredictorSettings
    aggregation: AggregationSettings
    sweep_metric_group: str
    task_groups: dict[str, tuple[str, ...]]
    metric_groups: dict[str, tuple[str, ...]]
    claims: dict[str, EvidenceSelection] = Field(min_length=1)
    default_related_tables: tuple[EvidenceTable, ...] = Field(min_length=1)
    measurements: MeasurementSettings = Field(default_factory=MeasurementSettings)
    measurement_claims: dict[str, tuple[TableSelection, ...]] = Field(
        default_factory=dict
    )

    @property
    def benchmarks(self) -> tuple[str, ...]:
        return self.task_groups[self.aggregation.benchmark_group]

    @property
    def tasks(self) -> tuple[str, ...]:
        return (*self.benchmarks, self.aggregation.macro_task)

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.metric_groups[self.sweep_metric_group]

    def tasks_for(self, claim: EvidenceSelection) -> tuple[str, ...]:
        return _resolve_selection(claim.tasks, claim.task_groups, self.task_groups)

    def metrics_for(self, claim: EvidenceSelection) -> tuple[str, ...]:
        return _resolve_selection(
            claim.metrics, claim.metric_groups, self.metric_groups
        )

    def _validate_aggregation(self) -> None:
        if self.aggregation.benchmark_group not in self.task_groups:
            raise ValueError("unknown benchmark task group")
        if self.sweep_metric_group not in self.metric_groups:
            raise ValueError("unknown sweep metric group")
        if self.aggregation.subject_task not in self.benchmarks:
            raise ValueError("subject aggregate must be an OLMES benchmark")
        if self.aggregation.macro_task in self.benchmarks:
            raise ValueError("macro task cannot be one of its own benchmarks")
        if self.target.metric not in self.metrics:
            raise ValueError("target metric must be included in sweep metrics")

    def _validate_groups(self) -> None:
        for groups in (self.task_groups, self.metric_groups):
            for name, values in groups.items():
                if not values or len(set(values)) != len(values):
                    raise ValueError(f"invalid selection group: {name}")

    def _validate_seeds(self) -> None:
        seed_sets = (
            self.target.seeds,
            self.predictors.default_seeds,
            *self.predictors.seeds_by_size.values(),
        )
        if any(not seeds or len(set(seeds)) != len(seeds) for seeds in seed_sets):
            raise ValueError("seed sets must be nonempty and unique")

    def _validate_claims(self) -> None:
        for claim_id, claim in self.claims.items():
            if not self.tasks_for(claim) or set(self.tasks_for(claim)) - set(
                self.tasks
            ):
                raise ValueError(f"invalid tasks for {claim_id}")
            if not self.metrics_for(claim) or set(self.metrics_for(claim)) - set(
                self.metrics
            ):
                raise ValueError(f"invalid metrics for {claim_id}")

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        self._validate_aggregation()
        self._validate_groups()
        self._validate_seeds()
        self._validate_claims()
        if set(
            self.measurements.score_metrics + self.measurements.budget_metrics
        ) - set(self.metrics):
            raise ValueError("measurement metrics must be included in sweep metrics")
        return self


def load_evaluation_config(path: Path | None = None) -> EvaluationConfig:
    """Load an explicit TOML file, or the bundled default experiment."""
    if path is not None:
        with path.open("rb") as file:
            return EvaluationConfig.model_validate(tomllib.load(file))
    packaged = files("repro").joinpath(DEFAULT_CONFIG_PATH.as_posix())
    if packaged.is_file():
        with packaged.open("rb") as file:
            return EvaluationConfig.model_validate(tomllib.load(file))
    return load_evaluation_config(Path(__file__).parents[2] / DEFAULT_CONFIG_PATH)

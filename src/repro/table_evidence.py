"""Named saved measurement tables and claim-to-table row references."""

from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify


@verify(UNIQUE)
class EvidenceTable(StrEnum):
    BUDGET_ACCURACY = "budget_accuracy.parquet"
    RECIPE_SCORES = "recipe_scores.parquet"
    SCORE_SUMMARIES = "score_summaries.parquet"
    NOISE_SPREAD = "noise_spread.parquet"
    SCALING_OBSERVATIONS = "scaling_observations.parquet"
    SCALING_ERRORS = "scaling_errors.parquet"
    SCALING_SUMMARIES = "scaling_summaries.parquet"
    CURVES = "curves.csv"
    PROXIES = "proxy_comparisons.parquet"
    COMPUTE_MATCHES = "matched_compute.csv"
    CROSSOVERS = "recipe_crossovers.csv"


@dataclass(frozen=True, slots=True)
class TableEvidence:
    table: EvidenceTable
    row_ids: tuple[int, ...]

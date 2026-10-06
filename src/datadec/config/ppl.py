from __future__ import annotations

from dataclasses import dataclass

from datadec.config.catalog import CHECKPOINT_ENRICHMENT_COLUMNS


@dataclass(frozen=True, slots=True)
class PPLSchemaContract:
    identity_columns: tuple[str, ...]
    checkpoint_enrichment_columns: tuple[str, ...]
    metric_columns: tuple[str, ...]

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (
            self.identity_columns
            + self.checkpoint_enrichment_columns
            + self.metric_columns
        )


PPL_SCHEMA = PPLSchemaContract(
    identity_columns=("params", "data", "seed", "step"),
    checkpoint_enrichment_columns=CHECKPOINT_ENRICHMENT_COLUMNS,
    metric_columns=(
        "wikitext_103_valppl",
        "pile_valppl",
        "c4_en_valppl",
        "m2d2_s2orc_valppl",
        "ice_valppl",
        "dolma_wiki_valppl",
        "dolma_stack_valppl",
        "dolma_reddit_valppl",
        "dolma_pes2o_valppl",
        "dolma_common_crawl_valppl",
        "dolma_books_valppl",
    ),
)

PPL_IDENTITY_COLUMNS = PPL_SCHEMA.identity_columns
PPL_METRIC_COLUMNS = PPL_SCHEMA.metric_columns
PPL_OUTPUT_COLUMNS = PPL_SCHEMA.output_columns

from datadec.data.preprocess.ppl import (
    PplPreprocessResult,
    flatten_perplexity_rows,
    group_perplexity_rows,
    preprocess_ppl,
)
from datadec.data.preprocess.published_results import (
    PublishedResultPreprocessFile,
    PublishedResultsPreprocessResult,
    preprocess_published_results,
)
from datadec.data.preprocess.scaling_law import (
    RAW_COLUMNS as SCALING_LAW_RAW_COLUMNS,
    ScalingLawPreprocessResult,
    preprocess_scaling_law,
)

_OLMES_EXPORTS = {
    "OlmesDetailsPreprocessResult",
    "OlmesPreprocessResult",
    "flatten_olmes_rows",
    "group_olmes_rows",
    "preprocess_olmes",
    "preprocess_olmes_details",
}

__all__ = [
    *sorted(_OLMES_EXPORTS),
    "PplPreprocessResult",
    "PublishedResultPreprocessFile",
    "PublishedResultsPreprocessResult",
    "flatten_perplexity_rows",
    "group_perplexity_rows",
    "preprocess_ppl",
    "preprocess_published_results",
    "preprocess_scaling_law",
    "SCALING_LAW_RAW_COLUMNS",
    "ScalingLawPreprocessResult",
]


def __getattr__(name: str):
    if name in _OLMES_EXPORTS:
        if name in {"preprocess_olmes_details", "OlmesDetailsPreprocessResult"}:
            from datadec.data.preprocess import olmes_details as olmes_module
        else:
            from datadec.data.preprocess import olmes as olmes_module

        return getattr(olmes_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

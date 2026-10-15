from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from datadec.config.eval import load_eval_contract

import pyarrow.dataset as ds
from dr_hf import (
    DatasetPin,
    NamedSubset,
    origin_subsets,
    read_rows,
    resolve_dataset_pin,
)

from datadec.data.read import ProcessedTable, read_processed_table
from datadec.eval.olmes_rc import (
    RcRequests,
    build_arc_challenge_rc_requests,
)
from datadec.eval.scoring import ChoiceScore, DecisionRule, predicted_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dr_providers import (
        LocalExecutionEvidence,
        LocalModelProvider,
        ProviderCallConfig,
        ProviderScoreResponse,
    )

    from datadec.data.artifacts import DataArtifacts

__all__ = [
    "ArcChallengeSource",
    "Comparison",
    "PublishedChoice",
    "PublishedItem",
    "PublishedKey",
    "RuleComparison",
    "ScoredItem",
    "ScoringResult",
    "compare",
    "load_arc_challenge_test",
    "load_published",
    "score_items",
]

_MISMATCH_EXAMPLES: Final = 5


def _published_prediction_columns() -> dict[DecisionRule, str]:
    columns = load_eval_contract().verification.published_prediction_columns
    return {DecisionRule(rule): column for rule, column in columns.items()}


@dataclass(frozen=True, slots=True)
class ArcChallengeSource:
    pin: DatasetPin
    subset: NamedSubset
    requests: tuple[RcRequests, ...]


def load_arc_challenge_test(ref: str = "main") -> ArcChallengeSource:
    source = load_eval_contract().arc_challenge
    pin = resolve_dataset_pin(
        source.repo_id,
        config=source.config,
        ref=ref,
        native_id_field=source.native_id_field,
    )
    rows = list(read_rows(pin))
    subsets = {subset.key.name: subset for subset in origin_subsets(pin, rows)}
    subset = subsets[source.evaluation_split]
    by_id = {
        row.native_id: row
        for row in rows
        if row.origin_split == source.evaluation_split
    }
    requests = tuple(
        build_arc_challenge_rc_requests(by_id[native_id])
        for native_id in subset.native_ids
    )
    return ArcChallengeSource(pin=pin, subset=subset, requests=requests)


@dataclass(frozen=True, slots=True)
class ScoredItem:
    native_id: str
    gold_index: int
    choices: tuple[ChoiceScore, ...]
    conditional_input_tokens: int
    warnings: int


@dataclass(frozen=True, slots=True)
class ScoringResult:
    items: tuple[ScoredItem, ...]
    execution: LocalExecutionEvidence


def score_items(
    provider: LocalModelProvider,
    config: ProviderCallConfig,
    requests: Sequence[RcRequests],
) -> ScoringResult:
    from dr_providers import ProviderScoreRequest  # noqa: PLC0415

    executions: list[LocalExecutionEvidence] = []

    def score(context: str, continuations: tuple[str, ...]) -> ProviderScoreResponse:
        evidence = provider.invoke(
            ProviderScoreRequest(
                config=config, context=context, continuations=continuations
            )
        )
        response = evidence.score_response
        if (
            response is None
            or response.usage is None
            or evidence.local_execution is None
        ):
            raise RuntimeError(f"scoring failed: {evidence.failure}")
        executions.append(evidence.local_execution)
        return response

    items = []
    for request in requests:
        conditional = score(request.context, request.continuations)
        unconditional = score(request.unconditional_context, request.continuations)
        choices = tuple(
            ChoiceScore.for_continuation(
                continuation,
                log_likelihood=cond.log_likelihood,
                token_count=cond.token_count,
                unconditional_log_likelihood=uncond.log_likelihood,
            )
            for continuation, cond, uncond in zip(
                request.continuations,
                conditional.scores,
                unconditional.scores,
                strict=True,
            )
        )
        items.append(
            ScoredItem(
                native_id=request.native_id,
                gold_index=request.gold_index,
                choices=choices,
                conditional_input_tokens=_prompt_tokens(conditional),
                warnings=len(conditional.warnings) + len(unconditional.warnings),
            )
        )
    if not executions:
        raise ValueError("score_items requires at least one request")
    if any(execution != executions[0] for execution in executions):
        raise RuntimeError("local execution conditions changed during scoring")
    return ScoringResult(items=tuple(items), execution=executions[0])


def _prompt_tokens(response: ProviderScoreResponse) -> int:
    if response.usage is None or response.usage.prompt_tokens is None:
        raise RuntimeError("score response carries no prompt token usage")
    return response.usage.prompt_tokens


@dataclass(frozen=True, slots=True)
class PublishedKey:
    recipe: str
    params: str
    seed: str
    step: int
    task: str = field(default_factory=lambda: load_eval_contract().arc_challenge.task)


@dataclass(frozen=True, slots=True)
class PublishedChoice:
    sum_logits: float
    sum_logits_uncond: float | None
    num_tokens: int
    num_tokens_all: int
    logits_per_byte: float | None


@dataclass(frozen=True, slots=True)
class PublishedItem:
    native_id: str
    doc_id: int
    label: int
    predictions: dict[DecisionRule, int]
    choices: tuple[PublishedChoice, ...]


def load_published(
    artifacts: DataArtifacts, key: PublishedKey
) -> dict[str, PublishedItem]:
    filters = (
        (ds.field("recipe") == key.recipe)
        & (ds.field("params") == key.params)
        & (ds.field("seed") == key.seed)
        & (ds.field("step") == key.step)
        & (ds.field("task") == key.task)
    )
    instances = read_processed_table(
        artifacts,
        ProcessedTable.OLMES_DETAILS_INSTANCES,
        filters=filters,
        columns=[
            "doc_id",
            "native_id",
            "label",
            *_published_prediction_columns().values(),
        ],
    ).to_pylist()
    choices = read_processed_table(
        artifacts,
        ProcessedTable.OLMES_DETAILS_CHOICES,
        filters=filters,
        columns=[
            "doc_id",
            "choice_index",
            "sum_logits",
            "sum_logits_uncond",
            "num_tokens",
            "num_tokens_all",
            "logits_per_byte",
        ],
    ).to_pylist()
    if not instances:
        raise LookupError(f"no published instances for {key}")
    by_doc: dict[int, list[dict[str, object]]] = {}
    for row in sorted(
        choices,
        key=lambda r: (int(r["doc_id"]), int(r["choice_index"])),  # type: ignore[arg-type]
    ):
        by_doc.setdefault(int(row["doc_id"]), []).append(row)  # type: ignore[arg-type]
    published = {}
    for row in instances:
        doc_id = int(row["doc_id"])
        doc_choices = tuple(
            PublishedChoice(
                sum_logits=float(choice["sum_logits"]),  # type: ignore[arg-type]
                sum_logits_uncond=_optional_float(choice["sum_logits_uncond"]),
                num_tokens=int(choice["num_tokens"]),  # type: ignore[arg-type]
                num_tokens_all=int(choice["num_tokens_all"]),  # type: ignore[arg-type]
                logits_per_byte=_optional_float(choice["logits_per_byte"]),
            )
            for choice in by_doc[doc_id]
        )
        predictions = {
            rule: int(row[column])  # type: ignore[arg-type]
            for rule, column in _published_prediction_columns().items()
        }
        bits_per_byte = [choice.logits_per_byte for choice in doc_choices]
        if all(value is not None for value in bits_per_byte):
            values = [float(value) for value in bits_per_byte]  # type: ignore[arg-type]
            predictions[DecisionRule.PER_BYTE] = (
                per_byte_prediction_from_published_bits(values)
            )
        native_id = str(row["native_id"])
        published[native_id] = PublishedItem(
            native_id=native_id,
            doc_id=doc_id,
            label=int(row["label"]),  # type: ignore[arg-type]
            predictions=predictions,
            choices=doc_choices,
        )
    return published


def per_byte_prediction_from_published_bits(bits_per_byte: list[float]) -> int:
    return bits_per_byte.index(min(bits_per_byte))


def forward_input_tokens_from_num_tokens_all(num_tokens_all: int) -> int:
    return num_tokens_all - 1


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RuleComparison:
    rule: DecisionRule
    agreement: float
    accuracy_ours: float
    accuracy_published: float
    first_mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Comparison:
    items: int
    choices: int
    rules: tuple[RuleComparison, ...]
    sum_logits_mean_abs_diff: float
    sum_logits_max_abs_diff: float
    sum_logits_uncond_mean_abs_diff: float
    sum_logits_uncond_max_abs_diff: float
    continuation_token_agreement: float
    context_token_agreement: float
    warnings: int
    acceptance: dict[str, bool] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return all(self.acceptance.values())

    def rule(self, rule: DecisionRule) -> RuleComparison:
        return next(item for item in self.rules if item.rule is rule)


def compare(
    ours: Sequence[ScoredItem], published: dict[str, PublishedItem]
) -> Comparison:
    our_ids = {item.native_id for item in ours}
    if our_ids != set(published):
        missing = sorted(set(published) - our_ids)[:5]
        extra = sorted(our_ids - set(published))[:5]
        raise ValueError(
            f"item sets differ: published-only {missing}, ours-only {extra}"
        )
    rules = []
    for rule in DecisionRule:
        agree = 0
        correct_ours = 0
        correct_published = 0
        mismatches: list[str] = []
        for item in ours:
            reference = published[item.native_id]
            ours_index = predicted_index(rule, item.choices)
            published_index = reference.predictions[rule]
            agree += ours_index == published_index
            correct_ours += ours_index == item.gold_index
            correct_published += published_index == reference.label
            if ours_index != published_index and len(mismatches) < _MISMATCH_EXAMPLES:
                mismatches.append(
                    f"{item.native_id}: ours {ours_index}, published {published_index}"
                )
        rules.append(
            RuleComparison(
                rule=rule,
                agreement=agree / len(ours),
                accuracy_ours=correct_ours / len(ours),
                accuracy_published=correct_published / len(ours),
                first_mismatches=tuple(mismatches),
            )
        )
    diffs: list[float] = []
    uncond_diffs: list[float] = []
    token_agree = 0
    context_agree = 0
    for item in ours:
        reference = published[item.native_id]
        if len(reference.choices) != len(item.choices):
            raise ValueError(f"{item.native_id}: choice counts differ")
        if reference.label != item.gold_index:
            raise ValueError(f"{item.native_id}: gold labels differ")
        for mine, theirs in zip(item.choices, reference.choices, strict=True):
            diffs.append(abs(mine.log_likelihood - theirs.sum_logits))
            if (
                theirs.sum_logits_uncond is not None
                and mine.unconditional_log_likelihood is not None
            ):
                uncond_diffs.append(
                    abs(mine.unconditional_log_likelihood - theirs.sum_logits_uncond)
                )
            token_agree += mine.token_count == theirs.num_tokens
        context_agree += item.conditional_input_tokens == sum(
            forward_input_tokens_from_num_tokens_all(choice.num_tokens_all)
            for choice in reference.choices
        )
    comparison = Comparison(
        items=len(ours),
        choices=len(diffs),
        rules=tuple(rules),
        sum_logits_mean_abs_diff=statistics.fmean(diffs),
        sum_logits_max_abs_diff=max(diffs),
        sum_logits_uncond_mean_abs_diff=statistics.fmean(uncond_diffs),
        sum_logits_uncond_max_abs_diff=max(uncond_diffs),
        continuation_token_agreement=token_agree / len(diffs),
        context_token_agreement=context_agree / len(ours),
        warnings=sum(item.warnings for item in ours),
    )
    thresholds = load_eval_contract().verification.acceptance
    acceptance = {
        "pmi_agreement": comparison.rule(DecisionRule.PMI).agreement
        > thresholds.min_predicted_index_agreement_pmi,
        "per_char_agreement": comparison.rule(DecisionRule.PER_CHAR).agreement
        > thresholds.min_predicted_index_agreement_per_char,
        "mean_abs_sum_logits_diff": comparison.sum_logits_mean_abs_diff
        < thresholds.max_mean_abs_sum_logits_diff_nats,
    }
    return replace(comparison, acceptance=acceptance)

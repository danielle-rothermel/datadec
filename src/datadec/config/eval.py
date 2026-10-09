from __future__ import annotations

from functools import cache
from typing import Self

from pydantic import Field, model_validator

from datadec.config._load import ConfigModel, load_toml
from datadec.config.olmes import load_olmes_contract


class ArcChallengeSourceContract(ConfigModel):
    repo_id: str = Field(min_length=1)
    config: str = Field(min_length=1)
    native_id_field: str = Field(min_length=1)
    evaluation_split: str = Field(min_length=1)
    task: str = Field(min_length=1)


class FewShotSourceContract(ConfigModel):
    repo: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    license: str = Field(min_length=1)


class FewShotContract(ConfigModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: tuple[str, ...] = Field(min_length=2)
    answer_key: str = Field(min_length=1)


class OlmesRcContract(ConfigModel):
    question_prefix: str
    answer_prefix: str
    continuation_prefix: str
    shot_separator: str
    unconditional_context: str
    answer_letters: str = Field(min_length=1)
    numeric_answer_keys: dict[str, str]
    arc_challenge_shots_source: FewShotSourceContract
    arc_challenge_shots: tuple[FewShotContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_answer_keys(self) -> Self:
        letters = set(self.answer_letters)
        if len(letters) != len(self.answer_letters):
            raise ValueError("answer_letters must be unique")
        bad = {k: v for k, v in self.numeric_answer_keys.items() if v not in letters}
        if bad:
            raise ValueError(f"numeric answer keys map outside answer_letters: {bad}")
        for shot in self.arc_challenge_shots:
            letter = self.numeric_answer_keys.get(shot.answer_key, shot.answer_key)
            if letter not in letters or self.answer_letters.index(letter) >= len(
                shot.choices
            ):
                raise ValueError(
                    f"shot {shot.id} has an answer key outside its choices"
                )
        return self


class VerificationAcceptanceContract(ConfigModel):
    min_predicted_index_agreement_pmi: float = Field(ge=0.0, le=1.0)
    min_predicted_index_agreement_per_char: float = Field(ge=0.0, le=1.0)
    max_mean_abs_sum_logits_diff_nats: float = Field(gt=0.0)


class VerificationContract(ConfigModel):
    acceptance: VerificationAcceptanceContract
    published_prediction_columns: dict[str, str]

    @model_validator(mode="after")
    def validate_columns_exist_in_olmes_details(self) -> Self:
        known = set(load_olmes_contract().metrics.detailed_instances)
        unknown = sorted(set(self.published_prediction_columns.values()) - known)
        if unknown:
            raise ValueError(f"not OLMES detailed instance columns: {unknown}")
        return self


class EvalContract(ConfigModel):
    arc_challenge: ArcChallengeSourceContract
    olmes_rc: OlmesRcContract
    verification: VerificationContract


@cache
def load_eval_contract() -> EvalContract:
    return EvalContract.model_validate(load_toml("eval.toml"))

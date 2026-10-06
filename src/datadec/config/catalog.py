from __future__ import annotations

from functools import cache
from typing import Literal, Self

from pydantic import Field, model_validator

from datadec.config._load import ConfigModel, load_toml

type CheckpointLogicalType = Literal["string", "int64", "float64", "bool"]

MODEL_DETAIL_TYPES: tuple[tuple[str, CheckpointLogicalType], ...] = (
    ("default_seed", "int64"),
    ("length_str", "string"),
    ("lr_warmup_start", "float64"),
    ("d_model", "int64"),
    ("n_heads", "int64"),
    ("n_layers", "int64"),
    ("mlp_ratio", "int64"),
    ("weight_tying", "bool"),
    ("alibi", "bool"),
    ("rope", "bool"),
    ("flash_attention", "bool"),
    ("attention_dropout", "float64"),
    ("attention_layer_norm", "bool"),
    ("include_bias", "bool"),
    ("layer_norm_type", "string"),
    ("layer_norm_with_affine", "bool"),
    ("layer_norm_eps", "float64"),
    ("bias_for_layer_norm", "bool"),
    ("attention_layer_norm_with_affine", "bool"),
    ("activation_type", "string"),
    ("residual_dropout", "float64"),
    ("embedding_dropout", "float64"),
    ("max_sequence_length", "int64"),
    ("vocab_size", "int64"),
    ("embedding_size", "int64"),
    ("eos_token_id", "int64"),
    ("pad_token_id", "int64"),
    ("init_device", "string"),
    ("init_fn", "string"),
    ("init_std", "float64"),
    ("init_cutoff_factor", "int64"),
    ("nominal_parameter_count", "int64"),
    ("training_parameter_count", "int64"),
    ("exact_parameter_count", "int64"),
    ("batch_size", "int64"),
    ("total_tokens", "int64"),
    ("warmup_tokens", "int64"),
    ("lr_max", "float64"),
    ("lr_final", "float64"),
    ("total_steps", "int64"),
    ("total_seqs", "int64"),
    ("warmup_perc", "float64"),
    ("warmup_steps", "int64"),
    ("lr_decay_tokens", "int64"),
    ("lr_decay_steps", "int64"),
    ("tokens_per_step", "int64"),
    ("compute_per_step", "float64"),
)

CHECKPOINT_ENRICHMENT_TYPES: tuple[tuple[str, CheckpointLogicalType], ...] = (
    ("tokens", "int64"),
    ("compute", "float64"),
    *MODEL_DETAIL_TYPES,
    ("lr_at_step", "float64"),
    ("cumulative_lr", "float64"),
)
CHECKPOINT_ENRICHMENT_COLUMNS: tuple[str, ...] = tuple(
    name for name, _ in CHECKPOINT_ENRICHMENT_TYPES
)


class TrainingConstants(ConfigModel):
    max_sequence_length: int
    token_length_multiplier: int
    model_size_normalization: int
    learning_rate_exponent: float
    learning_rate_base: float
    final_learning_rate_ratio: float
    batch_size_coefficient: int
    batch_size_exponent: float
    gpus_per_node: int
    microbatch_size: int
    flops_per_token_per_parameter: int = Field(gt=0)


class ModelDefaults(ConfigModel):
    default_seed: int
    length_str: str
    lr_warmup_start: float
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    weight_tying: bool
    alibi: bool
    rope: bool
    flash_attention: bool
    attention_dropout: float
    attention_layer_norm: bool
    include_bias: bool
    layer_norm_type: str
    layer_norm_with_affine: bool
    layer_norm_eps: float
    bias_for_layer_norm: bool
    attention_layer_norm_with_affine: bool
    activation_type: str
    residual_dropout: float
    embedding_dropout: float
    max_sequence_length: int
    vocab_size: int
    embedding_size: int
    eos_token_id: int
    pad_token_id: int
    init_device: str
    init_fn: str
    init_std: float
    init_cutoff_factor: int


class ModelDefinition(ConfigModel):
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    nominal_parameter_count: int = Field(gt=0)
    training_parameter_count: int = Field(gt=0)
    exact_parameter_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_nominal_parameter_count(self) -> Self:
        suffix_multipliers = {"M": 1_000_000, "B": 1_000_000_000}
        suffix = self.name[-1:]
        try:
            expected = int(self.name[:-1]) * suffix_multipliers[suffix]
        except (KeyError, ValueError):
            raise ValueError(
                "model names must encode nominal parameter counts with M or B"
            ) from None
        if self.nominal_parameter_count != expected:
            raise ValueError("model nominal_parameter_count must match its model name")
        return self


class RecipeGroups(ConfigModel):
    base_recipes: list[str]
    base_and_quality_control: list[str]
    without_ablations: list[str]
    custom_families: dict[str, list[str]]
    perplexity_performance: dict[str, list[str]]
    olmes_performance: dict[str, list[str]]


class DataDecideCatalog(ConfigModel):
    metric_names: list[str]
    drop_metrics: list[str]
    training: TrainingConstants
    model_defaults: ModelDefaults
    models: list[ModelDefinition]
    data_recipe_families: dict[str, list[str]]
    seed_map: dict[str, int]
    recipe_groups: RecipeGroups

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return self


@cache
def load_catalog() -> DataDecideCatalog:
    return DataDecideCatalog.model_validate(load_toml("catalog.toml"))

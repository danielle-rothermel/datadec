from __future__ import annotations

from datadec.config.checkpoints import load_checkpoint_contract


def test_conversion_literals_are_pinned() -> None:
    conversion = load_checkpoint_contract().conversion
    assert conversion.source_model_type == "hf_olmo"
    assert conversion.source_tensor_prefix == "model.transformer."
    assert conversion.target_model_type == "llama"
    assert conversion.target_architecture == "LlamaForCausalLM"
    assert conversion.weights_filename == "model.safetensors"
    assert conversion.conversion_record_filename == "conversion.json"


def test_supported_source_layout_is_pinned() -> None:
    layout = load_checkpoint_contract().conversion.supported_source_layout
    assert layout.required_absent == (
        "clip_qkv",
        "mlp_hidden_size",
        "multi_query_attention",
        "n_kv_heads",
    )
    assert layout.required_values == {
        "activation_type": "swiglu",
        "alibi": False,
        "attention_layer_norm": False,
        "bias_for_layer_norm": False,
        "block_group_size": 1,
        "block_type": "sequential",
        "embedding_layer_norm": False,
        "include_bias": False,
        "layer_norm_type": "rms",
        "layer_norm_with_affine": True,
        "norm_after": False,
        "rope": True,
        "scale_logits": False,
        "weight_tying": False,
    }


def test_tensor_map_covers_every_olmo_tensor_once() -> None:
    rules = load_checkpoint_contract().conversion.tensor_map
    assert [rule.source for rule in rules] == [
        "wte.weight",
        "blocks.{layer}.att_proj.weight",
        "blocks.{layer}.attn_out.weight",
        "blocks.{layer}.ff_proj.weight",
        "blocks.{layer}.ff_out.weight",
        "blocks.{layer}.attn_norm.weight",
        "blocks.{layer}.ff_norm.weight",
        "ln_f.weight",
        "ff_out.weight",
    ]
    assert [rule.transform for rule in rules if len(rule.targets) > 1] == [
        "split_rows_equal",
        "split_rows_equal",
    ]


def test_strict_check_prompts_are_pinned() -> None:
    strict = load_checkpoint_contract().strict_check
    assert strict.python == "3.12"
    assert len(strict.prompts) == 5
    assert strict.prompts[0].startswith("Question: George wants to warm his hands")
    assert [r.requirement for r in strict.reference_environment] == [
        "ai2-olmo==0.6.0",
        "transformers==4.50.3",
        "huggingface-hub<1.0",
        "datasets",
    ]

# This module contains test cases for the Ministral3 model, designed to verify the precision of forward outcomes
# between PyTorch and MindSpore implementations.

import inspect

import numpy as np
import pytest
import torch
from transformers import Ministral3Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]  # 1: pynative mode (graph mode may not be fully supported yet)


class Ministral3ModelTester:
    config_class = Ministral3Config

    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        # config
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_dropout=0.0,
        sliding_window=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters or {
            "rope_theta": 10000.0,
            "rope_type": "default",
        }
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window

    def get_config(self):
        return Ministral3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_parameters=self.rope_parameters,
            attention_dropout=self.attention_dropout,
            sliding_window=self.sliding_window,
        )

    def prepare_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = None
        if self.use_input_mask:
            attention_mask = np.ones((self.batch_size, self.seq_length), dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# Test cases: [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map]
test_cases = [
    [
        "Ministral3Model",
        "transformers.Ministral3Model",
        "mindone.transformers.Ministral3Model",
        (Ministral3ModelTester().get_config(),),
        {},
        (),
        Ministral3ModelTester().prepare_inputs(),
        {"loss": 0, "logits": 1},
    ],
    [
        "Ministral3ForCausalLM",
        "transformers.Ministral3ForCausalLM",
        "mindone.transformers.Ministral3ForCausalLM",
        (Ministral3ModelTester().get_config(),),
        {},
        (),
        Ministral3ModelTester().prepare_inputs(),
        {"loss": 0, "logits": 1},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", test_cases
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
):
    """
    Generic test function for comparing PyTorch and MindSpore module outputs.
    
    Args:
        name: Test case name
        pt_module: PyTorch module path
        ms_module: MindSpore module path
        init_args: Initialization args
        init_kwargs: Initialization kwargs
        inputs_args: Forward pass args
        inputs_kwargs: Forward pass kwargs
        outputs_map: Mapping of output names to indices
    """
    (
        ms_dtype,
        pt_dtype,
        dtype,
    ) = MS_DTYPE_MAPPING["fp32"], PT_DTYPE_MAPPING["fp32"], "fp32"

    # Get module classes
    ms_model, pt_model = get_modules(ms_module, pt_module)

    # Parse and prepare arguments
    ms_init_args, pt_init_args = generalized_parse_args(init_args, "init_args")
    ms_init_kwargs, pt_init_kwargs = generalized_parse_args(init_kwargs, "init_kwargs")
    ms_inputs_args, pt_inputs_args = generalized_parse_args(inputs_args, "inputs_args")
    ms_inputs_kwargs, pt_inputs_kwargs = generalized_parse_args(inputs_kwargs, "inputs_kwargs")

    # Initialize models
    pt_model_instance = pt_model(*pt_init_args, **pt_init_kwargs).to(pt_dtype).eval()
    ms_model_instance = ms_model(*ms_init_args, **ms_init_kwargs).to_float(ms_dtype)
    ms_model_instance.set_train(False)

    # Load PyTorch weights into MindSpore model
    ms_params = ms_model_instance.parameters_dict()
    for name, param in pt_model_instance.named_parameters():
        if name in ms_params:
            ms_params[name].set_data(
                ms.Tensor(param.detach().cpu().numpy(), dtype=ms_params[name].dtype)
            )

    # Run models
    with torch.no_grad():
        pt_outputs = pt_model_instance(*pt_inputs_args, **pt_inputs_kwargs)

    ms_outputs = ms_model_instance(*ms_inputs_args, **ms_inputs_kwargs)

    # Compare outputs
    diffs = compute_diffs(pt_outputs, ms_outputs, outputs_map, pt_dtype, ms_dtype)

    threshold = DTYPE_AND_THRESHOLDS[dtype]
    for key, (max_diff, mean_diff) in diffs.items():
        assert (
            max_diff < threshold
        ), f"{name} output {key}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, threshold={threshold:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

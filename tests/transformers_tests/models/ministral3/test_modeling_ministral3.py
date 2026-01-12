# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the MindSpore Ministral3 model."""

import unittest

import mindspore as ms
from mindspore import ops

from mindone.transformers import Ministral3Config, Ministral3Model


class Ministral3ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = Ministral3Config(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        ms.set_context(mode=ms.PYNATIVE_MODE)

    def test_model_creation(self):
        model = Ministral3Model(self.config)
        self.assertIsNotNone(model)
        
    def test_model_forward(self):
        model = Ministral3Model(self.config)
        input_ids = ops.randint(0, self.config.vocab_size, (2, 16))
        output = model(input_ids)
        self.assertEqual(output.last_hidden_state.shape, (2, 16, self.config.hidden_size))


if __name__ == "__main__":
    unittest.main()

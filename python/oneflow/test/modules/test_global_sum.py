"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=False, rtol=1e-3, atol=1e-4)
def _test_global_sum_against_pytorch(test_case, placement, sbp):
    x = random_tensor(4, 8, 16, 8, 24).to_global(placement, sbp)
    y = torch.sum(x)
    return y


@autotest(n=3, check_graph=False)
def _test_global_sum_with_0_size_tensor(test_case, placement, sbp):
    x = random_tensor(4, 8, 16, 0, 24).to_global(placement, sbp)
    y = torch.sum(x, dim=random(0, 3).to(int))
    return y


class TestGlobalSumModule(flow.unittest.TestCase):
    @globaltest
    def test_global_sum_against_pytorch(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_global_sum_against_pytorch(test_case, placement, sbp)

    @globaltest
    def test_global_sum_with_0_size_tensor(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4, valid_split_axis=[0, 1, 3]):
                _test_global_sum_with_0_size_tensor(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

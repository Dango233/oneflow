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
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestLogAddExpModule(flow.unittest.TestCase):
    @autotest(n=3, check_graph=True)
    def test_log_add_exp_against_pytorch(test_case):
        device = random_device()
        dim1 = random(1, 5)
        dim2 = random(1, 5)
        x = random_tensor(2, dim1, dim2).to(device)
        y = random_tensor(2, dim1, dim2).to(device)
        z = torch.logaddexp(x, y)
        return z

    @autotest(n=3, check_graph=True)
    def test_log_add_exp_with_0dim_tensor(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = random_tensor(2, random(1, 5), random(1, 5)).to(device)
        z = torch.logaddexp(x, y)
        return z

    @autotest(n=3, check_graph=True)
    def test_tensor_log_add_exp_against_pytorch(test_case):
        device = random_device()
        dim1 = random(1, 5)
        dim2 = random(1, 5)
        x = random_tensor(2, dim1, dim2).to(device)
        y = random_tensor(2, dim1, dim2).to(device)
        z = x.logaddexp(y)
        return z

    @autotest(n=3, check_graph=True)
    def test_tensor_log_add_exp_with_0dim_tensor(test_case):
        device = random_device()
        y = random_tensor(ndim=0).to(device)
        x = random_tensor(2, random(1, 5), random(1, 5)).to(device)
        z = x.logaddexp(y)
        return z


if __name__ == "__main__":
    unittest.main()

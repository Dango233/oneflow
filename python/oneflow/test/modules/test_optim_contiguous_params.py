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
from copy import deepcopy
import pdb

import oneflow
from oneflow import nn
from oneflow.test_utils.test_util import GenArgDict
from oneflow.nn.optimizer.contiguous_params import ContiguousParams


def test_equal_optimizer_update(
    test_case, device, x_shape, param_num, weight_decay, learning_rate, train_iters
):
    xx = []
    for i in range(train_iters):
        xx.append(np.random.uniform(size=x_shape).astype(np.float32))
    yy = np.random.uniform(size=x_shape).astype(np.float32)
    ce = nn.CrossEntropyLoss()

    model_ref = nn.Sequential(*[nn.Linear(x_shape[-1], 
                            x_shape[-1]) for i in range(param_num)])
    model_ref = model_ref.to(device)
    optimizer = oneflow.optim.SGD(model_ref.parameters(), lr=learning_rate, 
                                    weight_decay=weight_decay)
    
    model_c = deepcopy(model_ref)
    parameters_c = ContiguousParams(model_c.parameters())
    optimizer_c = oneflow.optim.SGD(parameters_c.contiguous(), lr=learning_rate,
                                    weight_decay=weight_decay)

    for model, optimizer in zip([model_ref, model_c], [optimizer, optimizer_c]):
        for i in range(train_iters):
            x = oneflow.tensor(xx[i], device=device)
            y = oneflow.tensor(yy, device=device)
            #print(list(model.parameters())[0][0])
            loss = ce(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for p1, p2 in zip(model_ref.parameters(), model_c.parameters()):
        test_case.assertTrue(np.allclose(p1.numpy(), p2.numpy(), atol=1e-06))

@oneflow.unittest.skip_unless_1n1d()
class Test(oneflow.unittest.TestCase):
    def test_multi_tensor_sgd_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["x_shape"] = [(1, 8), (1, 20)]
        arg_dict["param_num"] = [10, 20] # too few params may cause the gradient to disappear
        arg_dict["weight_decay"] = [0.0, 0.5]
        arg_dict["learning_rate"] = [1.0, 1e-3]
        arg_dict["train_iters"] = [5]
        for arg in GenArgDict(arg_dict):
            test_equal_optimizer_update(test_case, **arg)

if __name__ == "__main__":
    unittest.main()

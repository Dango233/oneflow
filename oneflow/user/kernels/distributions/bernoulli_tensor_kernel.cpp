/*
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
*/
#include <math.h>
#include <array>
#include <cmath>
#include <cstdint>

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

// NOTE(Liang Depeng): The implementation of BernoulliTensorCpuKerenl is modified from
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/DistributionTemplates.h
namespace oneflow {

template<typename T>
class BernoulliTensorCpuKerenl final : public user_op::OpKernel {
 public:
  BernoulliTensorCpuKerenl() = default;
  ~BernoulliTensorCpuKerenl() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCPU));
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* p_blob = ctx->Tensor4ArgNameAndIndex("p", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    const T* p_dptr = p_blob->dptr<T>();
    T* out_dptr = out_blob->mut_dptr<T>();
    const int64_t elem_cnt = p_blob->shape_view().elem_cnt();
    CHECK_GT(elem_cnt, 0);

    auto* kernel_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const auto& generator = kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto& cpu_generator = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    std::lock_guard<std::mutex> lock(cpu_generator->mutex_);
    one::pytorch_mt19937_engine& engine = cpu_generator->torch_engine();
    if (p_blob->data_type() == DataType::kDouble) {
      for (int64_t i = 0; i < elem_cnt; ++i) {
        uint64_t rand_unit = random64(engine);
        double random_val = uniform_real(rand_unit, 0.0, 1.0);
        out_dptr[i] = static_cast<T>(random_val < p_dptr[i]);
      }
    } else {
      for (int64_t i = 0; i < elem_cnt; ++i) {
        uint32_t rand_unit = random(engine);
        float random_val = uniform_real(rand_unit, 0.0f, 1.0f);
        out_dptr[i] = static_cast<T>(random_val < p_dptr[i]);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BERNOULLI_TENSOR_CPU_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("bernoulli_tensor")                                            \
      .SetCreateFn<BernoulliTensorCpuKerenl<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                 \
                       && (user_op::HobDataType("p", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_BERNOULLI_TENSOR_CPU_KERNEL(float)
REGISTER_BERNOULLI_TENSOR_CPU_KERNEL(double)

}  // namespace oneflow

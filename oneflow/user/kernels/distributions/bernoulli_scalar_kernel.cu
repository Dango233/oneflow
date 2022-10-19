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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/cuda_macros.h"

// NOTE(Liang Depeng): the implementation of BernoulliScalarGpuKerenl is modified from
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/DistributionTemplates.h
namespace oneflow {

namespace {

OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_double(int32_t numel, uint64_t seed,
                                                                   uint64_t offset, double p,
                                                                   double* out_ptr) {
  int32_t unroll_factor = 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    double2 rand = curand_uniform2_double(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) { out_ptr[li] = static_cast<double>((&rand.x)[ii] < p); }
    }
    __syncthreads();
  }
}

OF_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel_float(int32_t numel, uint64_t seed,
                                                                  uint64_t offset, float p,
                                                                  float* out_ptr) {
  int32_t unroll_factor = 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  int rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) * blockDim.x
                     * gridDim.x * unroll_factor;
  for (int32_t linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    float4 rand = curand_uniform4(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) { out_ptr[li] = static_cast<float>((&rand.x)[ii] < p); }
    }
    __syncthreads();
  }
}

}  // namespace

class BernoulliScalarGpuFloatKerenl final : public user_op::OpKernel {
 public:
  BernoulliScalarGpuFloatKerenl() = default;
  ~BernoulliScalarGpuFloatKerenl() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    float* out_dptr = out_blob->mut_dptr<float>();
    const int64_t elem_cnt = out_blob->shape_view().elem_cnt();
    CHECK_GT(elem_cnt, 0);

    auto* kernel_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const auto& generator = kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto& gpu_generator = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    float p = ctx->Attr<float>("p");

    ep::CudaStream* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    auto execution_policy = calc_execution_policy(elem_cnt, cuda_stream);

    auto counter_offset = std::get<0>(execution_policy);
    auto grid = std::get<1>(execution_policy);
    auto block = std::get<2>(execution_policy);

    uint64_t offset = 0;
    uint64_t seed = gpu_generator->current_seed();
    {
      std::lock_guard<std::mutex> lock(gpu_generator->mutex_);
      offset = gpu_generator->get_philox_offset(counter_offset);
    }

    distribution_elementwise_grid_stride_kernel_float<<<
        grid, block, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(elem_cnt, seed,
                                                                              offset, p, out_dptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class BernoulliScalarGpuDoubleKerenl final : public user_op::OpKernel {
 public:
  BernoulliScalarGpuDoubleKerenl() = default;
  ~BernoulliScalarGpuDoubleKerenl() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    double* out_dptr = out_blob->mut_dptr<double>();
    const int64_t elem_cnt = out_blob->shape_view().elem_cnt();
    CHECK_GT(elem_cnt, 0);

    auto* kernel_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    const auto& generator = kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto& gpu_generator = CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>());
    double p = static_cast<double>(ctx->Attr<float>("p"));

    ep::CudaStream* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    auto execution_policy = calc_execution_policy(elem_cnt, cuda_stream);

    auto counter_offset = std::get<0>(execution_policy);
    auto grid = std::get<1>(execution_policy);
    auto block = std::get<2>(execution_policy);

    uint64_t offset = 0;
    uint64_t seed = gpu_generator->current_seed();
    {
      std::lock_guard<std::mutex> lock(gpu_generator->mutex_);
      offset = gpu_generator->get_philox_offset(counter_offset);
    }

    distribution_elementwise_grid_stride_kernel_double<<<
        grid, block, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(elem_cnt, seed,
                                                                              offset, p, out_dptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("bernoulli_scalar")
    .SetCreateFn<BernoulliScalarGpuFloatKerenl>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("out", 0) == GetDataType<float>::value));

REGISTER_USER_KERNEL("bernoulli_scalar")
    .SetCreateFn<BernoulliScalarGpuDoubleKerenl>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("out", 0) == GetDataType<double>::value));

}  // namespace oneflow

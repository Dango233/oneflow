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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/to_contiguous_kernel.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"

namespace oneflow {

namespace {

std::unique_ptr<ep::primitive::BroadcastElementwiseUnary> NewPrimitive(
    user_op::KernelComputeContext* ctx) {
  const auto* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const auto* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  size_t max_ndim = std::max(in_desc->shape().size(), out_desc->shape().size());
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kIdentity, in_desc->data_type(),
      out_desc->data_type(), max_ndim);
}

template<DeviceType device_type, typename T>
class ToContiguousKernel final : public user_op::OpKernel {
 public:
  ToContiguousKernel() = default;
  ~ToContiguousKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const ShapeView& in_shape = in->shape_view();
    const ShapeView& out_shape = out->shape_view();
    CHECK_EQ(out_shape, in_shape);
    CHECK_EQ(out->data_type(), in->data_type());

    auto prim = NewPrimitive(ctx);
    CHECK(prim);
    prim->Launch(ctx->stream(), in_shape.size(), in_shape.data(), in->stride().data(), in->dptr(),
                 out_shape.size(), out_shape.data(), out->stride().data(), out->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TO_CONTIGUOUS_KERNEL(device_type, cpp_type, data_type) \
  REGISTER_USER_KERNEL("to_contiguous")                                 \
      .SetCreateFn<ToContiguousKernel<device_type, cpp_type>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type)        \
                       && (user_op::HobDataType("in", 0) == data_type));

#define REGISTER_TO_CONTIGUOUS_CPU_KERNEL(cpp_type, data_type) \
  REGISTER_TO_CONTIGUOUS_KERNEL(DeviceType::kCPU, cpp_type, data_type)
#define REGISTER_TO_CONTIGUOUS_CUDA_KERNEL(cpp_type, data_type) \
  REGISTER_TO_CONTIGUOUS_KERNEL(DeviceType::kCUDA, cpp_type, data_type)

#define REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CPU_TYPES \
  OF_PP_FOR_EACH_TUPLE(REGISTER_TO_CONTIGUOUS_CPU_KERNEL, TO_CONTIGUOUS_CPU_TYPES)

#define REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CUDA_TYPES       \
  OF_PP_FOR_EACH_TUPLE(REGISTER_TO_CONTIGUOUS_CUDA_KERNEL, \
                       TO_CONTIGUOUS_COMMON_TYPES TO_CONTIGUOUS_CUDA_SPECIAL_TYPE)

REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CPU_TYPES
#ifdef WITH_CUDA
REGISTER_TO_CONTIGUOUS_KERNEL_FOR_CUDA_TYPES
#endif

}  // namespace
}  // namespace oneflow

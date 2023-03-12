#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace {

template<typename SRC, typename DST>
struct SkipLoad {
    using LoadType = DST;
    SkipLoad(const SRC* src, const SRC* bias, const SRC* skip, int64_t row_size) : src(src), bias(bias), skip(skip), row_size(row_size) {}
    template<int N>
    __device__ void load(DST* dst, int64_t row, int64_t col) const {
        cuda::layer_norm::Pack<DST, N> src_pack;
        cuda::layer_norm::Pack<DST, N> bias_pack;
        cuda::layer_norm::Pack<DST, N> skip_pack;
        const int64_t offset = (row * row_size + col) / N;
        const int64_t bias_offset = col / N;
        src_pack.storage = *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(src) + offset);
        if (bias) {
            bias_pack.storage = 
                *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(bias) + bias_offset);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) { bias_pack.elem[i] = static_cast<SRC>(0.f); }
        }
        if (skip) {
            skip_pack.storage = 
                *(reinterpret_cast<const cuda::layer_norm::PackType<SRC, N>*>(skip) + offset);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) { skip_pack.elem[i] = static_cast<SRC>(0.f); }
        }
#pragma unroll
        for (int i = 0; i < N; ++i) {
            dst[i] = static_cast<DST>(src_pack.elem[i] + bias_pack.elem[i] + skip_pack.elem[i]);
        }
    }
    const SRC* src;
    const SRC* bias;
    const SRC* skip;
    int64_t row_size;
};

template<typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::layer_norm::Pack<DST, N> y_pack;
    cuda::layer_norm::Pack<DST, N> gamma_pack;
    cuda::layer_norm::Pack<DST, N> beta_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    if (do_scale) {
      gamma_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(gamma) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = static_cast<DST>(1.f); }
    }
    if (do_center) {
      beta_pack.storage =
          *(reinterpret_cast<const cuda::layer_norm::PackType<DST, N>*>(beta) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { beta_pack.elem[i] = static_cast<DST>(0.f); }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<cuda::layer_norm::PackType<DST, N>*>(y) + offset) = y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename T, bool do_scale, bool do_center>
void LaunchAddBiasResidualLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                        const int64_t norm_size, const double epsilon, const T* x_ptr,
                        const T* gamma_ptr, const T* beta_ptr, const T* bias_ptr, 
                        const T* skip_ptr, const double alpha, T* y_ptr,
                        user_op::Tensor* mean, user_op::Tensor* inv_variance){
    constexpr int32_t block_size = 128;
    unsigned int nb_element = norm_size*num_instances;
    unsigned int grid_size = (nb_element + block_size - 1) / block_size;
    using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;
    SkipLoad<T, T> load(x_ptr, bias_ptr, skip_ptr, norm_size);
    AffineStore<ComputeType, T, do_scale, do_center> store(y_ptr, norm_size, gamma_ptr, beta_ptr);
    cuda::layer_norm::DispatchLayerNorm<decltype(load), decltype(store), ComputeType>(
      stream->As<ep::CudaStream>()->cuda_stream(), load, store, num_instances, norm_size, epsilon,
      mean->mut_dptr<ComputeType>(), inv_variance->mut_dptr<ComputeType>());
}

template<typename T>
void DispatchAddBiasResidualLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                        const int64_t norm_size, const double epsilon, const T* x_ptr,
                        const T* gamma_ptr, const T* beta_ptr, const T* bias_ptr, 
                        const T* skip_ptr, const double alpha, T* y_ptr,
                        user_op::Tensor* mean, user_op::Tensor* inv_variance){

#define LAUNCH_GPU_KERNEL(has_gamma, has_beta) \
    LaunchAddBiasResidualLayerNormForwardGpu<T, has_gamma, has_beta>(stream, num_instances, norm_size, \
        epsilon, x_ptr, gamma_ptr, beta_ptr, bias_ptr, skip_ptr, alpha, y_ptr, mean, inv_variance);

    if(gamma_ptr != nullptr && beta_ptr != nullptr){
        LAUNCH_GPU_KERNEL(true, true);
    } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
        LAUNCH_GPU_KERNEL(false, true);
    } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
        LAUNCH_GPU_KERNEL(true, false);
    } else {
        LAUNCH_GPU_KERNEL(false, false);
    }

#undef LAUNCH_GPU_KERNEL
}

template<typename T>
class SkipLayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  SkipLayerNormGpuKernel() = default;
  ~SkipLayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain x and check its shape
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const ShapeView &x_shape = x->shape_view();
    CHECK_GT(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();

#define GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(tensor) \
    if (ctx->has_input(#tensor, 0)) { \
        const user_op::Tensor* tensor = ctx->Tensor4ArgNameAndIndex(#tensor, 0); \
        tensor##_shape = tensor->shape_view(); \
        tensor##_ptr = tensor->dptr<T>(); \
        CHECK_EQ(tensor##_shape.NumAxes(), 1) \
            << "number of axes of \'" << #tensor << "\' should have be greater than 1, yet get " \
            << tensor##_shape.NumAxes(); \
        CHECK_EQ(tensor##_shape.At(0), x_shape.At(x_shape.NumAxes() - 1)) \
            << "dimension 1 of \'" << #tensor << "\'(" << tensor##_shape.At(0) \
            << ") is not consistant with the last dimension of \'x\'(" \
            << x_shape.At(x_shape.NumAxes() - 1) << ")"; \
    }

    // obtain gamma and check its shape
    const T* gamma_ptr = nullptr;
    ShapeView gamma_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(gamma);
    
    // obtain beta and check its shape
    const T* beta_ptr = nullptr;
    ShapeView beta_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(beta);

    // obtain bias and check its shape
    const T* bias_ptr = nullptr;
    ShapeView bias_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(bias);

#undef GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK

    // obtain residual and check its shape
    const T* skip_ptr = nullptr;
    ShapeView skip_shape;
    if (ctx->has_input("skip", 0)) {
        const user_op::Tensor* skip = ctx->Tensor4ArgNameAndIndex("skip", 0);
        skip_shape = skip->shape_view();
        skip_ptr = skip->dptr<T>();
        CHECK_EQ(skip_shape, x_shape);
    }

    // obtain epsilon and check its value
    const double epsilon = ctx->Attr<double>("epsilon");
    const double alpha = ctx->Attr<double>("alpha");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);

    // obtain output tensors
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const ShapeView &y_shape = y->shape_view();
    const ShapeView &mean_shape = mean->shape_view();
    const ShapeView &inv_variance_shape = inv_variance->shape_view();

    // calculate number of instances and norm size
    const int64_t num_instances = mean->shape_view().elem_cnt();
    const int64_t norm_size = x->shape_view().elem_cnt() / num_instances;

    // dispatch kernel
    DispatchAddBiasResidualLayerNormForwardGpu<T>(ctx->stream(), num_instances, norm_size, 
        epsilon, x->dptr<T>(), gamma_ptr, beta_ptr, bias_ptr, skip_ptr, alpha, 
        y->mut_dptr<T>(), mean, inv_variance);
  }
};

} // namespace

#define REGISTER_SKIP_LAYER_NORM_CUDA_KERNEL(dtype) \
    REGISTER_USER_KERNEL("skip_layer_norm") \
        .SetCreateFn<SkipLayerNormGpuKernel<dtype>>() \
        .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_SKIP_LAYER_NORM_CUDA_KERNEL(float)
REGISTER_SKIP_LAYER_NORM_CUDA_KERNEL(double)
REGISTER_SKIP_LAYER_NORM_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_SKIP_LAYER_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

} // namespace oneflow
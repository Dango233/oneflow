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
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kGelu, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const {
    return static_cast<Src>(0.5) * src
           * (static_cast<Src>(1.0) + erf(static_cast<Src>(M_SQRT1_2) * src));
  }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, float, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC float operator()(float src) const { return tanhf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, double, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC double operator()(double src) const { return tanh(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTanh, half, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC half operator()(half src) const { return __float2half(tanhf(__half2float(src))); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kRsqrt, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return rsqrt(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAcos, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return acos(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAcosh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return acosh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAsin, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return asin(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAsinh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return asinh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAtan, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return atan(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kAtanh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return atanh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCeil, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return ceil(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCos, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return cos(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kCosh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return cosh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kErf, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return erf(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kErfc, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return erfc(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kExp, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return exp(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kExpm1, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return expm1(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kFloor, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return floor(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kLgamma, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return lgamma(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kLog, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return log(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kLog2, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return log2(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kLog1p, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return log1p(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kRint, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return rint(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kRound, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return nearbyint(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kSin, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return sin(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kSinh, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return sinh(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kSqrt, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return sqrt(src); }
};

template<typename Dst, typename Src>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kTan, Dst, Src> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC Dst operator()(Src src) const { return tan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(half src) const { return isinf(__half2float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return isinf(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, half> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(half src) const { return isnan(__half2float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, float> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(float src) const { return isnan(src); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, double> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(double src) const { return isnan(src); }
};

#define SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(op)                          \
  template<>                                                                  \
  struct UnaryFunctor<DeviceType::kCUDA, op, half, half> {                    \
    UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                              \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;          \
    OF_DEVICE_FUNC half operator()(half src) const {                          \
      return __float2half(float_functor(__half2float(src)));                  \
    }                                                                         \
  };

SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSoftPlus);

SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAbs);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAcos);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAcosh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAsin);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAsinh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAtan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kAtanh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCeil);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCos);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kCosh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kErf);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kErfc);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kExp);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kExpm1);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kFloor);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLgamma);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog2);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLog1p);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kLogSigmoid);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kNegative);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kReciprocal);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kReciprocalNoNan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRint);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRound);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kRsqrt);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSigmoid);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSign);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSin);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSinh);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSqrt);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kSquare);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kTan);
SPECIALIZATION_PSEUDO_HALF_UNARY_FUNCTOR(UnaryOp::kNotEqualZero);

/*********nv_bfloat16_kernel*******/

#if CUDA_VERSION >= 11000

#define SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(op)                      \
  template<>                                                                  \
  struct UnaryFunctor<DeviceType::kCUDA, op, nv_bfloat16, nv_bfloat16> {      \
    UnaryFunctor(Scalar attr0, Scalar attr1) : float_functor(attr0, attr1) {} \
                                                                              \
    UnaryFunctor<DeviceType::kCUDA, op, float, float> float_functor;          \
    OF_DEVICE_FUNC nv_bfloat16 operator()(nv_bfloat16 src) const {            \
      return __float2bfloat16(float_functor(__bfloat162float(src)));          \
    }                                                                         \
  };

SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kElu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kGelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSwish);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kHardTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLeakyRelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kMish);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSelu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSilu);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftShrink);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftSign);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSoftPlus);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kThreshold);

SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAbs);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcos);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAcosh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsin);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAsinh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kAtanh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCeil);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCos);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kCosh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErf);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kErfc);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExp);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kExpm1);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kFloor);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLgamma);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog2);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLog1p);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kLogSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kNegative);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kReciprocal);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kReciprocalNoNan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRint);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRound);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kRsqrt);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSigmoid);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSign);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSin);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSinh);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSqrt);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kSquare);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kTan);
SPECIALIZATION_PSEUDO_BFLOAT16_UNARY_FUNCTOR(UnaryOp::kNotEqualZero);

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsInf, bool, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(nv_bfloat16 src) const { return isinf(__bfloat162float(src)); }
};

template<>
struct UnaryFunctor<DeviceType::kCUDA, UnaryOp::kIsNan, bool, nv_bfloat16> {
  OF_DEVICE_FUNC UnaryFunctor(Scalar attr0, Scalar attr1) {}

  OF_DEVICE_FUNC bool operator()(nv_bfloat16 src) const { return isnan(__bfloat162float(src)); }
};

#endif

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

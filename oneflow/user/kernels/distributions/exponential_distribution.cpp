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
#include "oneflow/user/kernels/distributions/exponential_distribution.h"

namespace oneflow {

template<typename T>
void ExponentialDistribution<DeviceType::kCPU, T>::operator()(
    ep::Stream* stream, const int64_t elem_cnt, T* dptr,
    const std::shared_ptr<one::Generator>& generator) const {
  CHECK_GE(elem_cnt, 0);
  auto gen = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
  one::pytorch_mt19937_engine& engine = gen->torch_engine();
  for (int64_t i = 0; i < elem_cnt; ++i) {
    uint64_t rand_unit = random64(engine);
    T random_val = uniform_real(rand_unit, 0.0, 1.0);
    dptr[i] = static_cast<T>(-1.0) / lambd_ * std::log(static_cast<T>(1.0) - random_val);
  }
}

#define INITIATE_CPU_UNIFORM_DISTRIBUTION(T, typeproto)                   \
  template void ExponentialDistribution<DeviceType::kCPU, T>::operator()( \
      ep::Stream* stream, const int64_t elem_cnt, T* dptr,                \
      const std::shared_ptr<one::Generator>& generator) const;

OF_PP_FOR_EACH_TUPLE(INITIATE_CPU_UNIFORM_DISTRIBUTION, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow

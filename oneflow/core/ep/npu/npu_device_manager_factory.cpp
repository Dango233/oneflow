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
#ifdef WITH_NPU
#include "oneflow/core/ep/include/device_manager_factory.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/npu/npu_device_manager.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
namespace oneflow {

namespace ep {

namespace {

class NpuDeviceManagerFactory : public DeviceManagerFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NpuDeviceManagerFactory);
  NpuDeviceManagerFactory() = default;
  ~NpuDeviceManagerFactory() override = default;

  std::unique_ptr<DeviceManager> NewDeviceManager(DeviceManagerRegistry* registry) override {
    return std::make_unique<NpuDeviceManager>(registry);
  }

  DeviceType device_type() const override { return DeviceType::kNPU; }

  std::string device_type_name() const override { return "npu"; }
};

COMMAND(DeviceManagerRegistry::RegisterDeviceManagerFactory(
    std::make_unique<NpuDeviceManagerFactory>()))

}  // namespace

}  // namespace ep

}  // namespace oneflow
#endif
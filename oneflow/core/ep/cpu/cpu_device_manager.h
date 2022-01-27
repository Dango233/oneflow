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
#ifndef ONEFLOW_CORE_EP_CPU_CPU_DEVICE_MANAGER_H_
#define ONEFLOW_CORE_EP_CPU_CPU_DEVICE_MANAGER_H_

#include "oneflow/core/ep/include/device_manager.h"

namespace oneflow {

namespace ep {

class CpuDeviceManager : public DeviceManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceManager);
  CpuDeviceManager(DeviceManagerRegistry* registry);
  ~CpuDeviceManager() override;

  DeviceManagerRegistry* registry() const override;
  std::shared_ptr<Device> GetDevice(size_t device_index) override;
  size_t GetDeviceCount(size_t primary_device_index) override;
  size_t GetDeviceCount() override;
  size_t GetActiveDeviceIndex() override;
  void SetActiveDeviceByIndex(size_t device_index) override;
  void SetDeviceNumThreads(size_t num_threads);\

 private:
  size_t device_num_threads_;
  std::mutex device_mutex_;
  std::shared_ptr<Device> device_;
  DeviceManagerRegistry* registry_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_DEVICE_MANAGER_H_

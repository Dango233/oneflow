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
#include "oneflow/core/vm/stream_wait_instruction_type.h"
#include "oneflow/core/vm/ep_event.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/ep/cuda/cuda_event.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/core/vm/ep_device_context.h"

namespace oneflow {
namespace vm {

bool StreamWaitInstructionType::Prescheduleable(const Stream* src, const Stream* dst) const {
  return &src->thread_ctx() == &dst->thread_ctx();
}

void StreamWaitInstructionType::InitInstructionStatus(Instruction* instruction) const {
  auto* phy_instr_operand = instruction->phy_instr_operand().get();
  auto* operand = dynamic_cast<StreamWaitPhyInstrOperand*>(phy_instr_operand);
  auto* stream = operand->mut_from_vm_stream();
  instruction->stream_type().InitInstructionStatus(*stream, instruction->mut_status_buffer());
  auto* ep_device_ctx = dynamic_cast<EpDeviceCtx*>(stream->device_ctx().get());
  auto* ep_event_provider = ep_device_ctx->ep_event_provider();
  const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
  operand->mut_ep_event() = ep_event;
}

void StreamWaitInstructionType::DeleteInstructionStatus(Instruction* instruction) const {
  auto* phy_instr_operand = instruction->phy_instr_operand().get();
  auto* operand = dynamic_cast<StreamWaitPhyInstrOperand*>(phy_instr_operand);
  auto* stream = operand->mut_from_vm_stream();
  instruction->stream_type().DeleteInstructionStatus(*stream, instruction->mut_status_buffer());
  operand->mut_ep_event().reset();
}

void StreamWaitInstructionType::Compute(vm::Instruction* instruction) const {
  auto* operand = dynamic_cast<StreamWaitPhyInstrOperand*>(instruction->phy_instr_operand().get());
  const auto& ep_event = operand->mut_ep_event();
  {
    // Record event.
    auto* from_device_ctx = operand->mut_from_vm_stream()->device_ctx().get();
    auto* from_ep_device_ctx = CHECK_NOTNULL(dynamic_cast<vm::EpDeviceCtx*>(from_device_ctx));
    auto* from_stream = from_ep_device_ctx->stream();
    from_stream->RecordEvent(ep_event->mut_event());
  }
  {
    // Wait event.
    auto* to_device_ctx = instruction->mut_stream()->device_ctx().get();
    auto* to_ep_device_ctx = CHECK_NOTNULL(dynamic_cast<vm::EpDeviceCtx*>(to_device_ctx));
    auto* to_ep_stream = to_ep_device_ctx->stream();
    CHECK_EQ(ep_event->mut_device(), to_ep_stream->device())
        << "only support waiting events from same device";
    ep_event->mut_device()->SetAsActiveDevice();
#ifdef WITH_CUDA

    auto* ep_cuda_event = CHECK_NOTNULL(dynamic_cast<ep::CudaEvent*>(ep_event->mut_event()));
    auto* ep_cuda_stream = CHECK_NOTNULL(dynamic_cast<ep::CudaStream*>(to_ep_stream));

    OF_CUDA_CHECK(cudaStreamWaitEvent(ep_cuda_stream->cuda_stream(), ep_cuda_event->cuda_event(),
                                      cudaEventWaitDefault));
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  }
}

}  // namespace vm
}  // namespace oneflow

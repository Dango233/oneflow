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
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/JITEngine.h"
#include "oneflow/ir/include/OneFlow/Extension.h"
#include <glog/logging.h>

namespace oneflow {
SharedLibs* MutSharedLibPaths() {
  static SharedLibs libs = {};
  return &libs;
}
const SharedLibs* SharedLibPaths() { return MutSharedLibPaths(); }
}  // namespace oneflow


JIT_Engine::JIT_Engine(mlir::ModuleOp module) {
  llvm::SmallVector<llvm::StringRef, 4> ext_libs(
      {oneflow::SharedLibPaths()->begin(), oneflow::SharedLibPaths()->end()});
  mlir::ExecutionEngineOptions jitOptions;
  jitOptions.transformer = {};
  jitOptions.jitCodeGenOptLevel = llvm::None;
  jitOptions.sharedLibPaths = ext_libs;

  auto jit_or_error = mlir::ExecutionEngine::create(module, jitOptions);
  CHECK(!!jit_or_error) << "failed to create JIT exe engine, "
                        << llvm::toString((jit_or_error).takeError());
  jit_or_error->swap(engine_);
}

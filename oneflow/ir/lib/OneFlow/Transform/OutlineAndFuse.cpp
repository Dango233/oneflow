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
#include "OneFlow/OKL/OKLDialect.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <string>

using namespace mlir;

namespace mlir {
namespace oneflow {

namespace {

class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class LowerToOKLPass : public LowerToOKLPassBase<LowerToOKLPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<okl::OKLDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateLowerToOKLPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
class ConvertOFKLCalleeToLLVMPass
    : public ConvertOFKLCalleeToLLVMPassBase<ConvertOFKLCalleeToLLVMPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateConvertOFKLCalleeToLLVMPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class WrapOps2KernelLaunchPass : public WrapOps2KernelLaunchPassBase<WrapOps2KernelLaunchPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<oneflow::OneFlowDialect>();
  }

  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateWrapOps2KernelLaunchPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

class FuseIntoExistingOpPass : public FuseIntoExistingOpPassBase<FuseIntoExistingOpPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    populateFuserForExistingOp(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

std::unique_ptr<Pass> createWrapOps2KernelLaunchPass() {
  return std::make_unique<WrapOps2KernelLaunchPass>();
}

std::unique_ptr<mlir::Pass> createConvertOFKLCalleeToLLVMPass() {
  return std::make_unique<ConvertOFKLCalleeToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createLowerToOKLPass() { return std::make_unique<LowerToOKLPass>(); }

std::unique_ptr<Pass> createFuseIntoExistingOpPass() {
  return std::make_unique<FuseIntoExistingOpPass>();
}

}  // namespace oneflow
}  // namespace mlir

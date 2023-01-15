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
#ifndef ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_
#define ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

DEFINE_THREAD_LOCAL_ENV_BOOL(MULTI_IN, false);

class JobCompleter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobCompleter);
  JobCompleter() = default;
  ~JobCompleter() = default;

  Maybe<void> Complete(Job* job) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_JOB_COMPLETER_H_

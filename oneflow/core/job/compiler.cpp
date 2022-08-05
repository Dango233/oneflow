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
#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/intra_job_mem_sharing_util.h"
#include "oneflow/core/job/plan_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_completer.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/time_util.h"

namespace oneflow {

void CreateOpAttributeRef(Plan* plan, int64_t job_id, TaskProto* task_proto) {
  auto* job_id2op_attribute_ref_table = plan->mutable_job_id2op_attribute_ref_table();
  CHECK(task_proto->exec_sequence().exec_node_size() == 1);
  auto* exec_node = task_proto->mutable_exec_sequence()->mutable_exec_node(0);
  CHECK(exec_node->kernel_conf().has_op_attribute());
  const std::string op_name = exec_node->kernel_conf().op_attribute().op_conf().name();
  auto* op_name2op_attribute =
      (*job_id2op_attribute_ref_table)[job_id].mutable_op_name2op_attribute();
  auto find_it = op_name2op_attribute->find(op_name);
  if (find_it == op_name2op_attribute->end()) {
    op_name2op_attribute->insert(
        {op_name, task_proto->exec_sequence().exec_node(0).kernel_conf().op_attribute()});
  }
  auto* kernel_conf =
      task_proto->mutable_exec_sequence()->mutable_exec_node(0)->mutable_kernel_conf();
  kernel_conf->set_op_attribute_ref(op_name);
  // NOTE(levi): memory of op_attribute_ is released here.
  kernel_conf->set_allocated_op_attribute(nullptr);
}

void Compiler::Compile(Job* job, Plan* plan) const {
  const std::string job_name = job->job_conf().job_name();
  auto tc = std::make_unique<TimeCounter<std::chrono::milliseconds>>(true);
  // Step1: new Singleton<OpGraph> and set log configs.
  Singleton<OpGraph>::New(*job);
  tc->Count("Graph name: " + job_name + " NewOpGraph", 1);
  const JobDesc& job_desc = GlobalJobDesc();
  if (Singleton<ResourceDesc, ForSession>::Get()->enable_debug_mode()
      || Singleton<ResourceDesc, ForSession>::Get()->enable_dry_run()) {
    TeePersistentLogStream::Create(StrCat("optimized_job", job_desc.job_id()))->Write(*job);
    Singleton<OpGraph>::Get()->ToDotWithFilePath(
        "optimized_dlnet_" + std::to_string(job_desc.job_id()) + "_op_graph.dot");
  }
  tc->Count("Graph name: " + job_name + " LogOptimizedJob", 1);

  // Step2: build task_gph.
  // TODO(levi): we can rewrite this part of code in visitor pattern.
  auto task_gph =
      std::make_unique<TaskGraph>(job->job_conf().enable_straighten_algorithm_in_task_graph());
  tc->Count("Graph name: " + job_name + " NewTaskGraph", 1);
  using std::placeholders::_1;
  task_gph->ForEachNode(std::bind(&TaskNode::ProduceAllRegstsAndBindEdges, _1));
  tc->Count("Graph name: " + job_name + " ProduceAllRegstsAndBindEdges", 1);
  task_gph->ForEachNode(std::bind(&TaskNode::ConsumeAllRegsts, _1));
  tc->Count("Graph name: " + job_name + " ConsumeAllRegsts", 1);
  task_gph->ForEachNode(std::bind(&TaskNode::PinConsumedRegst, _1));
  tc->Count("Graph name: " + job_name + " PinConsumedRegst", 1);
  task_gph->TopoForEachNode(&TaskNode::Build);
  tc->Count("Graph name: " + job_name + " TaskNode::Build", 1);
  task_gph->RemoveEmptyRegsts();
  tc->Count("Graph name: " + job_name + " RemoveEmptyRegsts", 1);
  task_gph->MergeChainAndAddOrderingCtrlEdgeInSameChain();
  tc->Count("Graph name: " + job_name + " MergeChainAndAddOrderingCtrlEdgeInSameChain", 1);
  auto IsReachable = Singleton<OpGraph>::Get()->MakePredicatorIsOpNameDataOrCtrlReachable();
  if (job_desc.enable_inplace()) { task_gph->EnableInplaceMemSharing(IsReachable); }
  tc->Count("Graph name: " + job_name + " EnableInplaceMemSharing", 1);
  task_gph->TopoForEachNode(&TaskNode::InferTimeShapeIfMeaningful);
  tc->Count("Graph name: " + job_name + " InferTimeShapeIfMeaningful", 1);
  task_gph->ForEachEdge([&](TaskEdge* task_edge) { task_edge->CheckRegstLbiValid(); });
  tc->Count("Graph name: " + job_name + " CheckRegstLbiValid", 1);

  // Step3: put infomation from task_gph into plan.
  const int64_t node_num = task_gph->node_num();
  const int64_t cpu_num = std::thread::hardware_concurrency();
  const int64_t thread_pool_size = std::min(node_num, cpu_num);
  BlockingCounter counter(node_num);
  std::mutex mtx;
  ThreadPool thread_pool(thread_pool_size);
  task_gph->ForEachNode([&](TaskNode* task_node) {
    thread_pool.AddWork([task_node, plan, &job_desc, &counter, &mtx]() {
      if (!task_node->IsMeaningLess()) {
        TaskProto task_proto;
        task_node->ToProto(&task_proto);
        {
          std::unique_lock<std::mutex> guard(mtx);
          if (task_node->GetTaskType() == kNormalForward || task_node->GetTaskType() == kRepeat
              || task_node->GetTaskType() == kAcc) {
            CreateOpAttributeRef(plan, job_desc.job_id(), &task_proto);
          }
          plan->mutable_task()->Add(std::move(task_proto));
        }  // guard(mtx)
      }
      counter.Decrease();
    } /* thread_pool.AddWork */);
  } /* task_gph->ForEachNode */);
  counter.WaitForeverUntilCntEqualZero();
  tc->Count("Graph name: " + job_name + " AddTaskIntoPlan", 1);
  // NOTE(levi): release task_gph here to decrise memory peak.
  task_gph.reset();
  tc->Count("Graph name: " + job_name + " ReleaseTaskGraph", 1);

  // Step4: post-process for plan and delete Singleton<OpGraph>.
  auto* job_id2job_conf = plan->mutable_job_confs()->mutable_job_id2job_conf();
  (*job_id2job_conf)[GlobalJobDesc().job_id()] = GlobalJobDesc().job_conf();
  // NOTE(chengcheng): infer mem blob id & set inplace & add ctrl
  IntraJobMemSharingUtil::InferMemBlockId4MemReusedRegst(plan, IsReachable);
  tc->Count("Graph name: " + job_name + " InferMemBlockId4MemReusedRegst", 1);
  PlanUtil::SetUniqueMemBlockId4UnreusedMemRegst(plan);
  tc->Count("Graph name: " + job_name + " SetUniqueMemBlockId4UnreusedMemRegst", 1);
  Singleton<OpGraph>::Delete();
  tc->Count("Graph name: " + job_name + " ReleaseOpGraph", 1);
}

}  // namespace oneflow

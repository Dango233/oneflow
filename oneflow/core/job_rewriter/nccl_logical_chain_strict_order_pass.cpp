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
#ifdef WITH_CUDA
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

namespace {

class NcclLogicalChainStrictOrderPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalChainStrictOrderPass);
  NcclLogicalChainStrictOrderPass() = default;
  ~NcclLogicalChainStrictOrderPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

bool IsAccOrPackOpNode(const OpNode* node) {
  const auto& op_conf = node->op().op_conf();
  return op_conf.has_user_conf()
         && (op_conf.user_conf().op_type_name() == "acc"
             || op_conf.user_conf().op_type_name() == "pack");
}

Maybe<void> NcclLogicalChainStrictOrderPass::Apply(const OpGraph& op_graph,
                                                   JobBuilder* job_builder) const {
  HashMap<int64_t, const OpNode*> nccl_chain_id2last_node;
  HashMap<std::string, OperatorConf> mut_op_name2conf;
  HashMap<std::string, const OpNode*> placement2last_normal_node;
  HashMap<std::string, const OpNode*> placement2first_after_acc_node;
  auto IsReachable = op_graph.MakePredicatorIsOpNameDataOrCtrlReachable();

  bool need_insert_ctrl_before_acc = false;
  const int64_t acc_num = job_builder->job().job_conf().num_gradient_accumulation_steps();
  if (acc_num > 1) { need_insert_ctrl_before_acc = true; }

  // TODO(chengcheng, yipeng) better order for memory strighten.
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    if (!node->op().op_conf().has_logical_chain_id()) { return; }
    const int64_t logical_chain_id = node->op().op_conf().logical_chain_id();

    // add ctrl edge for strict order
    auto it = nccl_chain_id2last_node.find(logical_chain_id);
    if (it == nccl_chain_id2last_node.end()) {
      nccl_chain_id2last_node.emplace(logical_chain_id, node);
    } else {
      const std::string& this_op_name = node->op().op_name();
      const std::string& prev_op_name = it->second->op().op_name();
      if (!IsReachable(prev_op_name, this_op_name)) {
        CHECK(mut_op_name2conf.emplace(this_op_name, node->op().op_conf()).second);
        mut_op_name2conf.at(this_op_name).add_ctrl_in_op_name(prev_op_name);
        LOG(INFO) << " ccdebuglog: logical_chain_id: " << logical_chain_id << " insert ctrl [ "
                  << prev_op_name << " ] -> [ " << this_op_name << " ]";
      }
      it->second = node;
      LOG(INFO) << " ccdebuglog: update last op of logical_chain_id : " << logical_chain_id
                << " is " << it->second->op().op_name() << " now. ";
    }

    if (need_insert_ctrl_before_acc) {
      const int64_t time_shape_cnt =
          CHECK_JUST(node->op().GetInputOutputFastestTimeShape())->elem_cnt();
      CHECK(time_shape_cnt == acc_num || time_shape_cnt == 1)
          << " invalid time shape count = " << time_shape_cnt << " which should be : [ " << acc_num
          << " , 1 ]";
      std::string placement_key = GenParallelConfKey(node->parallel_desc().parallel_conf());
      if (time_shape_cnt == acc_num) {
        // fw/bw chain
        placement2last_normal_node[placement_key] = node;  // create or update
        // placement2last_normal_node.emplace(placement_key, node);
        LOG(INFO) << " ccdebuglog v2: update last normal op of placement: " << placement_key
                  << " is " << node->op().op_name() << " now. "
                  << " \n\n and placement2last_normal_node = "
                  << placement2last_normal_node.at(placement_key)->op().op_name();

      } else {
        // acc chain
        if (placement2first_after_acc_node.find(placement_key)
            == placement2first_after_acc_node.end()) {
          CHECK(placement2first_after_acc_node.emplace(placement_key, node).second);
          LOG(INFO) << " ccdebuglog v2: first after acc op of placement: " << placement_key
                    << " is " << node->op().op_name() << " now. ";
        }
      }
    }
  });

  if (need_insert_ctrl_before_acc) {
    for (const auto& pair : placement2last_normal_node) {
      if (placement2first_after_acc_node.find(pair.first) == placement2first_after_acc_node.end()) {
        continue;
      }
      const OpNode* last_bw_node = pair.second;
      const OpNode* first_after_acc_node = JUST(MapAt(placement2first_after_acc_node, pair.first));
      const std::string& last_bw_op_name = last_bw_node->op().op_name();
      const std::string& first_after_acc_op_name = first_after_acc_node->op().op_name();

      LOG(INFO) << " ccdebuglog v2: last normal op of placement: " << pair.first
                << " is : " << last_bw_op_name;

      CHECK_OR_RETURN(!IsReachable(first_after_acc_op_name, last_bw_op_name))
          << Error::RuntimeError()
          << " Error! Cycle control edge from first acc chain op: " << first_after_acc_op_name
          << " to last bw chain sink op: " << last_bw_op_name;

      const auto& bw_sink_obns = last_bw_node->op().output_bns();
      CHECK_OR_RETURN(!bw_sink_obns.empty());
      const std::string bw_sink_lbn =
          GenLogicalBlobName(last_bw_node->op().BnInOp2Lbi(bw_sink_obns.Get(0)));
      VLOG(3) << " bw_sink_lbn : " << bw_sink_lbn;

      user_op::UserOpConfWrapper cast_to_tick_op =
          user_op::UserOpConfWrapperBuilder("Sys-LastNcclChainSink-CastToTick-" + NewUniqueId())
              .OpTypeName("cast_to_tick")
              .Input("in", bw_sink_lbn)
              .Output("out")
              .ScopeSymbolId(last_bw_node->op().op_conf().scope_symbol_id())
              .Build();

      JUST(job_builder->AddOp(last_bw_node->parallel_desc().parallel_conf(),
                              cast_to_tick_op.op_conf()));

      std::string acc_tick_output_lbn = cast_to_tick_op.output("out", 0);
      if (!IsAccOrPackOpNode(last_bw_node)) {
        // NOTE(chengcheng): Acc Op can be merged in fw/bw chain, if the last op is acc op,
        //  there is no need and CANNOT insert acc tick op.

        OperatorConf sink_acc_tick_conf;
        sink_acc_tick_conf.set_name(std::string("Sys-LastNcclChainSink-AccTick_") + NewUniqueId());
        sink_acc_tick_conf.set_scope_symbol_id(last_bw_node->op().op_conf().scope_symbol_id());
        auto* acc_conf = sink_acc_tick_conf.mutable_acc_tick_conf();
        acc_conf->set_one(cast_to_tick_op.output("out", 0));
        acc_conf->set_acc("acc");
        acc_conf->set_max_acc_num(acc_num);

        acc_tick_output_lbn = GenLogicalBlobName(sink_acc_tick_conf.name(), "acc");

        VLOG(3) << " insert acc tick op : " << sink_acc_tick_conf.name()
                << " of last op in fw/bw chain.";

        JUST(job_builder->AddOp(last_bw_node->parallel_desc().parallel_conf(), sink_acc_tick_conf));
      }

      OperatorConf sink_final_tick_conf;
      sink_final_tick_conf.set_name(std::string("Sys-LastNcclChainSink-FinalTick-DeviceTick_")
                                    + NewUniqueId());
      sink_final_tick_conf.set_scope_symbol_id(last_bw_node->op().op_conf().scope_symbol_id());
      auto* tick_conf = sink_final_tick_conf.mutable_device_tick_conf();
      tick_conf->add_tick(acc_tick_output_lbn);
      tick_conf->set_out("out");

      JUST(job_builder->AddOp(last_bw_node->parallel_desc().parallel_conf(), sink_final_tick_conf));

      if (mut_op_name2conf.find(first_after_acc_op_name) == mut_op_name2conf.end()) {
        mut_op_name2conf.emplace(first_after_acc_op_name, first_after_acc_node->op().op_conf());
      }
      JUST(MapAt(mut_op_name2conf, first_after_acc_op_name))
          .add_ctrl_in_op_name(sink_final_tick_conf.name());

      VLOG(2) << " In: " << pair.first << " , insert ctrl edge from: [ " << last_bw_op_name
              << " ] to: [ " << first_after_acc_op_name << " ]";
    }
  }

  for (const auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("NcclLogicalChainStrictOrderPass", NcclLogicalChainStrictOrderPass);

}  // namespace oneflow

#endif  // WITH_CUDA

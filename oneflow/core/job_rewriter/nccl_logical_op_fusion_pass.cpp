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
#include "oneflow/core/job/nd_sbp_util.h"
#ifdef WITH_CUDA
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job_rewriter/calculation_pass.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

namespace {

class NcclLogicalOpFusionPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclLogicalOpFusionPass);
  NcclLogicalOpFusionPass() = default;
  ~NcclLogicalOpFusionPass() = default;

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

const std::string kNcclLogicalFusionOpNamePrefix = "Sys-NCCL-Logical-Fusion";

bool IsNcclLogicalOpNode(const OpNode* node) {
  if (node->op().op_conf().has_user_conf()) {
    const std::string& user_type_name = node->op().op_conf().user_conf().op_type_name();
    if (user_type_name == "_nccl_logical_all_reduce"
        || user_type_name == "_nccl_logical_reduce_scatter"
        || user_type_name == "_nccl_logical_reduce_scatter_noncontinuous"
        || user_type_name == "_nccl_logical_all_gather"
        || user_type_name == "_nccl_logical_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_s2s"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_reduce"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather"
        || user_type_name == "_nccl_logical_2D_same_dim0_all_gather_noncontinuous"
        || user_type_name == "_nccl_logical_2D_same_dim0_all2all"
        || user_type_name == "_nccl_logical_2D_same_dim1_all_reduce"
        || user_type_name == "_nccl_logical_send_recv") {
      return true;
    }
  }
  return false;
}

Maybe<void> ReplaceNcclOpsWithFusionOp(std::vector<OperatorConf>* nccl_fusion_ops,
                                       std::vector<ParallelConf>* nccl_fusion_op_parallel_confs,
                                       std::unordered_set<std::string>* del_ops,
                                       HashMap<std::string, OperatorConf>* mut_op_name2conf,
                                       const std::vector<const OpNode*>& nccl_ops) {
  if (nccl_ops.size() <= 1) { return Maybe<void>::Ok(); }
  const int32_t nccl_size = nccl_ops.size();
  const OpNode* first_nccl = nccl_ops.front();
  const ParallelDesc& seed_placement = first_nccl->parallel_desc();
  const int64_t scope_symbol_id = first_nccl->op().op_conf().scope_symbol_id();
  std::vector<std::string> src_nd_sbp_str_list;
  std::vector<std::string> dst_nd_sbp_str_list;
  auto& fusion_builder = user_op::UserOpConfWrapperBuilder("Sys-NCCL-fusion-" + NewUniqueId())
                             .Op("_nccl_logical_fusion");
  for (const OpNode* nccl_op : nccl_ops) {
    fusion_builder = fusion_builder.Input(
        "in", GenLogicalBlobName(nccl_op->op().BnInOp2Lbi(nccl_op->op().SoleIbn())));
    src_nd_sbp_str_list.push_back(
        NdSbpToLongString(nccl_op->NdSbp4BnInOp(nccl_op->op().SoleIbn())));
    dst_nd_sbp_str_list.push_back(
        NdSbpToLongString(nccl_op->NdSbp4BnInOp(nccl_op->op().SoleObn())));
    // 1. update del op
    VLOG(3) << " Del op: " << nccl_op->op().op_name();
    del_ops->insert(nccl_op->op().op_name());
  }

  auto fusion_nccl_op =
      fusion_builder.Output("out", nccl_size)
          .Attr<std::vector<std::string>>("src_nd_sbp_str_list", src_nd_sbp_str_list)
          .Attr<std::vector<std::string>>("dst_nd_sbp_str_list", dst_nd_sbp_str_list)
          .ScopeSymbolId(scope_symbol_id)
          .Build();

  // 2. update fusion op
  VLOG(3) << " Add fusion op : " << fusion_nccl_op.op_conf().DebugString()
          << " \n with placement: " << seed_placement.parallel_conf().DebugString();
  nccl_fusion_ops->push_back(fusion_nccl_op.op_conf());
  nccl_fusion_op_parallel_confs->push_back(seed_placement.parallel_conf());

  for (int32_t i = 0; i < nccl_size; ++i) {
    std::string output_lbn = fusion_nccl_op.output("out", i);
    std::string input_lbn = fusion_nccl_op.input("in", i);
    const OpEdge* origin_edge = nccl_ops.at(i)->SoleOutEdge();
    const OpNode* origin_consumer = origin_edge->dst_node();
    const std::string& consumer_op_name = origin_consumer->op().op_name();
    if (mut_op_name2conf->find(consumer_op_name) == mut_op_name2conf->end()) {
      mut_op_name2conf->emplace(consumer_op_name, origin_consumer->op().op_conf());
    }
    CHECK_EQ(origin_edge->lbis().size(), 1);
    const LogicalBlobId& lbi = origin_edge->lbis().front();
    CHECK(input_lbn == GenLogicalBlobName(lbi));

    // 3. update consumer op
    for (const std::string& ibn : origin_edge->lbi2ibns().at(lbi)) {
      std::string old_lbn = ReplaceInputLbnInOpCustomizedConf(
          &mut_op_name2conf->at(consumer_op_name), ibn, output_lbn);
      CHECK(old_lbn == input_lbn);
    }

    VLOG(3) << " Update origin consumer op from: \n [ "
            << origin_consumer->op().op_conf().DebugString() << " ] \n to \n [ "
            << mut_op_name2conf->at(consumer_op_name).DebugString() << " ] \n";
  }
  return Maybe<void>::Ok();
}

Maybe<void> NcclLogicalOpFusionPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashMap<const OpNode*, int64_t> op_node2nccl_depth;
  HashMap<int64_t, std::vector<const OpNode*>> nccl_depth2nccl_ops;
  auto ConstForEachDataAndCtrlInNode = [&](const OpNode* node,
                                           const std::function<void(const OpNode*)>& Handler) {
    node->ForEachNodeOnInEdge(Handler);
    for (const auto& ctrl_in_op_name : node->op().op_conf().ctrl_in_op_name()) {
      const OpNode* in_node = op_graph.OpNode4OpName(ctrl_in_op_name);
      CHECK(in_node) << " cannot find ctrl_in_op_name: [" << ctrl_in_op_name << "] of op: ["
                     << node->op().op_name() << "] in OpGraph. ";
      Handler(in_node);
    }
  };
  op_graph.TopoForEachNodeWithCtrlEdge([&](const OpNode* node) {
    int64_t nccl_depth = 0;
    ConstForEachDataAndCtrlInNode(node, [&](const OpNode* in_node) {
      auto it = op_node2nccl_depth.find(in_node);
      CHECK(it != op_node2nccl_depth.end());  // topo search
      nccl_depth = std::max(nccl_depth, it->second);
    });
    if (IsNcclLogicalOpNode(node)) {
      nccl_depth++;  // ONLY nccl node update depth
      nccl_depth2nccl_ops[nccl_depth].push_back(node);
    }
    CHECK(op_node2nccl_depth.emplace(node, nccl_depth).second);
  });

  if (nccl_depth2nccl_ops.empty()) { return Maybe<void>::Ok(); }

  std::vector<OperatorConf> nccl_fusion_ops;
  std::vector<ParallelConf> nccl_fusion_op_parallel_confs;

  std::unordered_set<std::string> del_ops;
  HashMap<std::string, OperatorConf> mut_op_name2conf;

  for (const auto& pair : nccl_depth2nccl_ops) {
    HashMap<int64_t, HashMap<Shape, std::vector<const OpNode*>>> chain2hierarchy2nccl_ops;
    for (const OpNode* nccl_op : pair.second) {
      int64_t logical_chain_id = nccl_op->op().op_conf().logical_chain_id();
      const auto& hierarchy = nccl_op->parallel_desc().hierarchy();
      chain2hierarchy2nccl_ops[logical_chain_id][*hierarchy].push_back(nccl_op);
    }
    for (const auto& chain_pair : chain2hierarchy2nccl_ops) {
      for (const auto& hierarchy_pair : chain_pair.second) {
        JUST(ReplaceNcclOpsWithFusionOp(&nccl_fusion_ops, &nccl_fusion_op_parallel_confs, &del_ops,
                                        &mut_op_name2conf, hierarchy_pair.second));
      }
    }
  }

  job_builder->RemoveOpByName(del_ops);
  for (const auto& pair : mut_op_name2conf) { JUST(job_builder->MutOpOnlyOnce(pair.second)); }
  CHECK_EQ_OR_RETURN(nccl_fusion_ops.size(), nccl_fusion_op_parallel_confs.size());
  for (int32_t i = 0; i < nccl_fusion_ops.size(); ++i) {
    JUST(job_builder->AddOp(nccl_fusion_op_parallel_confs.at(i), nccl_fusion_ops.at(i)););
  }
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("NcclLogicalOpFusionPass", NcclLogicalOpFusionPass);

}  // namespace oneflow

#endif  // WITH_CUDA

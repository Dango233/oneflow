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
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/graph/task_graph_rebuild_ctx.h"

namespace oneflow {

namespace {

void ForEachDataEdge(const std::unordered_set<TaskEdge*>& edges,
                     const std::function<void(TaskEdge*)>& Handler) {
  for (TaskEdge* edge : edges) {
    const auto& regsts = edge->GetRegsts();
    int32_t data_regst_size =
        std::count_if(regsts.begin(), regsts.end(), [](const std::shared_ptr<RegstDesc>& regst) {
          return regst->regst_desc_type().has_data_regst_desc();
        });
    if (data_regst_size == regsts.size()) {
      Handler(edge);
    } else {
      CHECK_EQ(data_regst_size, 0);
    }
  }
}

}  // namespace

TaskNode::TaskNode()
    : machine_id_(-1), thrd_id_(-1), task_id_(-1), chain_id_(-1), order_in_chain_(-1) {}

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegst(const std::string& name) {
  auto produced_regsts_it = produced_regsts_.find(name);
  if (produced_regsts_it == produced_regsts_.end()) {
    return nullptr;
  } else {
    return produced_regsts_it->second;
  }
}

const std::list<std::shared_ptr<RegstDesc>>& TaskNode::GetConsumedRegst(const std::string& name) {
  return consumed_regsts_.at(name);
}

std::shared_ptr<RegstDesc> TaskNode::GetSoleConsumedRegst(const std::string& name) {
  auto it = consumed_regsts_.find(name);
  if (it == consumed_regsts_.end()) { return nullptr; }
  const std::list<std::shared_ptr<RegstDesc>>& vec = it->second;
  CHECK_EQ(vec.size(), 1);
  return vec.front();
}

const StreamId& TaskNode::stream_id() const {
  CHECK(new_task_id_);
  return new_task_id_->stream_id();
}

DeviceType TaskNode::device_type() const { return stream_id().device_id().device_type(); }

void TaskNode::set_machine_id(int64_t val) {
  CHECK_EQ(machine_id_, -1);
  machine_id_ = val;
  if (thrd_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_thrd_id(int64_t val) {
  CHECK_EQ(thrd_id_, -1);
  thrd_id_ = val;
  CHECK_GE(thrd_id_, 0);
  if (machine_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_chain_id(int64_t val) {
  CHECK(!IsValidChainId(chain_id_));
  chain_id_ = val;
}

void TaskNode::set_order_in_chain(int64_t val) {
  CHECK_EQ(order_in_chain_, -1);
  order_in_chain_ = val;
}

void TaskNode::PinConsumedRegst() {
  for (auto& pair : consumed_regsts_) {
    for (const std::shared_ptr<RegstDesc>& regst : pair.second) {
      PinConsumedRegstMemCase(regst->mut_mem_case());
    }
  }
}

void TaskNode::NaiveInferProducedDataRegstTimeShape() {
  if (IsMeaningLess()) { return; }
  std::shared_ptr<Shape> time_shape;
  ForEachConsumedDataRegst([&time_shape](const std::string& name, const RegstDesc* regst) {
    if (time_shape) {
      CHECK_EQ(*time_shape.get(), *regst->data_regst_time_shape().get());
    } else {
      time_shape = regst->data_regst_time_shape();
    }
  });

  CHECK(time_shape);

  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

void TaskNode::InferTimeShapeIfMeaningful() {
  if (!IsMeaningLess()) { InferProducedDataRegstTimeShape(); }
}

std::shared_ptr<Shape> TaskNode::GetFastestInputOutputTimeShape() const {
  std::shared_ptr<Shape> shape;
  auto UpdateRetShape = [&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) {
      if (!shape || shape->elem_cnt() < regst->data_regst_time_shape()->elem_cnt()) {
        shape = regst->data_regst_time_shape();
      }
    }
  };
  ForEachOutDataEdge(UpdateRetShape);
  if (shape) { return shape; }
  ForEachInDataEdge(UpdateRetShape);
  return shape;
}

void TaskNode::ForEachConsumedDataRegst(
    const std::function<void(const std::string&, const RegstDesc*)>& Handler) const {
  for (const auto& pair : consumed_regsts_) {
    for (const auto& regst : pair.second) {
      if (!regst->regst_desc_type().has_data_regst_desc()) { continue; }
      Handler(pair.first, regst.get());
    }
  }
}

void TaskNode::ForEachProducedDataRegst(
    const std::function<void(const std::string&, RegstDesc*)>& Handler) {
  for (auto& pair : produced_regsts_) {
    if (!pair.second->regst_desc_type().has_data_regst_desc()) { continue; }
    Handler(pair.first, pair.second.get());
  }
}

void TaskNode::Build() { BuildExecGphAndRegst(); }

void TaskNode::EraseUninitializedShapeProducedBlob() {
  for (auto& pair : produced_regsts_) { pair.second->EraseUninitializedShapeBlob(); }
}

void TaskNode::EraseZeroSizeConsumedRegst() {
  for (auto& pair : consumed_regsts_) {
    for (auto it = pair.second.begin(); it != pair.second.end();) {
      auto regst_ptr = *it;
      CHECK(regst_ptr);
      if (regst_ptr->regst_desc_type().has_data_regst_desc() && regst_ptr->NumOfLbi() == 0) {
        it = pair.second.erase(it);
      } else {
        ++it;
      }
    }
  }
  EraseIf<std::string, std::list<std::shared_ptr<RegstDesc>>>(
      &consumed_regsts_,
      [](HashMap<std::string, std::list<std::shared_ptr<RegstDesc>>>::iterator it) {
        return it->second.empty();
      });
}

void TaskNode::EraseZeroSizeProducedRegst() {
  EraseIf<std::string, std::shared_ptr<RegstDesc>>(
      &produced_regsts_, [](HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
        return it->second->regst_desc_type().has_data_regst_desc() && it->second->NumOfLbi() == 0;
      });
}

void TaskNode::UnbindBnWithEmptyRegst() {
  exec_gph_.ForEachNode([&](ExecNode* exec_node) { exec_node->UnbindBnWithEmptyRegst(); });
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << ":" << thrd_id_ << "\\n"
     << task_id_;
  return ss.str();
}

bool TaskNode::IsMeaningLess() { return produced_regsts_.empty() && consumed_regsts_.empty(); }

void TaskNode::InitFromProtoExceptConsumedRegsts(const TaskProto& task_proto) {
  // Step1: init some scalar items.
  CHECK(task_proto.task_type() == GetTaskType());
  machine_id_ = task_proto.machine_id();
  thrd_id_ = task_proto.thrd_id();
  task_id_ = task_proto.task_id();
  new_task_id_.reset(new TaskId(DecodeTaskIdFromInt64(task_id_)));
  CHECK(task_proto.job_id() == GlobalJobDesc().job_id());
  chain_id_ = task_proto.chain_id();
  order_in_chain_ = task_proto.order_in_chain();
  // Step2: check exec_gph empty.
  CHECK(task_proto.exec_sequence().exec_node().empty());
  // Step3: init produced_regst.
  for (const auto& pair : task_proto.produced_regst_desc()) {
    const auto& regst_desc = ProduceRegst(pair.first, pair.second.enable_reuse_mem());
    // regst_desc->consumers_ will be initialized by RegstDesc::InitConsumersFromProto.
    regst_desc->InitFromProtoExceptConsumers(pair.second);
  }
}

Maybe<void> TaskNode::InitConsumedRegstsFromProto(
    const TaskProto& task_proto,
    const std::function<Maybe<RegstDesc>(int64_t regst_desc_id)>& RegstDesc4Id) {
  // init consumed_regst.
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      ConsumeRegst(pair.first, JUST(RegstDesc4Id(regst_desc_id)));
    }
  }
  return Maybe<void>::Ok();
}

void TaskNode::ToProto(TaskProto* task_proto, bool check) const {
  // Step1: process some scalar items.
  task_proto->set_task_type(GetTaskType());
  task_proto->set_machine_id(machine_id_);
  task_proto->set_thrd_id(thrd_id_);
  task_proto->set_task_id(task_id_);
  task_proto->set_job_id(GlobalJobDesc().job_id());
  task_proto->set_chain_id(chain_id_);
  task_proto->set_order_in_chain(order_in_chain_);

  // Step2: process exec_gph.
  exec_gph_.ToExecSequence(parallel_ctx(), task_proto->mutable_exec_sequence());

  // Step3: process produced_regst.
  auto* produced_regst_proto = task_proto->mutable_produced_regst_desc();
  for (auto& pair : produced_regsts_) {
    RegstDescProto regst_desc_proto;
    pair.second->ToProto(&regst_desc_proto, check);
    CHECK(produced_regst_proto->insert({pair.first, regst_desc_proto}).second);
  }

  // Step4: process consumed_regst.
  auto* consumed_regst_proto = task_proto->mutable_consumed_regst_desc_id();
  for (const auto& pair : consumed_regsts_) {
    RegstDescIdSet regst_desc_ids;
    for (const std::shared_ptr<RegstDesc>& regst : pair.second) {
      regst_desc_ids.add_regst_desc_id(regst->regst_desc_id());
    }
    CHECK(consumed_regst_proto->insert({pair.first, regst_desc_ids}).second);
  }
}

MemZoneId TaskNode::MemZoneId121() const {
  StreamId stream_id = DecodeStreamIdFromInt64(thrd_id_);
  return stream_id.device_id();
}

bool TaskNode::BuildCtrlRegstDescIfNeed(TaskNode* dst_node, std::string* name) {
  if (IsMeaningLess() || dst_node->IsMeaningLess()) { return false; }
  for (const TaskEdge* in_edge : dst_node->in_edges()) {
    if (in_edge->src_node() == this) { return false; }
  }
  BuildCtrlRegstDesc(dst_node, name);
  return true;
}

RegstDesc* TaskNode::BuildCtrlRegstDesc(TaskNode* dst_node) {
  std::string name;
  return BuildCtrlRegstDesc(dst_node, &name);
}

RegstDesc* TaskNode::BuildCtrlRegstDesc(TaskNode* dst_node, std::string* name) {
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_ctrl_regst_desc();
  auto regst = NewProducedRegst(false, 1, kMaxRegisterNum, regst_desc_type);
  *name = "out_ctrl_" + std::to_string(regst->regst_desc_id());
  CHECK(produced_regsts_.emplace(*name, regst).second);
  dst_node->ConsumeRegst("in_ctrl", regst);
  return regst.get();
}

void TaskNode::BindEdgeWithProducedRegst(TaskEdge* edge, const std::string& name) {
  if (edge->HasRegst(name)) { return; }
  edge->AddRegst(name, GetProducedRegst(name));
}

std::shared_ptr<RegstDesc> TaskNode::GetAndCheckRegst(const std::string& name,
                                                      bool enable_reuse_mem,
                                                      int32_t min_register_num,
                                                      int32_t max_register_num) const {
  auto iter = produced_regsts_.find(name);
  if (iter == produced_regsts_.end()) { return nullptr; }
  const auto& regst = (iter->second);
  CHECK_EQ(regst->min_register_num(), min_register_num);
  CHECK_EQ(regst->max_register_num(), max_register_num);
  CHECK_EQ(regst->enable_reuse_mem(), enable_reuse_mem);
  return regst;
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem) {
  return ProduceRegst(name, enable_reuse_mem, 1, kMaxRegisterNum);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num) {
  // Because the Regst of separate compilation is not created in order, some Regst may have been
  // built. This implementation can avoid ProduceRegst being called multiple times.
  const auto& regst = GetAndCheckRegst(name, enable_reuse_mem, min_register_num, max_register_num);
  if (regst) { return regst; }
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_data_regst_desc();
  return ProduceRegst(name, enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num,
                                                  const RegstDescTypeProto& regst_desc_type) {
  auto regst =
      NewProducedRegst(enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
  CHECK(produced_regsts_.emplace(name, regst).second);
  return regst;
}

std::shared_ptr<RegstDesc> TaskNode::NewProducedRegst(bool enable_reuse_mem,
                                                      int32_t min_register_num,
                                                      int32_t max_register_num,
                                                      const RegstDescTypeProto& regst_desc_type) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  *(regst->mut_regst_desc_type()) = regst_desc_type;
  regst->UpdtMinRegstNumIfNeed(min_register_num);
  regst->UpdtMaxRegstNumIfNeed(max_register_num);
  regst->set_enable_reuse_mem(GlobalJobDesc().enable_reuse_mem() && enable_reuse_mem);
  InitProducedRegstMemCase(regst.get());
  return regst;
}

void TaskNode::InitProducedRegstMemCase(RegstDesc* regst) {
  InitProducedRegstMemCase(regst->mut_mem_case());
}

void TaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  mem_case->set_device_type(device_type());
  mem_case->set_device_id(stream_id().device_id().device_index());
}

void TaskNode::PinConsumedRegstMemCase(MemoryCase* mem_case) {
  // When a node located on non-cpu device consumes a cpu regst,
  // the regst memory should be pinned on host memory (locked page memory).
  // When the regst is not on host, skip pinning
  if (!memory::IsHostMem(*mem_case)) { return; }
  // When the node is located on host, skip pinning
  if (device_type() == DeviceType::kCPU) { return; }
  mem_case->set_pinned_device_type(device_type());
  mem_case->set_pinned_device_id(stream_id().device_id().device_index());
}

void TaskNode::ConsumeRegst(const std::string& name) {
  consumed_regsts_.emplace(name, std::list<std::shared_ptr<RegstDesc>>{});
}

void TaskNode::ConsumeRegst(const std::string& name, const std::shared_ptr<RegstDesc>& regst) {
  regst->AddConsumer(this);
  consumed_regsts_[name].emplace_back(regst);
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  StreamId stream_id = DecodeStreamIdFromInt64(thrd_id_);
  new_task_id_.reset(
      new TaskId(Singleton<IDMgr>::Get()->GetTaskIdGenerator()->Generate(stream_id)));
  task_id_ = EncodeTaskIdToInt64(*new_task_id_);
}

void TaskNode::EraseConsumedRegstsByName(const std::string& name) {
  if (consumed_regsts_.find(name) != consumed_regsts_.end()) {
    for (auto& regst : consumed_regsts_[name]) { regst->DeleteConsumer(this); }
    CHECK_EQ(consumed_regsts_.erase(name), 1);
  }
}

std::shared_ptr<RegstDesc> TaskEdge::GetRegst(const std::string& name_in_producer) const {
  return name_in_producer2regst_.at(name_in_producer);
}

bool TaskEdge::HasRegst(const std::string& name_in_producer) const {
  return (name_in_producer2regst_.find(name_in_producer) != name_in_producer2regst_.end());
}

std::shared_ptr<RegstDesc> TaskEdge::GetSoleRegst() const {
  CHECK_EQ(name_in_producer2regst_.size(), 1)
      << "edge: " << this << ", src: " << src_node()->task_id()
      << ", dst: " << dst_node()->task_id();
  return name_in_producer2regst_.begin()->second;
}

std::vector<std::shared_ptr<RegstDesc>> TaskEdge::GetRegsts() const {
  std::vector<std::shared_ptr<RegstDesc>> regst_descs;
  regst_descs.reserve(name_in_producer2regst_.size());
  for (auto& pair : name_in_producer2regst_) { regst_descs.emplace_back(pair.second); }
  return regst_descs;
}

void TaskEdge::AddRegst(const std::string& name_in_producer,
                        const std::shared_ptr<RegstDesc>& regst) {
  if (HasRegst(name_in_producer)) {
    CHECK(CHECK_JUST(MapAt(name_in_producer2regst_, name_in_producer))->regst_desc_id()
          == regst->regst_desc_id());
    return;
  }
  CHECK(name_in_producer2regst_.emplace(name_in_producer, regst).second);
}

void TaskEdge::CheckRegstLbiValid() const {
  HashMap<LogicalBlobId, std::shared_ptr<RegstDesc>> lbi2data_regst;
  for (auto& pair : name_in_producer2regst_) {
    std::shared_ptr<RegstDesc> regst = pair.second;
    if (regst->regst_desc_type().has_data_regst_desc()) {
      // NOTE(chengcheng): regst_desc_type is Set, BUT regst_desc_type.data_regst_desc is UNSET!
      //  So you can ONLY use NumOfLbi and ForEachLbi interface.
      CHECK_EQ(regst->NumOfLbi(), 1);
      regst->ForEachLbi(
          [&](const LogicalBlobId& lbi) { CHECK(lbi2data_regst.emplace(lbi, regst).second); });
    }
  }

  CHECK_EQ(lbi2data_regst.size(), lbis_.size())
      << " \n\n TaskEdge lbi and regst NOT match."
      << " TaskEdge: edge_id = " << edge_id() << " From: [" << src_node()->VisualStr() << "] To: ["
      << dst_node()->VisualStr() << "]\n";
  for (auto& lbi : lbis_) {
    CHECK(lbi2data_regst.find(lbi) != lbi2data_regst.end())
        << " \n\n Cannot find lbi: " << lbi.DebugString() << " in TaskEdge From: ["
        << src_node()->VisualStr() << "] To: [" << dst_node()->VisualStr() << "]\n\n";
  }
}

RegstDescProto* FindOrCreateProducedCtrlRegstDesc(TaskProto* task_proto,
                                                  const std::string& regst_desc_name) {
  auto* produced_regst_desc = task_proto->mutable_produced_regst_desc();
  if (produced_regst_desc->find(regst_desc_name) == produced_regst_desc->end()) {
    RegstDescProto ctrl_regst_desc;
    InitCtrlRegstDesc(task_proto->task_id(), &ctrl_regst_desc);
    CHECK(produced_regst_desc->insert({regst_desc_name, ctrl_regst_desc}).second);
  }
  return &produced_regst_desc->at(regst_desc_name);
}

RegstDescIdSet* FindOrCreateConsumedCtrlRegstDescIdSet(TaskProto* task_proto,
                                                       const std::string& regst_desc_name) {
  auto* consumed_regst_desc_id_sets = task_proto->mutable_consumed_regst_desc_id();
  if (consumed_regst_desc_id_sets->find(regst_desc_name) == consumed_regst_desc_id_sets->end()) {
    CHECK(consumed_regst_desc_id_sets->insert({regst_desc_name, RegstDescIdSet()}).second);
  }
  return &consumed_regst_desc_id_sets->at(regst_desc_name);
}

void TaskNode::ForEachInDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(in_edges(), Handler);
}

void TaskNode::ForEachOutDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(out_edges(), Handler);
}

void TaskNode::ForEachNodeOnInDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachInDataEdge([&](TaskEdge* in_edge) { Handler(in_edge->src_node()); });
}

void TaskNode::ForEachNodeOnOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachOutDataEdge([&](TaskEdge* out_edge) { Handler(out_edge->dst_node()); });
}

void TaskNode::ForEachNodeOnInOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachNodeOnInDataEdge(Handler);
  ForEachNodeOnOutDataEdge(Handler);
}

TaskEdge* TaskNode::GetSoleEdge(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                    const) const {
  TaskEdge* ret = nullptr;
  (this->*ForEachEdge)([&](TaskEdge* edge) {
    CHECK(ret == nullptr);
    ret = edge;
  });
  CHECK_NOTNULL(ret);
  return ret;
}

size_t TaskNode::GetEdgesSize(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                  const) const {
  size_t size = 0;
  (this->*ForEachEdge)([&](TaskEdge* edge) { ++size; });
  return size;
}

TaskEdge* TaskNode::SoleInDataEdge() const { return GetSoleEdge(&TaskNode::ForEachInDataEdge); }

TaskEdge* TaskNode::SoleOutDataEdge() const { return GetSoleEdge(&TaskNode::ForEachOutDataEdge); }

size_t TaskNode::in_data_edges_size() const { return GetEdgesSize(&TaskNode::ForEachInDataEdge); }

size_t TaskNode::out_data_edges_size() const { return GetEdgesSize(&TaskNode::ForEachOutDataEdge); }

Maybe<void> TaskEdge::InitFromProto(const TaskEdgeProto& proto,
                                    const TaskGraphRebuildCtx& task_graph_rebuild_ctx) {
  CHECK_NE_OR_RETURN(proto.src_task_id(), proto.dst_task_id()) << "self-loop are not supported";
  JUST(task_graph_rebuild_ctx.TaskNode4Id(proto.src_task_id()));
  JUST(task_graph_rebuild_ctx.TaskNode4Id(proto.dst_task_id()));
  // Note that edge id from proto is ignored.
  lbis_.insert(proto.lbi().begin(), proto.lbi().end());
  for (const auto& pair : proto.name_in_producer2regst_desc_id()) {
    AddRegst(pair.first, JUST(task_graph_rebuild_ctx.RegstDesc4Id(pair.second)));
  }
  return Maybe<void>::Ok();
}

void TaskEdge::ToProto(TaskEdgeProto* proto) const {
  // proto->set_task_edge_uid(edge_id());
  proto->set_task_edge_uid(reinterpret_cast<int64_t>(this));
  proto->set_src_task_id(src_node()->task_id());
  proto->set_dst_task_id(dst_node()->task_id());
  *proto->mutable_lbi() = {lbis_.begin(), lbis_.end()};
  auto* map = proto->mutable_name_in_producer2regst_desc_id();
  for (const auto& pair : name_in_producer2regst_) {
    CHECK(map->insert({pair.first, pair.second->regst_desc_id()}).second);
  }
}

}  // namespace oneflow

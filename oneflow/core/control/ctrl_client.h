#ifndef ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_
#define ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

class CtrlClient final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlClient);
  ~CtrlClient() = default;

  void Barrier(const std::string& barrier_name);
  void Barrier(const std::string& barrier_name, int32_t barrier_num);

  TryLockResult TryLock(const std::string& name);
  void NotifyDone(const std::string& name);
  void WaitUntilDone(const std::string& name);

  void PushKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PushKV(const std::string& k, const std::string& v);
  void PushKV(const std::string& k, const PbMessage& msg);
  template<typename T>
  void PushKVT(const std::string& k, T v) {
    static_assert(std::is_arithmetic<T>::value, "");
    PushKV(k, std::to_string(v));
  }

  void ClearKV(const std::string& k);
  void PullKV(const std::string& k,
              std::function<void(const std::string&)> VGetter);
  void PullKV(const std::string& k, std::string* v);
  void PullKV(const std::string& k, PbMessage* msg);
  template<typename T>
  void PullKVT(const std::string& k, T* v) {
    static_assert(std::is_arithmetic<T>::value, "");
    std::string v_str;
    PullKV(k, &v_str);
    *v = oneflow_cast<T>(v_str);
  }

  void PushActEvent(const ActEvent&);
  void Clear();

  int32_t IncreaseCount(const std::string& k, int32_t v);
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  void EraseCount(const std::string& k);

 private:
  friend class Global<CtrlClient>;
  CtrlClient();
  void LoadServer(const std::string& server_addr, CtrlService::Stub* stub);
  CtrlService::Stub* GetMasterStub() { return stubs_[0].get(); }
  CtrlService::Stub* GetThisStub();
  CtrlService::Stub* GetResponsibleStub(const std::string& key);

  std::vector<std::unique_ptr<CtrlService::Stub>> stubs_;
  std::mutex done_names_mtx_;
  HashSet<std::string> done_names_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() Global<CtrlClient>::Get()->Barrier(FILE_LINE_STR)

#define OF_CALL_ONCE(name, ...)                                        \
  do {                                                                 \
    TryLockResult lock_ret = Global<CtrlClient>::Get()->TryLock(name); \
    if (lock_ret == TryLockResult::kLocked) {                          \
      __VA_ARGS__;                                                     \
      Global<CtrlClient>::Get()->NotifyDone(name);                     \
    } else if (lock_ret == TryLockResult::kDone) {                     \
    } else if (lock_ret == TryLockResult::kDoing) {                    \
      Global<CtrlClient>::Get()->WaitUntilDone(name);                  \
    } else {                                                           \
      UNIMPLEMENTED();                                                 \
    }                                                                  \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_

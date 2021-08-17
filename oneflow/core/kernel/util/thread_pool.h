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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <utility>
#include "oneflow/core/kernel/util/registry.h"
#include "oneflow/core/kernel/util/thread_name.h"

namespace oneflow {
namespace internal {

class TaskThreadPoolBase {
 public:
  virtual void run(std::function<void()> func) = 0;

  virtual size_t size() const = 0;

  /**
   * The number of available (i.e. idle) threads in this thread pool.
   */
  virtual size_t numAvailable() const = 0;

  /**
   * Check if the current thread is from the thread pool.
   */
  virtual bool inThreadPool() const = 0;

  virtual ~TaskThreadPoolBase() noexcept {}

  static size_t defaultNumThreads() {
    auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    num_threads /= 2;
#endif
    return num_threads;
  }
};

class ThreadPool : public TaskThreadPoolBase {
 protected:
  struct task_element_t {
    bool run_with_id;
    const std::function<void()> no_id;
    const std::function<void(std::size_t)> with_id;

    explicit task_element_t(std::function<void()> f)
        : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
    explicit task_element_t(std::function<void(std::size_t)> f)
        : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
  };

  std::queue<task_element_t> tasks_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  std::atomic_bool running_;
  bool complete_;
  std::size_t available_;
  std::size_t total_;
  int numa_node_id_;

 public:
  ThreadPool() = delete;

  explicit ThreadPool(int pool_size, int numa_node_id = -1,
                      std::function<void()> init_thread = nullptr);

  ~ThreadPool();

  size_t size() const override;

  size_t numAvailable() const override;

  bool inThreadPool() const override;

  void run(std::function<void()> func) override;

  template<typename Task>
  void runTaskWithID(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Set task and signal condition variable so that a worker thread will
    // wake up and use the task.
    tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
    complete_ = false;
    condition_.notify_one();
  }

  /// @brief Wait for queue to be empty
  void waitWorkComplete();

 private:
  // @brief Entry point for pool threads.
  void main_loop(std::size_t index);
};

class TaskThreadPool : public ThreadPool {
 public:
  explicit TaskThreadPool(std::size_t pool_size, int numa_node_id = -1)
      : ThreadPool(pool_size, numa_node_id, [numa_node_id]() {
          setThreadName("OneFlowTaskThread");
          NUMABind(numa_node_id);
        }) {}
};

ONEFLOW_DECLARE_SHARED_REGISTRY(ThreadPoolRegistry, TaskThreadPoolBase, int, int, bool);

}  // namespace internal
}  // namespace oneflow

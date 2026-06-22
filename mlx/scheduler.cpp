// Copyright © 2023-2026 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/utils.h"

namespace mlx::core {

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
    f.wait();
  } else {
    gpu::synchronize(s);
  }
}

void synchronize(ThreadLocalStream s) {
  synchronize(stream_from_thread_local_stream(s));
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

void clear_streams() {
  cpu::clear_streams();
  gpu::clear_streams();
}

namespace scheduler {

Scheduler::Scheduler() {
  is_main_thread();
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::enqueue(Stream s, std::function<void()> task) {
  StreamThread* st = nullptr;
  {
    std::shared_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it != threads_.end()) {
      st = it->second.get();
    }
  }
  if (!st) {
    std::unique_lock lock(threads_mtx_);
    auto it = threads_.find(s.index);
    if (it == threads_.end()) {
      it = threads_.emplace(s.index, std::make_unique<StreamThread>()).first;
    }
    st = it->second.get();
  }
  st->enqueue(std::move(task));
}

void Scheduler::enqueue_event(
    Stream s,
    Event event,
    std::function<void(Event&)> task) {
  assert(s.device == Device::cpu);
  // Keep a copy of the event until it is processed.
  decltype(events_)::mapped_type::iterator iter;
  {
    std::unique_lock lock(events_mtx_);
    auto& list = events_[s.index];
    iter = list.insert(list.end(), std::move(event));
  }
  enqueue(s, [this, s, iter, task = std::move(task)]() {
    task(*iter);
    {
      std::unique_lock lock(events_mtx_);
      auto& list = events_[s.index];
      auto err = iter->error();
      list.erase(iter);
      // Poison all pending events if there was an error.
      if (err) {
        for (auto& event : list) {
          event.set_error(err);
        }
      }
    }
  });
}

void Scheduler::set_error(Stream s, std::shared_ptr<std::string> error) {
  assert(s.device == Device::cpu);
  std::unique_lock lock(events_mtx_);
  for (auto& event : events_[s.index]) {
    event.set_error(error);
  }
}

// Leak the scheduler singleton on all platforms. During static destruction,
// worker threads may still be executing JIT-compiled code that has been
// unmapped, causing SIGSEGV (macOS/Linux) or join() deadlocks (Windows/MSVC
// CRT).
// The OS reclaims all resources at process exit anyway.
Scheduler& scheduler() {
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
}

} // namespace scheduler
} // namespace mlx::core

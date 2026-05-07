/**
 * @file wbc_core/wbc_util/include/wbc_util/step_count_guard.hpp
 * @brief RAII guard for incrementing control-step counter at scope exit.
 */
#pragma once

namespace util {

/**
 * @brief Increment the pointed counter exactly once when leaving scope.
 *
 * @details
 * This is intended for control-loop bookkeeping where multiple early returns
 * exist but the tick count must be updated exactly once.
 */
template <typename CounterT>
class StepCountGuard final {
public:
  explicit StepCountGuard(CounterT* counter) : counter_(counter) {}

  ~StepCountGuard() {
    if (counter_ != nullptr) {
      ++(*counter_);
    }
  }

  StepCountGuard(const StepCountGuard&) = delete;
  StepCountGuard& operator=(const StepCountGuard&) = delete;

  StepCountGuard(StepCountGuard&& other) noexcept : counter_(other.counter_) {
    other.counter_ = nullptr;
  }

  StepCountGuard& operator=(StepCountGuard&& other) noexcept {
    if (this != &other) {
      counter_ = other.counter_;
      other.counter_ = nullptr;
    }
    return *this;
  }

private:
  CounterT* counter_{nullptr};
};

} // namespace util

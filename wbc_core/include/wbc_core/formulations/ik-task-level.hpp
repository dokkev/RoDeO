//
// Copyright (c) 2026
//
// Legacy IK-stage task metadata.
//
// This header exists only for the staged IK formulation used by the old
// redundancy-resolution path. Final-form IDHQP runtime code should not depend on
// these types.
//

#ifndef __wbc_formulations_ik_task_level_hpp__
#define __wbc_formulations_ik_task_level_hpp__

#include "wbc_core/tasks/task-base.hpp"

namespace wbc {

enum class IKMode {
  kPreserve,
  kTrack,
};

struct IKTaskLevel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  tasks::TaskBase& task;
  double ik_error_sign{1.0};
  double kp_ik{1.0};
  IKMode ik_mode{IKMode::kPreserve};

  IKTaskLevel(tasks::TaskBase& task, double kp_ik = 1.0,
              IKMode ik_mode = IKMode::kPreserve,
              double ik_error_sign = 1.0)
      : task(task),
        ik_error_sign(ik_error_sign),
        kp_ik(kp_ik),
        ik_mode(ik_mode) {}
};

}  // namespace wbc

#endif  // ifndef __wbc_formulations_ik_task_level_hpp__

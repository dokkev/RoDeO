/**
 * @file wbc_core/wbc_formulation/src/force_task.cpp
 * @brief Doxygen documentation for force_task module.
 */
#include "wbc_formulation/force_task.hpp"

#include <stdexcept>
#include <string>

namespace wbc {
////////////////////////////////////////////////////////////////////////////////
ForceTask::ForceTask(PinocchioRobotSystem* robot, Contact* contact)
    : robot_(robot),
      contact_(contact),
      dim_(contact->Dim()),
      rf_des_(Eigen::VectorXd::Zero(contact->Dim())),
      rf_cmd_(Eigen::VectorXd::Zero(contact->Dim())),
      weight_(Eigen::VectorXd::Zero(contact->Dim())) {
  if (dim_ != 3 && dim_ != 6) {
    throw std::runtime_error("[ForceTask] Unsupported contact dimension: " +
                             std::to_string(dim_));
  }
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateDesired(const Eigen::VectorXd& rf_des) {
  rf_des_ = rf_des;
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateDesiredToLocal(const Eigen::VectorXd& rf_des) {
  const Eigen::Matrix3d r =
      robot_->GetLinkIsometry(contact_->TargetLinkIdx()).linear().transpose();
  if (dim_ == 6) {
    Eigen::Matrix<double, 6, 6> local_R_world = Eigen::Matrix<double, 6, 6>::Zero();
    local_R_world.topLeftCorner<3, 3>()     = r;
    local_R_world.bottomRightCorner<3, 3>() = r;
    rf_des_ = local_R_world * rf_des;
  } else {  // dim_ == 3, validated at construction
    rf_des_ = r * rf_des;
  }
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateCmd(const Eigen::VectorXd& rf_cmd) {
  rf_cmd_ = rf_cmd;
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::SetParameters(const ForceTaskConfig& config) {
  if (static_cast<int>(config.weight.size()) != dim_) {
    // Silently reject: throwing inside Step() (the RT loop) would crash the
    // controller thread. Misconfigured weights are caught at startup when
    // the YAML is validated against the contact dimension.
    return;
  }
  weight_ = config.weight;
}

} // namespace wbc

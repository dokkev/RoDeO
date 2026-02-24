#include "wbc_formulation/force_task.hpp"

#include <cassert>

namespace wbc {
////////////////////////////////////////////////////////////////////////////////
ForceTask::ForceTask(PinocchioRobotSystem* robot, Contact* contact)
    : robot_(robot),
      contact_(contact),
      dim_(contact->Dim()),
      rf_des_(Eigen::VectorXd::Zero(contact->Dim())),
      rf_cmd_(Eigen::VectorXd::Zero(contact->Dim())),
      weight_(Eigen::VectorXd::Zero(contact->Dim())) {}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateDesired(const Eigen::VectorXd& rf_des) {
  rf_des_ = rf_des;
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateDesiredToLocal(const Eigen::VectorXd& rf_des) {
  Eigen::MatrixXd local_R_world(contact_->Dim(), contact_->Dim());
  local_R_world.setZero();

  if (contact_->Dim() == 6) {
    const Eigen::Matrix3d r =
        robot_->GetLinkIsometry(contact_->TargetLinkIdx()).linear().transpose();
    local_R_world.topLeftCorner<3, 3>() = r;
    local_R_world.bottomRightCorner<3, 3>() = r;
    rf_des_ = local_R_world * rf_des;
  } else if (contact_->Dim() == 3) {
    local_R_world =
        robot_->GetLinkIsometry(contact_->TargetLinkIdx()).linear().transpose();
    rf_des_ = local_R_world * rf_des;
  } else {
    assert(false && "Unsupported contact dimension");
  }
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::UpdateCmd(const Eigen::VectorXd& rf_cmd) {
  rf_cmd_ = rf_cmd;
}

////////////////////////////////////////////////////////////////////////////////
void ForceTask::SetParameters(const ForceTaskConfig& config) {
  assert(config.weight.size() == dim_ &&
         "ForceTask weight dimension mismatch");
  weight_ = config.weight;
}

} // namespace wbc

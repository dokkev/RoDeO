#pragma once

#include "wbc_formulation/friction_cone.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

namespace wbc {
/**
 * @brief Typed parameters for force task.
 */
struct ForceTaskConfig {
  Eigen::VectorXd weight;
};

/**
 * @brief Reaction-force tracking task for a given contact constraint.
 */
class ForceTask {
public:
  ForceTask(PinocchioRobotSystem* robot, Contact* contact);
  ~ForceTask() = default;

  void UpdateDesired(const Eigen::VectorXd& rf_des);
  void UpdateDesiredToLocal(const Eigen::VectorXd& rf_des);
  void UpdateCmd(const Eigen::VectorXd& rf_cmd);

  const Eigen::VectorXd& DesiredRf() const { return rf_des_; }
  const Eigen::VectorXd& CmdRf() const { return rf_cmd_; }
  const Eigen::VectorXd& Weight() const { return weight_; }
  int Dim() const { return dim_; }
  Contact* GetContact() const { return contact_; }

  void SetParameters(const ForceTaskConfig& config);

private:
  PinocchioRobotSystem* robot_;
  Contact* contact_;
  int dim_;
  Eigen::VectorXd rf_des_;
  Eigen::VectorXd rf_cmd_;
  Eigen::VectorXd weight_;
};

} // namespace wbc

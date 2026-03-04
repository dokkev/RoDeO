/**
 * @file wbc_core/wbc_solver/include/wbc_solver/interface/wbc.hpp
 * @brief Doxygen documentation for wbc module.
 */
#pragma once

#include <Eigen/Dense>

#include <vector>

namespace wbc {
struct WbcFormulation;

/**
 * @brief Base class for WBC algorithms.
 */
class WBC {
public:
  explicit WBC(const std::vector<bool>& act_qdot_list)
      : num_qdot_(static_cast<int>(act_qdot_list.size())),
        num_active_(0),
        num_floating_(0),
        dim_contact_(0),
        is_floating_base_(false),
        has_contact_(true),
        settings_updated_(false) {
    if (act_qdot_list.empty()) {
      return;
    }

    if (act_qdot_list[0]) {
      num_floating_ = 0;
      is_floating_base_ = false;
    } else {
      num_floating_ = 6;
      is_floating_base_ = true;
    }

    for (int i = 0; i < num_qdot_; ++i) {
      if (act_qdot_list[i]) {
        ++num_active_;
      }
    }

    M_ = Eigen::MatrixXd::Zero(num_qdot_, num_qdot_);
    Minv_ = Eigen::MatrixXd::Zero(num_qdot_, num_qdot_);
    cori_ = Eigen::VectorXd::Zero(num_qdot_);
    grav_ = Eigen::VectorXd::Zero(num_qdot_);

    sa_.setZero(num_active_, num_qdot_);   // S_a: actuated DOF selector.
    sf_.setZero(num_floating_, num_qdot_); // S_f: floating-base DOF selector.
    snf_.setZero(num_qdot_ - num_floating_,
                 num_qdot_); // S_nf: non-floating DOF selector.

    int active_idx = 0;
    int float_idx = 0;
    int nonfloat_idx = 0;
    for (int i = 0; i < num_qdot_; ++i) {
      if (act_qdot_list[i]) {
        sa_(active_idx++, i) = 1.0;
      } else {
        if (i < num_floating_) {
          sf_(float_idx++, i) = 1.0;
        }
      }
      if (i >= num_floating_) {
        snf_(nonfloat_idx++, i) = 1.0;
      }
    }
  }

  virtual ~WBC() = default;

  /**
   * @brief Update dynamic model terms used by the solver.
   */
  template <typename DerivedM, typename DerivedMinv, typename DerivedCori,
            typename DerivedGrav>
  void UpdateSetting(const Eigen::MatrixBase<DerivedM>& M,
                     const Eigen::MatrixBase<DerivedMinv>& Minv,
                     const Eigen::MatrixBase<DerivedCori>& cori,
                     const Eigen::MatrixBase<DerivedGrav>& grav) {
    M_ = M;
    Minv_ = Minv;
    cori_ = cori;
    grav_ = grav;
    settings_updated_ = true;
  }

  /**
   * @brief Compute joint torque command from current formulation and desired qddot.
   */
  virtual bool MakeTorque(const WbcFormulation& formulation,
                          const Eigen::VectorXd& wbc_qddot_cmd,
                          Eigen::VectorXd& jtrq_cmd) = 0;

  virtual void SetParameters() {}

protected:
  int num_qdot_;
  int num_active_;
  int num_floating_;
  int dim_contact_;
  bool is_floating_base_;
  bool has_contact_;

  Eigen::MatrixXd sf_;
  Eigen::MatrixXd sa_;
  Eigen::MatrixXd snf_;

  Eigen::MatrixXd M_;
  Eigen::MatrixXd Minv_;
  Eigen::VectorXd cori_;
  Eigen::VectorXd grav_;
  bool settings_updated_;
};

} // namespace wbc

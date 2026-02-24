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
        b_floating_base_(false),
        b_contact_(true),
        b_update_setting_(false) {
    if (act_qdot_list.empty()) {
      return;
    }

    if (act_qdot_list[0]) {
      num_floating_ = 0;
      b_floating_base_ = false;
    } else {
      num_floating_ = 6;
      b_floating_base_ = true;
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

    sa_.setZero(num_active_, num_qdot_);
    sf_.setZero(num_floating_, num_qdot_);
    snf_.setZero(num_qdot_ - num_floating_, num_qdot_);

    int j = 0;
    int e = 0;
    int l = 0;
    for (int i = 0; i < num_qdot_; ++i) {
      if (act_qdot_list[i]) {
        sa_(j++, i) = 1.0;
      } else {
        if (i < num_floating_) {
          sf_(e++, i) = 1.0;
        }
      }
      if (i >= num_floating_) {
        snf_(l++, i) = 1.0;
      }
    }
  }

  virtual ~WBC() = default;

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
    b_update_setting_ = true;
  }

  virtual bool MakeTorque(const WbcFormulation& formulation,
                          const Eigen::VectorXd& wbc_qddot_cmd,
                          Eigen::VectorXd& jtrq_cmd) = 0;

  virtual void SetParameters() {}

protected:
  int num_qdot_;
  int num_active_;
  int num_floating_;
  int dim_contact_;
  bool b_floating_base_;
  bool b_contact_;

  Eigen::MatrixXd sf_;
  Eigen::MatrixXd sa_;
  Eigen::MatrixXd snf_;

  Eigen::MatrixXd M_;
  Eigen::MatrixXd Minv_;
  Eigen::VectorXd cori_;
  Eigen::VectorXd grav_;
  bool b_update_setting_;
};

} // namespace wbc

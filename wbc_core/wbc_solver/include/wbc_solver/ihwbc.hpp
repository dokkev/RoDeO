#pragma once

#include <vector>

#include "wbc_solver/interface/wbc.hpp"

namespace wbc {

// Placeholder interface to reserve explicit IHWBC API location in the new
// solver-oriented directory layout.
class IHWBC : public WBC {
public:
  explicit IHWBC(const std::vector<bool>& act_qdot_list) : WBC(act_qdot_list) {}
  ~IHWBC() override = default;

  bool MakeTorque(const WbcFormulation& /*formulation*/,
                  const Eigen::VectorXd& /*wbc_qddot_cmd*/,
                  Eigen::VectorXd& /*jtrq_cmd*/) override {
    return false;
  }
};

} // namespace wbc

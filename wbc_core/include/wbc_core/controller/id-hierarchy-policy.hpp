//
// Copyright (c) 2026
//
// Minimal hierarchy anchors for generic level-based IDHQP.
//

#ifndef __wbc_controller_id_hierarchy_policy_hpp__
#define __wbc_controller_id_hierarchy_policy_hpp__

#include <algorithm>

namespace wbc {

struct IDHierarchyPolicy {
  unsigned int physicsLevel{0};

  bool isObjectiveLevelValid(unsigned int level) const {
    return level > physicsLevel;
  }

  unsigned int regularizationLevel(unsigned int max_objective_level) const {
    return std::max(physicsLevel, max_objective_level) + 1u;
  }

  bool isValidFor(unsigned int max_objective_level) const {
    return physicsLevel < regularizationLevel(max_objective_level);
  }

  unsigned int numLevels(unsigned int max_objective_level) const {
    return regularizationLevel(max_objective_level) + 1u;
  }
};

}  // namespace wbc

#endif  // ifndef __wbc_controller_id_hierarchy_policy_hpp__

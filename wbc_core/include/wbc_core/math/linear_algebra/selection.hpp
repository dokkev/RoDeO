#pragma once

#include "wbc_core/math/fwd.hpp"

namespace wbc::math {

void buildSelectionMatrix(ConstRefVector mask, Eigen::Index cols,
                          Eigen::Index col_offset, VectorXi& active_indices,
                          Matrix& selection);

void buildWeightedSelectionMatrix(ConstRefVector mask, Eigen::Index cols,
                                  Eigen::Index col_offset,
                                  ConstRefVector weights,
                                  VectorXi& active_indices,
                                  Matrix& selection);

}  // namespace wbc::math

#include "wbc_core/math/linear_algebra/selection.hpp"

#include <pinocchio/macros.hpp>

namespace wbc::math {
namespace {

Eigen::Index activeCount(ConstRefVector mask) {
  Eigen::Index count = 0;
  for (Eigen::Index i = 0; i < mask.size(); ++i) {
    if (mask(i) != 0.0) {
      PINOCCHIO_CHECK_INPUT_ARGUMENT(mask(i) == 1.0,
                                     "Mask entries must be either 0.0 or 1.0");
      ++count;
    }
  }
  return count;
}

}  // namespace

void buildSelectionMatrix(ConstRefVector mask, Eigen::Index cols,
                          Eigen::Index col_offset, VectorXi& active_indices,
                          Matrix& selection) {
  const Eigen::Index rows = activeCount(mask);
  active_indices.resize(rows);
  selection.setZero(rows, cols);

  Eigen::Index row = 0;
  for (Eigen::Index i = 0; i < mask.size(); ++i) {
    if (mask(i) == 0.0) continue;

    selection(row, col_offset + i) = 1.0;
    active_indices(row) = static_cast<int>(i);
    ++row;
  }
}

void buildWeightedSelectionMatrix(ConstRefVector mask, Eigen::Index cols,
                                  Eigen::Index col_offset,
                                  ConstRefVector weights,
                                  VectorXi& active_indices,
                                  Matrix& selection) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      weights.size() == mask.size(),
      "Weights must have the same size as the selection mask");

  buildSelectionMatrix(mask, cols, col_offset, active_indices, selection);
  for (Eigen::Index row = 0; row < active_indices.size(); ++row) {
    const Eigen::Index source = active_indices(row);
    selection(row, col_offset + source) = weights(source);
  }
}

}  // namespace wbc::math

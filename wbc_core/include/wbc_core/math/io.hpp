#pragma once

#include <fstream>
#include <string>

#include <Eigen/Core>

namespace wbc::math {

template <class Matrix>
bool writeMatrix(const std::string& filename,
                 const Eigen::MatrixBase<Matrix>& matrix) {
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;

  std::ofstream out(filename.c_str(),
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) return false;
  Index rows = matrix.rows(), cols = matrix.cols();
  out.write((char*)(&rows), sizeof(Index));
  out.write((char*)(&cols), sizeof(Index));
  out.write((char*)matrix.data(), rows * cols * sizeof(Scalar));
  out.close();
  return true;
}

template <class Matrix>
bool readMatrix(const std::string& filename,
                const Eigen::MatrixBase<Matrix>& matrix) {
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;

  std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
  if (!in.is_open()) return false;
  Index rows = 0, cols = 0;
  in.read((char*)(&rows), sizeof(Index));
  in.read((char*)(&cols), sizeof(Index));

  Eigen::MatrixBase<Matrix>& matrix_ =
      const_cast<Eigen::MatrixBase<Matrix>&>(matrix);

  matrix_.resize(rows, cols);
  in.read((char*)matrix_.data(), rows * cols * sizeof(Scalar));
  in.close();
  return true;
}

}  // namespace wbc::math

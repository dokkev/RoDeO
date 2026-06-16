#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace wbc {

template <typename T>
std::string toString(const T& v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

template <typename T>
std::string toString(const std::vector<T>& v,
                     const std::string separator = ", ") {
  std::stringstream ss;
  for (int i = 0; i < static_cast<int>(v.size()) - 1; i++) {
    ss << v[i] << separator;
  }
  if (!v.empty()) {
    ss << v[v.size() - 1];
  }
  return ss.str();
}

template <typename Derived>
std::string toString(const Eigen::MatrixBase<Derived>& v,
                     const std::string separator = ", ") {
  (void)separator;
  std::stringstream ss;
  if (v.rows() > v.cols()) {
    ss << v.transpose();
  } else {
    ss << v;
  }
  return ss.str();
}

}  // namespace wbc

namespace wbc::math {

static const Eigen::IOFormat CleanFmt(1, 0, ", ", "\n", "[", "]");
static const Eigen::IOFormat matlabPrintFormat(Eigen::FullPrecision,
                                               Eigen::DontAlignCols, " ", ";\n",
                                               "", "", "[", "];");

}  // namespace wbc::math

#define PRINT_VECTOR(a)                                 \
  std::cout << #a << "(" << a.rows() << "x" << a.cols() \
            << "): " << a.transpose().format(wbc::math::CleanFmt) << std::endl

#define PRINT_MATRIX(a)                                           \
  std::cout << #a << "(" << a.rows() << "x" << a.cols() << "):\n" \
            << a.format(wbc::math::CleanFmt) << std::endl

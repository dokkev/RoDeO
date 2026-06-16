//
// Copyright (c) 2017 CNRS
//

#ifndef __invdyn_math_constraint_equality_hpp__
#define __invdyn_math_constraint_equality_hpp__

#include "wbc_core/math/constraints/constraint-base.hpp"

namespace wbc {
namespace math {

class ConstraintEquality : public ConstraintBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ConstraintEquality(const std::string& name);

  ConstraintEquality(const std::string& name, const unsigned int rows,
                     const unsigned int cols);

  ConstraintEquality(const std::string& name, ConstRefMatrix A,
                     ConstRefVector b);

  unsigned int rows() const override;
  unsigned int cols() const override;
  void resize(unsigned int r, unsigned int c) override;

  bool isEquality() const override;
  bool isInequality() const override;
  bool isBound() const override;

  const Vector& vector() const override;
  const Vector& lowerBound() const override;
  const Vector& upperBound() const override;

  Vector& vector() override;
  Vector& lowerBound() override;
  Vector& upperBound() override;

  bool setVector(ConstRefVector b) override;
  bool setLowerBound(ConstRefVector lb) override;
  bool setUpperBound(ConstRefVector ub) override;

  bool isSatisfied(ConstRefVector x, double tol = 1e-6) const override;

 protected:
  Vector m_b;
};

}  // namespace math
}  // namespace wbc

#endif  // ifndef __invdyn_math_constraint_equality_hpp__

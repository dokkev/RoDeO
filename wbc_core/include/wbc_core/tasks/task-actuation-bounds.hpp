//
// Copyright (c) 2017 CNRS
//
// This file is part of tsid
// tsid is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// tsid is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// tsid If not, see
// <http://www.gnu.org/licenses/>.
//

#ifndef __invdyn_task_actuation_bounds_hpp__
#define __invdyn_task_actuation_bounds_hpp__

#include <wbc_core/tasks/task-actuation.hpp>
#include <wbc_core/trajectories/trajectory-base.hpp>
#include <wbc_core/math/constraint-inequality.hpp>

namespace wbc {
namespace tasks {

class TaskActuationBounds : public TaskActuation {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Index Index;
  typedef trajectories::TrajectorySample TrajectorySample;
  typedef math::Vector Vector;
  typedef math::VectorXi VectorXi;
  typedef math::ConstraintInequality ConstraintInequality;
  typedef pinocchio::Data Data;

  TaskActuationBounds(const std::string& name, RobotSystem& robot);

  int dim() const override;

  const ConstraintBase& compute(double t, ConstRefVector q, ConstRefVector v,
                                Data& data) override;

  const ConstraintBase& getConstraint() const override;

  void setBounds(ConstRefVector lower, ConstRefVector upper);
  const Vector& getLowerBounds() const;
  const Vector& getUpperBounds() const;

  const Vector& mask() const;
  void mask(const Vector& mask);

 protected:
  Vector m_mask;
  VectorXi m_activeAxes;
  ConstraintInequality m_constraint;
};

}  // namespace tasks
}  // namespace wbc

#endif  // ifndef __invdyn_task_actuation_bounds_hpp__

//
// Copyright (c) 2017 CNRS, NYU, MPI Tübingen
//

#ifndef __invdyn_contact_6d_hpp__
#define __invdyn_contact_6d_hpp__

#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/tasks/task-se3-equality.hpp"
#include "wbc_core/math/constraint-inequality.hpp"
#include "wbc_core/math/constraint-equality.hpp"

namespace wbc {
namespace contacts {
class Contact6d : public ContactBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::ConstRefMatrix ConstRefMatrix;
  typedef math::ConstRefVector ConstRefVector;
  typedef math::Matrix3x Matrix3x;
  typedef math::Vector6 Vector6;
  typedef math::Vector3 Vector3;
  typedef math::Vector Vector;
  typedef tasks::TaskSE3Equality TaskSE3Equality;
  typedef math::ConstraintInequality ConstraintInequality;
  typedef math::ConstraintEquality ConstraintEquality;
  typedef pinocchio::SE3 SE3;

  Contact6d(const std::string& name, RobotSystem& robot,
            const std::string& frameName, ConstRefMatrix contactPoints,
            ConstRefVector contactNormal, const double frictionCoefficient,
            const double minNormalForce, const double maxNormalForce);

  /// Return the number of motion constraints
  unsigned int n_motion() const override;

  /// Return the number of force variables
  unsigned int n_force() const override;

  const ConstraintBase& computeMotionConstraint(double t, ConstRefVector q,
                                                ConstRefVector v,
                                                Data& data) override;

  const ConstraintInequality& computeForceTask(double t, ConstRefVector q,
                                               ConstRefVector v,
                                               const Data& data) override;

  const Matrix& getForceGeneratorMatrix() const override;

  const ConstraintEquality& computeForceRegularizationTask(
      double t, ConstRefVector q, ConstRefVector v, const Data& data) override;

  const TaskSE3Equality& getMotionTask() const override;
  const ConstraintBase& getMotionConstraint() const override;
  const ConstraintInequality& getForceConstraint() const override;
  const ConstraintEquality& getForceRegularizationTask() const override;

  double getNormalForce(ConstRefVector f) const override;
  double getMinNormalForce() const override;
  double getMaxNormalForce() const override;
  const Matrix3x& getContactPoints() const override;

  const Vector& Kp() const;
  const Vector& Kd() const;
  void Kp(ConstRefVector Kp);
  void Kd(ConstRefVector Kd);

  bool setContactPoints(ConstRefMatrix contactPoints);
  bool setContactNormal(ConstRefVector contactNormal);

  bool setFrictionCoefficient(double frictionCoefficient);
  bool setMinNormalForce(double minNormalForce) override;
  bool setMaxNormalForce(double maxNormalForce) override;
  void setReference(const SE3& ref);
  void setForceReference(ConstRefVector& f_ref);
  void setRegularizationTaskWeightVector(ConstRefVector& w);

 private:
  void init();

 protected:
  void updateForceInequalityConstraints();
  void updateForceRegularizationTask();
  void updateForceGeneratorMatrix();

  TaskSE3Equality m_motionTask;
  ConstraintInequality m_forceInequality;
  ConstraintEquality m_forceRegTask;
  Matrix3x m_contactPoints;
  Vector3 m_contactNormal;
  Vector6 m_fRef;
  Vector6 m_weightForceRegTask;
  double m_mu;
  double m_fMin;
  double m_fMax;
  Matrix m_forceGenMat;
};
}  // namespace contacts
}  // namespace wbc

#endif  // ifndef __invdyn_contact_6d_hpp__

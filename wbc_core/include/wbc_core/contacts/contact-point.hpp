//
// Copyright (c) 2017 CNRS, NYU, MPI Tübingen
//

#ifndef __invdyn_contact_point_hpp__
#define __invdyn_contact_point_hpp__

#include "wbc_core/contacts/contact-base.hpp"
#include "wbc_core/tasks/task-se3-equality.hpp"
#include "wbc_core/math/constraint-inequality.hpp"
#include "wbc_core/math/constraint-equality.hpp"

namespace wbc {
namespace contacts {
class ContactPoint : public ContactBase {
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

  ContactPoint(const std::string& name, RobotSystem& robot,
               const std::string& frameName, ConstRefVector contactNormal,
               const double frictionCoefficient, const double minNormalForce,
               const double maxNormalForce);

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
  const Matrix3x& getContactPoints() const override;

  double getNormalForce(ConstRefVector f) const override;
  double getMinNormalForce() const override;
  double getMaxNormalForce() const override;

  const Vector&
  Kp();  // cannot be const because it set a member variable inside
  const Vector&
  Kd();  // cannot be const because it set a member variable inside
  void Kp(ConstRefVector Kp);
  void Kd(ConstRefVector Kd);

  bool setContactNormal(ConstRefVector contactNormal);

  bool setFrictionCoefficient(const double frictionCoefficient);
  bool setMinNormalForce(const double minNormalForce) override;
  bool setMaxNormalForce(const double maxNormalForce) override;
  void setReference(const SE3& ref);
  void setForceReference(ConstRefVector& f_ref);
  void setRegularizationTaskWeightVector(ConstRefVector& w);

  /**
   * @brief Specifies if properties of the contact point and motion task
   * are expressed in the local or local world oriented frame. The contact
   * forces, contact normal and contact coefficients are interpreted in
   * the specified frame.
   *
   * @param local_frame If true, use the local frame, otherwise use the
   * local world oriented
   */
  void useLocalFrame(bool local_frame);

 protected:
  void updateForceInequalityConstraints();
  void updateForceRegularizationTask();
  void updateForceGeneratorMatrix();

  TaskSE3Equality m_motionTask;
  ConstraintInequality m_forceInequality;
  ConstraintEquality m_forceRegTask;
  Vector3 m_contactNormal;
  Vector3 m_fRef;
  Vector3 m_weightForceRegTask;
  Matrix3x m_contactPoints;
  Vector m_Kp3, m_Kd3;  // gain vectors to be returned by reference
  double m_mu;
  double m_fMin;
  double m_fMax;
  Matrix m_forceGenMat;
};
}  // namespace contacts
}  // namespace wbc

#endif  // ifndef __invdyn_contact_point_hpp__

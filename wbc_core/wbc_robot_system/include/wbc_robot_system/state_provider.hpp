/**
 * @file wbc_core/wbc_robot_system/include/wbc_robot_system/state_provider.hpp
 * @brief Doxygen documentation for state_provider module.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace wbc {

/**
 * @brief Optional bundle of external desired values written by non-RT callers
 *        (controllers, teleoperation) and read by state machines in OneStep().
 *
 * All fields are optional — set only what is needed.
 * Thread safety is the caller's responsibility (RT safety deferred to later).
 */
struct TaskInput {
  std::optional<Eigen::Vector3d>    x_des;            ///< EE position (world or reference_frame)
  std::optional<Eigen::Quaterniond> quat_des;         ///< EE orientation
  std::optional<Eigen::VectorXd>    q_des;            ///< Joint position
  std::optional<Eigen::Vector3d>    com_pos_des;      ///< CoM position
  std::optional<Eigen::VectorXd>    wrench_des;       ///< Task-space wrench [fx,fy,fz,tx,ty,tz]
  std::optional<double>             traj_duration;    ///< Trajectory duration override (seconds)
  std::string                        reference_frame;  ///< "" = world frame
};

/**
 * @brief Actuated joint snapshot used on the control-critical path.
 *
 * @details
 * This container is intentionally minimal:
 * - `q`: actuated joint positions
 * - `qdot`: actuated joint velocities
 * - `tau`: measured/estimated joint torques
 *
 * It is typically produced by a controller from hardware state interfaces and
 * passed directly to `ControlArchitecture::Update(...)`.
 */
struct RobotJointState {
  Eigen::VectorXd q;
  Eigen::VectorXd qdot;
  Eigen::VectorXd tau;

  /**
   * @brief Reset all vectors to zero with a fixed DoF size.
   * @param dof Number of actuated DoF.
   */
  void Reset(Eigen::Index dof) {
    q.setZero(dof);
    qdot.setZero(dof);
    tau.setZero(dof);
  }
};

/**
 * @brief Floating-base estimate snapshot for whole-body control.
 *
 * @details
 * This structure holds estimator outputs (pose/velocity/orientation frame
 * conversion) and is consumed by
 * `ControlArchitecture::Update(joint, base, t, dt)` on floating-base robots.
 */
struct RobotBaseState {
  Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d lin_vel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ang_vel{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d rot_world_local{Eigen::Matrix3d::Identity()};

  /**
   * @brief Assign from an incoming estimate and normalize the quaternion.
   *
   * If the quaternion is near-zero (degenerate estimator output), falls back
   * to Reset() so downstream code always sees a valid rotation matrix.
   */
  void SetAndNormalize(const RobotBaseState& src) {
    *this = src;
    if (quat.squaredNorm() > std::numeric_limits<double>::epsilon()) {
      quat.normalize();
      rot_world_local = quat.toRotationMatrix();
    } else {
      Reset();
    }
  }

  /**
   * @brief Reset to identity/zero base state.
   */
  void Reset() {
    pos.setZero();
    quat.setIdentity();
    lin_vel.setZero();
    ang_vel.setZero();
    rot_world_local.setIdentity();
  }
};

/**
 * @brief Shared runtime context container used by FSM and architecture.
 *
 * @details
 * `StateProvider` is a lightweight state/context bus:
 * - Time/FSM bookkeeping (`current_time_`, `state_`, ...)
 * - Nominal joint posture (`nominal_jpos_`) populated each tick
 * - Estimated base state (`base_state_`) for floating-base robots
 * - Optional per-link contact flags/wrenches
 *
 * External desired values (teleop commands) are NOT stored here — each state
 * manages its own inputs via `StateMachine::SetExternalInput()`.
 *
 * It is not the owner of robot model state (`q`, `qdot`) in the direct joint
 * path. Joint states should travel through `RobotJointState` into
 * `ControlArchitecture::Update(...)`.
 */
class StateProvider {
public:
  explicit StateProvider(double dt = 0.001)
      : servo_dt_(dt),
        current_time_(0.0),
        count_(0),
        state_(0),
        prev_state_(0) {}
  virtual ~StateProvider() = default;

  // ---------------------------------------------------------------------------
  // 1. Common System Info
  // ---------------------------------------------------------------------------
  double servo_dt_;
  double current_time_;
  std::uint64_t count_; // Control tick count

  // ---------------------------------------------------------------------------
  // 2. Common Robot Context
  // ---------------------------------------------------------------------------
  Eigen::VectorXd nominal_jpos_; ///< Initial or nominal joint posture captured at startup

  // ---------------------------------------------------------------------------
  // 3. FSM State Management
  // ---------------------------------------------------------------------------
  int state_;       // Current State ID
  int prev_state_;  // Previous State ID

  // ---------------------------------------------------------------------------
  // 4. Shared Estimated Base State
  // ---------------------------------------------------------------------------
  bool is_floating_base_{false};
  RobotBaseState base_state_;

  // ---------------------------------------------------------------------------
  // 5. Per-link Contact State
  // Register and update through methods below.
  // ---------------------------------------------------------------------------

  /**
   * @brief Register one contact slot during non-RT initialization.
   * @param link_name Link identifier key.
   * @param wrench_dim Wrench vector dimension (default 6).
   * @return true when new slot was inserted, false on invalid input.
   */
  bool RegisterContactLink(const std::string& link_name,
                           Eigen::Index wrench_dim = 6) {
    if (link_name.empty() || wrench_dim <= 0) {
      return false;
    }

    // std::map has stable iterators/pointers — no rehash concern.
    auto [state_it, state_inserted] =
        link_contact_state_.emplace(link_name, false);
    auto [wrench_it, wrench_inserted] =
        link_contact_wrench_.emplace(link_name, Eigen::VectorXd::Zero(wrench_dim));
    if (!wrench_inserted && wrench_it->second.size() != wrench_dim) {
      wrench_it->second.setZero(wrench_dim);
    }

    auto index_it = link_contact_index_.find(link_name);
    if (index_it == link_contact_index_.end()) {
      const std::size_t index = contact_state_slots_.size();
      link_contact_index_.emplace(link_name, index);
      contact_state_slots_.push_back(nullptr);
      contact_wrench_slots_.push_back(nullptr);
    }

    RebindContactSlots();

    return state_inserted || wrench_inserted;
  }

  /**
   * @brief Resolve contact slot index once (non-RT usage).
   * @return Index in [0, NumRegisteredContactLinks), or -1 if not registered.
   */
  int ContactIndex(const std::string& link_name) const {
    const auto it = link_contact_index_.find(link_name);
    if (it == link_contact_index_.end()) {
      return -1;
    }
    return static_cast<int>(it->second);
  }

  /**
   * @brief Number of registered contact slots.
   */
  std::size_t NumRegisteredContactLinks() const {
    return contact_state_slots_.size();
  }

  /**
   * @brief RT-safe overwrite of a registered contact state.
   * @return false if the link was not pre-registered.
   */
  bool SetContactStateIfRegistered(const std::string& link_name,
                                   bool in_contact) {
    const auto index_it = link_contact_index_.find(link_name);
    if (index_it == link_contact_index_.end()) {
      return false;
    }
    return SetContactStateIfRegistered(index_it->second, in_contact);
  }

  /**
   * @brief RT-safe overwrite of a registered contact state by slot index.
   * @return false when index is out-of-range.
   */
  bool SetContactStateIfRegistered(std::size_t contact_index, bool in_contact) {
    if (contact_index >= contact_state_slots_.size() ||
        contact_state_slots_[contact_index] == nullptr) {
      return false;
    }
    *contact_state_slots_[contact_index] = in_contact;
    return true;
  }

  /**
   * @brief RT-safe overwrite of a registered contact wrench.
   * @return false if link is unknown or wrench dimension mismatches.
   */
  bool SetContactWrenchIfRegistered(const std::string& link_name,
                                    const Eigen::Ref<const Eigen::VectorXd>& wrench) {
    const auto index_it = link_contact_index_.find(link_name);
    if (index_it == link_contact_index_.end()) {
      return false;
    }
    return SetContactWrenchIfRegistered(index_it->second, wrench);
  }

  /**
   * @brief RT-safe overwrite of a registered contact wrench by slot index.
   * @return false when index is invalid or wrench dimension mismatches.
   */
  bool SetContactWrenchIfRegistered(
      std::size_t contact_index,
      const Eigen::Ref<const Eigen::VectorXd>& wrench) {
    if (contact_index >= contact_wrench_slots_.size() ||
        contact_wrench_slots_[contact_index] == nullptr) {
      return false;
    }
    Eigen::VectorXd* slot = contact_wrench_slots_[contact_index];
    if (slot->size() != wrench.size()) {
      return false;
    }
    *slot = wrench;
    return true;
  }

private:
  // Contact containers are private so registration/update invariants are
  // enforced through RegisterContactLink/SetContact* APIs only.
  // std::map (not unordered_map) so element pointers in contact_*_slots_
  // remain stable across insertions (no rehash invalidation).
  std::map<std::string, bool> link_contact_state_;
  std::map<std::string, Eigen::VectorXd> link_contact_wrench_;
  std::map<std::string, std::size_t> link_contact_index_;
  std::vector<bool*> contact_state_slots_;
  std::vector<Eigen::VectorXd*> contact_wrench_slots_;

  // Rebind slot pointers after any map insertion/rehash (non-RT path only).
  void RebindContactSlots() {
    for (const auto& [name, index] : link_contact_index_) {
      if (index >= contact_state_slots_.size() ||
          index >= contact_wrench_slots_.size()) {
        continue;
      }
      const auto state_it = link_contact_state_.find(name);
      contact_state_slots_[index] =
          (state_it != link_contact_state_.end()) ? &state_it->second : nullptr;
      const auto wrench_it = link_contact_wrench_.find(name);
      contact_wrench_slots_[index] =
          (wrench_it != link_contact_wrench_.end()) ? &wrench_it->second : nullptr;
    }
  }

};

} // namespace wbc

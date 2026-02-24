#pragma once

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h> // YAML 설정을 위해 추가

#include "wbc_robot_system/state_provider.hpp"
#include "wbc_formulation/interface/task.hpp"
#include "wbc_util/constraint_registry.hpp"
#include "wbc_util/task_registry.hpp"

namespace wbc {

class PinocchioRobotSystem;
using StateId = int;

class StateMachine {
public:
  // 생성자: Registry들을 직접 주입받아 "Manager-less" 구현을 지원
  StateMachine(StateId state_id, const std::string& state_name,
               PinocchioRobotSystem* robot, 
               TaskRegistry* task_reg,
               ConstraintRegistry* const_reg,
               StateProvider* state_provider = nullptr)
      : state_id_(state_id), state_name_(state_name), robot_(robot),
        task_registry_(task_reg), constraint_registry_(const_reg),
        sp_(state_provider),
        duration_(3.0), wait_time_(1.0), next_state_id_(state_id),
        stay_here_(false),
        start_time_(-1.0), current_time_(0.0) {} // start_time 초기화

  virtual ~StateMachine() = default;

  // [Lifecycle Interface] FSMHandler가 호출
  virtual void FirstVisit() = 0;
  virtual void OneStep() = 0;
  virtual void LastVisit() = 0;
  virtual bool EndOfState() {
    if (stay_here_) {
      return false;
    }
    return current_time_ >= (duration_ + wait_time_);
  }
  virtual StateId GetNextState() { return next_state_id_; }

  // [Parameter Interface] 공통 파라미터는 베이스에서 파싱
  virtual void SetParameters(const YAML::Node& node) {
    const YAML::Node params = ResolveParamsNode(node);
    if (!params) {
      return;
    }
    const bool has_next_state_id = static_cast<bool>(params["next_state_id"]);
    const bool has_b_stay_here = static_cast<bool>(params["b_stay_here"]);
    const bool has_stay_here = static_cast<bool>(params["stay_here"]);

    if (params["duration"]) {
      duration_ = params["duration"].as<double>();
    }
    if (params["wait_time"]) {
      wait_time_ = params["wait_time"].as<double>();
    }
    if (has_next_state_id) {
      next_state_id_ = params["next_state_id"].as<StateId>();
    }

    if (has_b_stay_here && has_stay_here) {
      const bool b_stay = params["b_stay_here"].as<bool>();
      const bool stay = params["stay_here"].as<bool>();
      if (b_stay != stay) {
        throw std::runtime_error(
            "[StateMachine] Both 'b_stay_here' and 'stay_here' are set with "
            "different values in state '" +
            state_name_ + "'.");
      }
      stay_here_ = b_stay;
    } else if (has_b_stay_here) {
      stay_here_ = params["b_stay_here"].as<bool>();
    } else if (has_stay_here) {
      stay_here_ = params["stay_here"].as<bool>();
    }

    if (stay_here_ && has_next_state_id) {
      std::cerr << "[StateMachine] State '" << state_name_
                << "' has b_stay_here=true and next_state_id set. "
                   "next_state_id will be ignored while stay_here is true."
                << std::endl;
    }
    if (duration_ < 0.0 || wait_time_ < 0.0) {
      throw std::runtime_error(
          "[StateMachine] duration and wait_time must be non-negative for "
          "state '" +
          state_name_ + "'.");
    }
  }

  // [Time Management]
  // 상태 진입 시점(전역 시간) 기록 (FSM Handler가 호출)
  void EnterState(double current_global_time) {
    start_time_ = current_global_time;
    current_time_ = 0.0;
    if (sp_ != nullptr) {
      sp_->current_time_ = current_global_time;
      sp_->prev_state_ = sp_->state_;
      sp_->state_ = state_id_;
      sp_->b_first_visit_ = true;
    }
  }

  // 매 틱마다 시간 업데이트 (FSM Handler가 OneStep 직전에 호출)
  void UpdateStateTime(double current_global_time) {
    if (start_time_ < 0.0) {
        // 방어 코드: FirstVisit이 안 불렸는데 OneStep이 불린 경우
        start_time_ = current_global_time;
    }
    current_time_ = current_global_time - start_time_;
    if (sp_ != nullptr) {
      sp_->current_time_ = current_global_time;
      sp_->state_ = state_id_;
      sp_->b_first_visit_ = false;
    }
  }

  // Getters
  StateId id() const { return state_id_; }
  const std::string& name() const { return state_name_; }
  double elapsed_time() const { return current_time_; }
  StateProvider* state_provider() const { return sp_; }

  // [Assignment Interface] YAML state recipe가 지정한 task들을
  // "name -> task" 맵으로 보관
  void AssignTask(const std::string& task_name, Task* task) {
    if (task_name.empty() || task == nullptr) {
      return;
    }
    assigned_tasks_[task_name] = task;
  }

  void AssignForceTask(const std::string& task_name, ForceTask* task) {
    if (task_name.empty() || task == nullptr) {
      return;
    }
    assigned_force_tasks_[task_name] = task;
  }

  void AssignContact(const std::string& contact_name, Contact* contact) {
    if (contact_name.empty() || contact == nullptr) {
      return;
    }
    assigned_contacts_[contact_name] = contact;
  }

  void AssignConstraint(const std::string& constraint_name,
                        Constraint* constraint) {
    if (constraint_name.empty() || constraint == nullptr) {
      return;
    }
    assigned_constraints_[constraint_name] = constraint;
  }

  // [Typed Access] 이름으로 찾고 타입으로 캐스팅
  template <typename TaskType>
  TaskType* GetTask(const std::string& task_name) const {
    auto it = assigned_tasks_.find(task_name);
    if (it == assigned_tasks_.end()) {
      return nullptr;
    }
    return dynamic_cast<TaskType*>(it->second);
  }

  ForceTask* GetForceTask(const std::string& task_name) const {
    auto it = assigned_force_tasks_.find(task_name);
    if (it == assigned_force_tasks_.end()) {
      return nullptr;
    }
    return it->second;
  }

  Contact* GetContact(const std::string& contact_name) const {
    auto it = assigned_contacts_.find(contact_name);
    if (it == assigned_contacts_.end()) {
      return nullptr;
    }
    return it->second;
  }

  Constraint* GetConstraint(const std::string& constraint_name) const {
    auto it = assigned_constraints_.find(constraint_name);
    if (it == assigned_constraints_.end()) {
      return nullptr;
    }
    return it->second;
  }

  // [Safety Helpers] "Fail Fast" - 없는 태스크를 찾으면 즉시 에러 발생
  // 자식 클래스에서: auto* task = RequireMotionTask("jpos_task");
  Task* RequireMotionTask(const std::string& task_name) const {
    Task* task = FindAssigned(assigned_tasks_, task_name);
    if (task == nullptr && task_registry_ != nullptr) {
      task = task_registry_->GetMotionTask(task_name);
    }
    if (task == nullptr) {
      throw MissingEntityError("Motion Task", task_name);
    }
    return task;
  }

  template <typename TaskType>
  TaskType* RequireTaskAs(const std::string& task_name) const {
    TaskType* typed_task =
        dynamic_cast<TaskType*>(FindAssigned(assigned_tasks_, task_name));
    if (typed_task == nullptr && task_registry_ != nullptr) {
      typed_task =
          dynamic_cast<TaskType*>(task_registry_->GetMotionTask(task_name));
    }
    if (typed_task == nullptr) {
      throw MissingEntityError("Typed Motion Task", task_name);
    }
    return typed_task;
  }

  ForceTask* RequireForceTask(const std::string& task_name) const {
    ForceTask* task = FindAssigned(assigned_force_tasks_, task_name);
    if (task == nullptr && task_registry_ != nullptr) {
      task = task_registry_->GetForceTask(task_name);
    }
    if (task == nullptr) {
      throw MissingEntityError("Force Task", task_name);
    }
    return task;
  }

  Constraint* RequireConstraint(const std::string& constraint_name) const {
    Constraint* constraint = FindAssigned(assigned_constraints_, constraint_name);
    if (constraint == nullptr && constraint_registry_ != nullptr) {
      constraint = constraint_registry_->GetConstraint(constraint_name);
    }
    if (constraint == nullptr) {
      throw MissingEntityError("Constraint", constraint_name);
    }
    return constraint;
  }

  Contact* RequireContact(const std::string& contact_name) const {
    Contact* contact = FindAssigned(assigned_contacts_, contact_name);
    if (contact == nullptr && constraint_registry_ != nullptr) {
      contact = constraint_registry_->GetContact(contact_name);
    }
    if (contact == nullptr) {
      throw MissingEntityError("Contact", contact_name);
    }
    return contact;
  }

protected:
  // 상태 정보
  StateId state_id_;
  std::string state_name_;

  // 로봇 및 도구함 (자식들이 직접 사용)
  PinocchioRobotSystem* robot_;
  TaskRegistry* task_registry_;
  ConstraintRegistry* constraint_registry_;
  StateProvider* sp_;
  std::unordered_map<std::string, Task*> assigned_tasks_;
  std::unordered_map<std::string, ForceTask*> assigned_force_tasks_;
  std::unordered_map<std::string, Contact*> assigned_contacts_;
  std::unordered_map<std::string, Constraint*> assigned_constraints_;

  // 공통 상태 파라미터
  double duration_;
  double wait_time_;
  StateId next_state_id_;
  bool stay_here_;

  // 시간 관리
  double start_time_;   // 상태 진입 시점의 전역 시간
  double current_time_; // 상태 진입 후 경과 시간 (Trajectory Interpolation용)

  static YAML::Node ResolveParamsNode(const YAML::Node& node) {
    if (!node) {
      return YAML::Node();
    }
    return node["params"] ? node["params"] : node;
  }

  template <typename MapT>
  static typename MapT::mapped_type FindAssigned(const MapT& assigned,
                                                 const std::string& key) {
    const auto it = assigned.find(key);
    if (it == assigned.end()) {
      return nullptr;
    }
    return it->second;
  }

  std::runtime_error MissingEntityError(const std::string& entity_kind,
                                        const std::string& entity_name) const {
    return std::runtime_error("[StateMachine] State '" + state_name_ +
                              "' (ID: " + std::to_string(state_id_) +
                              ") requires missing " + entity_kind + ": '" +
                              entity_name + "'");
  }
};

} // namespace wbc

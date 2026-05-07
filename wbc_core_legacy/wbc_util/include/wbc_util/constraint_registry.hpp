/**
 * @file wbc_core/wbc_util/include/wbc_util/constraint_registry.hpp
 * @brief Doxygen documentation for constraint_registry module.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace wbc {
// 전방 선언: 클래스의 실체는 .cpp에서 include 합니다.
class Constraint;
class Contact;
} // namespace wbc

namespace wbc {

/**
 * @brief Ownership registry for named constraints/contacts.
 */
class ConstraintRegistry {
public:
  ConstraintRegistry() = default;
  ~ConstraintRegistry() = default;

  /**
   * @brief Add or replace a constraint entry.
   */
  void AddConstraint(const std::string& name, std::unique_ptr<Constraint> constraint);
  
  // 단순 포인터 반환은 전방 선언만으로 가능
  Constraint* GetConstraint(const std::string& name) const;

  /**
   * @brief Return as `Contact*` when the named constraint is a contact.
   */
  Contact* GetContact(const std::string& name) const;

  const std::unordered_map<std::string, std::unique_ptr<Constraint>>&
  GetConstraints() const {
    return constraint_map_;
  }

  void Clear();

private:
  std::unordered_map<std::string, std::unique_ptr<Constraint>> constraint_map_;
};

} // namespace wbc

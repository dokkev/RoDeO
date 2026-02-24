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

class ConstraintRegistry {
public:
  ConstraintRegistry() = default;
  ~ConstraintRegistry() = default;

  void AddConstraint(const std::string& name, std::unique_ptr<Constraint> constraint);
  
  // 단순 포인터 반환은 전방 선언만으로 가능
  Constraint* GetConstraint(const std::string& name) const;

  /**
   * @brief Contact 타입으로 캐스팅하여 반환
   * @note 이 함수의 구현부는 상속 관계를 알아야 하므로 .cpp 파일에 두는 것이 좋습니다.
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

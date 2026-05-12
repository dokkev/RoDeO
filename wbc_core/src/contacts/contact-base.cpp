//
// Copyright (c) 2017 CNRS
//

#include "wbc_core/contacts/contact-base.hpp"

namespace wbc {
namespace contacts {
ContactBase::ContactBase(const std::string& name, RobotSystem& robot)
    : m_name(name), m_robot(robot) {}

const std::string& ContactBase::name() const { return m_name; }

void ContactBase::name(const std::string& name) { m_name = name; }

}  // namespace contacts
}  // namespace wbc

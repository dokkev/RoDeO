/**
 * @file draco_controller/test/test_draco_controller_unit.cpp
 * @brief Unit tests for draco_controller state machines.
 */
#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "wbc_architecture/control_architecture.hpp"
#include "wbc_fsm/state_factory.hpp"
#include "wbc_robot_system/pinocchio_robot_system.hpp"

// Force-link the draco state machine registrations.
#include "draco_controller/state_machines/initialize.hpp"
#include "draco_controller/state_machines/balance.hpp"

namespace wbc {
namespace {

TEST(DracoControllerUnit, StateFactoryRegistration) {
  // Verify draco state machine types are registered.
  EXPECT_TRUE(StateFactory::Instance().Has("draco_initialize"));
  EXPECT_TRUE(StateFactory::Instance().Has("draco_balance"));
  EXPECT_TRUE(StateFactory::Instance().Has("draco_home"));
}

}  // namespace
}  // namespace wbc

cmake_minimum_required(VERSION 3.8)
project(wbc_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)
find_package(eiquadprog REQUIRED)
find_package(ament_index_cpp REQUIRED)

set(wbc_solver_sources
  src/solvers/solver-HQP-base.cpp
  src/solvers/solver-HQP-factory.cpp
  src/solvers/solver-HQP-cascade.cpp
  src/solvers/solver-HQP-eiquadprog.cpp
  src/solvers/solver-HQP-eiquadprog-fast.cpp
  src/solvers/solver-qp-params.cpp
  src/solvers/utils.cpp
  src/solvers/solver-proxqp.cpp
)

set(${PROJECT_NAME}_SOURCES
  src/utils/stop-watch.cpp
  src/utils/math_util.cpp
  src/math/constraint-base.cpp
  src/math/constraint-equality.cpp
  src/math/constraint-inequality.cpp
  src/math/constraint-bound.cpp
  src/math/utils.cpp
  src/tasks/task-base.cpp
  src/tasks/task-motion.cpp
  src/tasks/task-actuation.cpp
  src/tasks/task-contact-force.cpp
  src/tasks/task-com-equality.cpp
  src/tasks/task-se3-equality.cpp
  src/tasks/task-contact-force-equality.cpp
  src/tasks/task-cop-equality.cpp
  src/tasks/task-actuation-equality.cpp
  src/tasks/task-actuation-bounds.cpp
  src/tasks/task-joint-bounds.cpp
  src/tasks/task-joint-posture.cpp
  src/tasks/task-joint-posVelAcc-bounds.cpp
  src/tasks/task-capture-point-inequality.cpp
  src/tasks/task-angular-momentum-equality.cpp
  src/tasks/task-two-frames-equality.cpp
  src/contacts/contact-base.cpp
  src/contacts/contact-6d.cpp
  src/contacts/contact-point.cpp
  src/contacts/measured-force-base.cpp
  src/contacts/measured-3d-force.cpp
  src/contacts/measured-6d-wrench.cpp
  src/contacts/contact-two-frame-positions.cpp
  src/trajectories/trajectory-se3.cpp
  src/trajectories/trajectory-euclidian.cpp
  src/trajectories/interpolation.cpp
  src/trajectories/joint_integrator.cpp
  ${wbc_solver_sources}
  src/robots/robot-system.cpp
  src/contacts/contact-level.cpp
  src/controller/id-hqp.cpp
  src/utils/ros_path_utils.cpp
)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

set(proxsuite_interface proxsuite::proxsuite)
if(TARGET proxsuite::proxsuite-vectorized)
  set(proxsuite_interface proxsuite::proxsuite-vectorized)
endif()

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
    ${proxsuite_interface}
    eiquadprog::eiquadprog
  PRIVATE
    ament_index_cpp::ament_index_cpp
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Path to local TSID model files (romeo, quadruped).
  set(TSID_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tsid/models")

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_robot_system_state_contract
    test/test_robot_system_state_contract.cpp)
  target_link_libraries(test_robot_system_state_contract ${PROJECT_NAME})
  target_compile_definitions(test_robot_system_state_contract PRIVATE
    TSID_MODEL_DIR="${TSID_MODEL_DIR}")
  set_tests_properties(test_robot_system_state_contract PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")

  ament_add_gtest(test_id_hqp test/test-id-hqp.cpp)
  target_link_libraries(test_id_hqp ${PROJECT_NAME})
  target_compile_definitions(test_id_hqp PRIVATE
    TSID_MODEL_DIR="${TSID_MODEL_DIR}")
  set_tests_properties(test_id_hqp PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")

  ament_add_gtest(test_wbc_solver_semantics test/test-wbc-solver-semantics.cpp)
  target_link_libraries(test_wbc_solver_semantics ${PROJECT_NAME})
  target_compile_definitions(test_wbc_solver_semantics PRIVATE
    TSID_MODEL_DIR="${TSID_MODEL_DIR}")
  set_tests_properties(test_wbc_solver_semantics PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")

  ament_add_gtest(test_solver_baselines test/test-solver-baselines.cpp)
  target_link_libraries(test_solver_baselines ${PROJECT_NAME})
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
set(wbc_export_dependencies
  eigen3_cmake_module
  Eigen3
  pinocchio
  proxsuite
  eiquadprog
  ament_index_cpp
)
ament_export_dependencies(${wbc_export_dependencies})

ament_package()

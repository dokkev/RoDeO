cmake_minimum_required(VERSION 3.8)
project(rpc_example)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(wbc_architecture REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_util REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/ExampleController.cpp
  src/example_state_provider.cpp
  src/state_machines/initialize.cpp
  src/state_machines/home.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
)

ament_target_dependencies(${PROJECT_NAME}
  wbc_architecture
  wbc_fsm
  wbc_formulation
  wbc_robot_system
  wbc_trajectory
  wbc_util
)

add_executable(example_controller_rbd_sim
  src/example_controller_rbd_sim.cpp
)
target_link_libraries(example_controller_rbd_sim
  ${PROJECT_NAME}
  Eigen3::Eigen
)
ament_target_dependencies(example_controller_rbd_sim
  ament_index_cpp
  pinocchio
  rclcpp
  sensor_msgs
)

add_executable(example_controller_wbc_contact_sim
  src/example_controller_wbc_contact_sim.cpp
)
target_link_libraries(example_controller_wbc_contact_sim
  ${PROJECT_NAME}
  Eigen3::Eigen
)
ament_target_dependencies(example_controller_wbc_contact_sim
  ament_index_cpp
  pinocchio
  rclcpp
  sensor_msgs
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY description/
  DESTINATION share/${PROJECT_NAME}/description
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(PROGRAMS
  scripts/pinocchio_rbd_visualization.py
  scripts/pinocchio_draco3_contact_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS example_controller_rbd_sim
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS example_controller_wbc_contact_sim
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_example_state_provider
    test/test_example_state_provider.cpp
  )
  if(TARGET test_example_state_provider)
    target_link_libraries(test_example_state_provider
      ${PROJECT_NAME}
      Eigen3::Eigen
    )
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  eigen3_cmake_module
  Eigen3
  ament_index_cpp
  pinocchio
  rclcpp
  sensor_msgs
  wbc_architecture
  wbc_fsm
  wbc_formulation
  wbc_robot_system
  wbc_trajectory
  wbc_util
)

ament_package()

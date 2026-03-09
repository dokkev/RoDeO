cmake_minimum_required(VERSION 3.8)
project(optimo_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(wbc_architecture REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_handlers REQUIRED)
find_package(wbc_msgs REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_solver REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_util REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/optimo_controller.cpp
  src/state_machines/cartesian_teleop.cpp
  src/state_machines/initialize.cpp
  src/state_machines/joint_teleop.cpp
  src/state_machines/sys_id.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  controller_interface
  geometry_msgs
  hardware_interface
  pinocchio
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  wbc_architecture
  wbc_formulation
  wbc_fsm
  wbc_handlers
  wbc_msgs
  wbc_robot_system
  wbc_solver
  wbc_trajectory
  wbc_util
)
target_link_libraries(${PROJECT_NAME} yaml-cpp)

pluginlib_export_plugin_description_file(
  controller_interface
  optimo_controller_plugin.xml
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

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(FILES optimo_controller_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  controller_interface
  geometry_msgs
  hardware_interface
  pinocchio
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  wbc_architecture
  wbc_formulation
  wbc_fsm
  wbc_handlers
  wbc_msgs
  wbc_robot_system
  wbc_solver
  wbc_trajectory
  wbc_util
)

ament_package()

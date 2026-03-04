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
find_package(wbc_robot_system REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_util REQUIRED)
find_package(pinocchio REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/optimo_controller.cpp
  src/state_machines/initialize.cpp
  src/state_machines/joint_teleop.cpp
  src/state_machines/ee_teleop.cpp
  src/manipulability_handler.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}
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
  wbc_robot_system
  wbc_trajectory
  wbc_util
)

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
  find_package(ament_cmake_gtest REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_optimo_controller_unit
    test/test_optimo_controller_unit.cpp
  )
  if(TARGET test_optimo_controller_unit)
    target_link_libraries(test_optimo_controller_unit
      ${PROJECT_NAME}
    )
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
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
  wbc_robot_system
  wbc_trajectory
  wbc_util
)

ament_package()

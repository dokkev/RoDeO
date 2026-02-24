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
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_solver REQUIRED)
find_package(wbc_architecture REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_util REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/optimo_wbc_controller.cpp
  src/optimo_state_provider.cpp
  src/state_machines/initialize.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  yaml-cpp
)
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  controller_interface
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  wbc_formulation
  wbc_fsm
  wbc_robot_system
  wbc_solver
  wbc_architecture
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
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  controller_interface
  eigen3_cmake_module
  Eigen3
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  yaml-cpp
  wbc_formulation
  wbc_fsm
  wbc_robot_system
  wbc_solver
  wbc_architecture
  wbc_trajectory
  wbc_util
)

ament_package()

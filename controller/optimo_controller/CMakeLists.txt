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
find_package(control_architecture REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(wbc_core REQUIRED)
find_package(wbc_ros REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/optimo_controller.cpp
  src/state_machines/cartesian_reference.cpp
  src/state_machines/initialize.cpp
  src/state_machines/joint_reference.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}
  control_architecture
  controller_interface
  pinocchio
  pluginlib
  wbc_core
  wbc_ros
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

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  control_architecture
  controller_interface
  pinocchio
  pluginlib
  wbc_core
  wbc_ros
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(plato_mppi)

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
find_package(geometry_msgs REQUIRED)
find_package(mppi_core REQUIRED)
find_package(pinocchio REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sdr_grasp_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(wbc_core REQUIRED)
find_package(wbc_msgs REQUIRED)

add_executable(jenga_grasp_mppi_node
  src/jenga_grasp_mppi_node.cpp
)
target_compile_features(jenga_grasp_mppi_node PUBLIC cxx_std_17)
target_link_libraries(jenga_grasp_mppi_node
  Eigen3::Eigen
  pinocchio::pinocchio
)
ament_target_dependencies(jenga_grasp_mppi_node
  ament_index_cpp
  geometry_msgs
  mppi_core
  pinocchio
  rclcpp
  sdr_grasp_msgs
  sensor_msgs
  wbc_core
  wbc_msgs
)

install(TARGETS jenga_grasp_mppi_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

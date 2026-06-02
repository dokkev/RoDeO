cmake_minimum_required(VERSION 3.8)
project(dynamixel_interface LANGUAGES CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(dynamixel_topic_controller
  src/dynamixel_topic_controller.cpp
)
target_compile_features(dynamixel_topic_controller PUBLIC cxx_std_17)
ament_target_dependencies(dynamixel_topic_controller
  rclcpp
  std_msgs
)
target_link_libraries(dynamixel_topic_controller
  dynamixel_sdk::dynamixel_sdk
)

install(TARGETS dynamixel_topic_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(dynamixel_sdk rclcpp std_msgs)
ament_package()

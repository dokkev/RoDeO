cmake_minimum_required(VERSION 3.16)
project(joint_impedance_controller)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=conversion -Werror=unused-but-set-variable
                      -Werror=return-type -Werror=shadow -Werror=format -Werror=range-loop-construct
                      -Werror=missing-braces)
endif()

# Using this instead of visibility macros
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  controller_interface
  generate_parameter_library
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
  std_msgs
  wbc_msgs
)

# Add this dependency
find_package(generate_parameter_library REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(backward_ros REQUIRED)
find_package(yaml-cpp REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

# Generate parameter library
generate_parameter_library(joint_impedance_controller_parameters
  src/joint_impedance_controller_parameters.yaml
)

# Add the library
add_library(joint_impedance_controller SHARED
  src/joint_impedance_controller.cpp
)
target_compile_features(joint_impedance_controller PUBLIC cxx_std_17)
target_include_directories(joint_impedance_controller PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/joint_impedance_controller>
)

target_link_libraries(joint_impedance_controller PUBLIC 
  joint_impedance_controller_parameters
)
ament_target_dependencies(joint_impedance_controller PUBLIC ${THIS_PACKAGE_INCLUDE_DEPENDS})

pluginlib_export_plugin_description_file(controller_interface joint_impedance_controller_plugin.xml)

add_library(joint_impedance_controller_impedance_handler SHARED
  src/impedance_handler.cpp
)
target_compile_features(joint_impedance_controller_impedance_handler PUBLIC cxx_std_17)
target_include_directories(joint_impedance_controller_impedance_handler PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/joint_impedance_controller>
)
target_link_libraries(joint_impedance_controller_impedance_handler yaml-cpp)

add_executable(impedance_trajectory_controller_node
  src/impedance_trajectory_controller_node.cpp
  src/impedance_trajectory_controller.cpp
)
target_compile_features(impedance_trajectory_controller_node PUBLIC cxx_std_17)
target_include_directories(impedance_trajectory_controller_node PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/joint_impedance_controller>
)
target_link_libraries(impedance_trajectory_controller_node
  joint_impedance_controller_impedance_handler
)
ament_target_dependencies(impedance_trajectory_controller_node
  ament_index_cpp
  rclcpp
  std_msgs
  sensor_msgs
  wbc_msgs
)


if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)

  ament_add_gtest(joint_impedance_trajectory_controller_test
    test/joint_impedance_trajectory_controller_test.cpp
    src/impedance_trajectory_controller.cpp
  )
  if(TARGET joint_impedance_trajectory_controller_test)
    target_include_directories(joint_impedance_trajectory_controller_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(joint_impedance_trajectory_controller_test PUBLIC cxx_std_17)
  endif()

  ament_add_gtest(joint_impedance_handler_test
    test/impedance_handler_test.cpp
  )
  if(TARGET joint_impedance_handler_test)
    target_include_directories(joint_impedance_handler_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(joint_impedance_handler_test
      joint_impedance_controller_impedance_handler
    )
    target_compile_features(joint_impedance_handler_test PUBLIC cxx_std_17)
  endif()

endif()

install(
  DIRECTORY include/
  DESTINATION include/joint_impedance_controller
)
install(
  TARGETS
    joint_impedance_controller
    joint_impedance_controller_impedance_handler
    joint_impedance_controller_parameters
  EXPORT export_joint_impedance_controller
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib

)
install(TARGETS impedance_trajectory_controller_node
  DESTINATION lib/${PROJECT_NAME}
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}/
  )
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

ament_export_targets(export_joint_impedance_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()

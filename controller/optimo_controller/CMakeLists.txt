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
find_package(wbc_logger REQUIRED)
find_package(wbc_msgs REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_handlers REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_util REQUIRED)
find_package(pinocchio REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/optimo_controller.cpp
  src/state_machines/initialize.cpp
  src/state_machines/joint_teleop.cpp
  src/state_machines/cartesian_teleop.cpp
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
  wbc_logger
  wbc_msgs
  wbc_fsm
  wbc_handlers
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

  ament_add_gtest(test_optimo_controller_service
    test/test_optimo_controller_service.cpp
    TIMEOUT 30
  )
  if(TARGET test_optimo_controller_service)
    target_link_libraries(test_optimo_controller_service
      -Wl,--no-as-needed
      ${PROJECT_NAME}
      -Wl,--as-needed
    )
    ament_target_dependencies(test_optimo_controller_service
      rclcpp
      wbc_msgs
    )
  endif()

  find_package(launch_testing_ament_cmake QUIET)
  find_package(mujoco_vendor QUIET)

  if(launch_testing_ament_cmake_FOUND AND mujoco_vendor_FOUND)
    add_launch_test(
      test/test_ros2_integration.py
      TIMEOUT 120
    )
  endif()

  if(mujoco_vendor_FOUND)
    ament_add_gtest(test_sea
      test/test_sea.cpp
      TIMEOUT 120
    )
    if(TARGET test_sea)
      target_link_libraries(test_sea
        -Wl,--no-as-needed
        ${PROJECT_NAME}
        -Wl,--as-needed
        mujoco::mujoco
        Eigen3::Eigen
      )
      ament_target_dependencies(test_sea
        wbc_util
      )
    endif()

    ament_add_gtest(test_gain_tuning
      test/test_gain_tuning.cpp
      TIMEOUT 300
    )
    if(TARGET test_gain_tuning)
      target_link_libraries(test_gain_tuning
        -Wl,--no-as-needed
        ${PROJECT_NAME}
        -Wl,--as-needed
        mujoco::mujoco
        Eigen3::Eigen
      )
      ament_target_dependencies(test_gain_tuning
        wbc_architecture
        wbc_formulation
        wbc_robot_system
      )
    endif()

    ament_add_gtest(test_dummy_teleop
      test/test_dummy_teleop.cpp
      TIMEOUT 120
    )
    if(TARGET test_dummy_teleop)
      target_link_libraries(test_dummy_teleop
        -Wl,--no-as-needed
        ${PROJECT_NAME}
        -Wl,--as-needed
        mujoco::mujoco
        Eigen3::Eigen
      )
      ament_target_dependencies(test_dummy_teleop
        wbc_architecture
        wbc_formulation
        wbc_robot_system
      )
    endif()

    ament_add_gtest(test_domain_randomization
      test/test_domain_randomization.cpp
      TIMEOUT 600
    )
    if(TARGET test_domain_randomization)
      target_link_libraries(test_domain_randomization
        -Wl,--no-as-needed
        ${PROJECT_NAME}
        -Wl,--as-needed
        mujoco::mujoco
        Eigen3::Eigen
      )
      ament_target_dependencies(test_domain_randomization
        wbc_architecture
        wbc_formulation
        wbc_robot_system
      )
    endif()
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
  wbc_logger
  wbc_msgs
  wbc_fsm
  wbc_handlers
  wbc_robot_system
  wbc_trajectory
  wbc_util
)

ament_package()

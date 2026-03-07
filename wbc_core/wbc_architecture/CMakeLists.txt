cmake_minimum_required(VERSION 3.8)
project(wbc_architecture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_solver REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_logger REQUIRED)
find_package(wbc_util REQUIRED)
find_package(residual_compensator REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/control_architecture.cpp
  src/config_compiler.cpp
  src/runtime_config.cpp
  src/task_weight_scheduler.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  yaml-cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  wbc_formulation
  wbc_fsm
  wbc_logger
  wbc_robot_system
  wbc_solver
  wbc_trajectory
  wbc_util
  residual_compensator
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_wbc_config_compiler
    test/test_wbc_config_compiler.cpp
  )
  if(TARGET test_wbc_config_compiler)
    target_link_libraries(test_wbc_config_compiler
      ${PROJECT_NAME}
      Eigen3::Eigen
      yaml-cpp
    )
  endif()

  ament_add_gtest(test_control_architecture_behavior
    test/test_control_architecture_behavior.cpp
  )
  if(TARGET test_control_architecture_behavior)
    target_link_libraries(test_control_architecture_behavior
      ${PROJECT_NAME}
      Eigen3::Eigen
      yaml-cpp
    )
  endif()

  ament_add_gtest(test_review_improvements
    test/test_review_improvements.cpp
  )
  if(TARGET test_review_improvements)
    target_link_libraries(test_review_improvements
      ${PROJECT_NAME}
      Eigen3::Eigen
      yaml-cpp
    )
  endif()

  ament_add_gtest(test_cascade_pid
    test/test_cascade_pid.cpp
  )
  if(TARGET test_cascade_pid)
    target_link_libraries(test_cascade_pid
      ${PROJECT_NAME}
      Eigen3::Eigen
    )
  endif()

  ament_add_gtest(test_real_tasks
    test/test_real_tasks.cpp
  )
  if(TARGET test_real_tasks)
    target_link_libraries(test_real_tasks
      ${PROJECT_NAME}
      Eigen3::Eigen
      yaml-cpp
    )
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  eigen3_cmake_module
  Eigen3
  rclcpp
  yaml-cpp
  wbc_formulation
  wbc_fsm
  wbc_robot_system
  wbc_solver
  wbc_logger
  wbc_trajectory
  wbc_util
  residual_compensator
)

ament_package()

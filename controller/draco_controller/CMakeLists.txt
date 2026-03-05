cmake_minimum_required(VERSION 3.8)
project(draco_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(wbc_architecture REQUIRED)
find_package(wbc_formulation REQUIRED)
find_package(wbc_fsm REQUIRED)
find_package(wbc_robot_system REQUIRED)
find_package(wbc_trajectory REQUIRED)
find_package(wbc_handlers REQUIRED)
find_package(wbc_util REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/state_machines/initialize.cpp
  src/state_machines/balance.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}
  Eigen3
  wbc_architecture
  wbc_formulation
  wbc_fsm
  wbc_robot_system
  wbc_trajectory
  wbc_handlers
  wbc_util
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

  ament_add_gtest(test_draco_controller_unit
    test/test_draco_controller_unit.cpp
  )
  if(TARGET test_draco_controller_unit)
    # --no-as-needed forces the linker to load libdraco_controller.so even when
    # no symbols are explicitly referenced.  This is required so that the
    # WBC_REGISTER_STATE static initialisers in the .so fire at load time.
    target_link_libraries(test_draco_controller_unit
      -Wl,--no-as-needed
      ${PROJECT_NAME}
      -Wl,--as-needed
    )
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  wbc_architecture
  wbc_formulation
  wbc_fsm
  wbc_robot_system
  wbc_trajectory
  wbc_handlers
  wbc_util
)

ament_package()

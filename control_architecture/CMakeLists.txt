cmake_minimum_required(VERSION 3.8)
project(control_architecture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(wbc_core REQUIRED)

set(${PROJECT_NAME}_SOURCES
  src/runtime/yaml_parser.cpp
  src/runtime/config_loader.cpp
  src/runtime/config_compiler.cpp
  src/runtime/config_validator.cpp
  src/runtime/runtime_loader.cpp
  src/runtime/runtime_assembler.cpp
  src/runtime/state_machine_assembler.cpp
  src/state_machine/state_factory.cpp
  src/state_machine/state_util.cpp
  src/control_architecture.cpp
)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  Eigen3
  pinocchio
  wbc_core
)

target_link_libraries(${PROJECT_NAME} yaml-cpp)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  set(TSID_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tsid/models")

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_architecture test/test-architecture.cpp)
  target_link_libraries(test_architecture ${PROJECT_NAME})
  target_compile_definitions(test_architecture PRIVATE
    TSID_MODEL_DIR="${TSID_MODEL_DIR}")
  set_tests_properties(test_architecture PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  eigen3_cmake_module
  Eigen3
  pinocchio
  wbc_core
)

ament_package()

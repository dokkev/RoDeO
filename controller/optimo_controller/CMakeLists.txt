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

# Roboligent SDK cmake modules (FindEtherCAT, FindYaml) must be on
# CMAKE_MODULE_PATH before find_package(optimo_api), because optimo_api
# transitively pulls in roboligent-config.cmake which needs them.
set(UNREL_OPTIMO_API_CMAKE
    "${CMAKE_INSTALL_PREFIX}/../optimo_api/share/optimo_api/cmake")
get_filename_component(OPTIMO_API_CMAKE "${UNREL_OPTIMO_API_CMAKE}" ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH ${OPTIMO_API_CMAKE})

find_package(ament_index_cpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(optimo_api REQUIRED)
find_package(optimo_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# Roboligent SDK include path override — roboligent-config.cmake bakes in
# CMAKE_INSTALL_PREFIX which resolves to the colcon workspace, not /usr/local.
find_package(Roboligent 1.0.5 QUIET)
if(Roboligent_FOUND OR ROBOLIGENT_SDK_LIBRARIES)
  set(ROBOLIGENT_FOUND TRUE)
  message(STATUS "Roboligent SDK found")
else()
  set(ROBOLIGENT_FOUND FALSE)
  message(STATUS "Roboligent SDK not found")
endif()

add_library(${PROJECT_NAME} SHARED
  src/optimo_controller.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  controller_interface
  geometry_msgs
  hardware_interface
  optimo_api
  optimo_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  std_srvs
  tf2
  tf2_ros
)

if(ROBOLIGENT_FOUND)
  set(ROBOLIGENT_INCLUDE_DIR "/usr/local/include/roboligent-1.0.5")
  target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${ROBOLIGENT_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME}
    ${ROBOLIGENT_SDK_LIBRARIES})
endif()

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
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ament_index_cpp
  controller_interface
  geometry_msgs
  hardware_interface
  optimo_api
  optimo_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  std_srvs
  tf2
  tf2_ros
)

ament_package()

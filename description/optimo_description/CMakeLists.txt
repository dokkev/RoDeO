cmake_minimum_required(VERSION 3.8)
project(optimo_description)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(XACRO_EXECUTABLE xacro REQUIRED)

# Generate optimo.urdf from xacro at build time
set(XACRO_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/urdf/optimo.urdf.xacro")
set(URDF_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/urdf/optimo.urdf")
set(GENERATE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_urdf.py")

file(GLOB_RECURSE XACRO_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/urdf/*.xacro")
add_custom_command(
  OUTPUT ${URDF_OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/urdf
  COMMAND ${Python3_EXECUTABLE} ${GENERATE_SCRIPT} ${XACRO_INPUT} ${URDF_OUTPUT}
  DEPENDS ${XACRO_DEPS} ${GENERATE_SCRIPT}
  COMMENT "Generating optimo.urdf from xacro"
)
add_custom_target(generate_urdf ALL DEPENDS ${URDF_OUTPUT})

# Install generated URDF
install(FILES ${URDF_OUTPUT} DESTINATION share/${PROJECT_NAME}/urdf)

# Install asset directories
foreach(_asset_dir IN ITEMS config launch mjcf rviz urdf)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_asset_dir}")
    install(
      DIRECTORY ${_asset_dir}
      DESTINATION share/${PROJECT_NAME}
      PATTERN "optimo.urdf" EXCLUDE
    )
  endif()
endforeach()

ament_package()

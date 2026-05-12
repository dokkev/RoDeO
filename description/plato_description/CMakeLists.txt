cmake_minimum_required(VERSION 3.8)
project(plato_description)

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(XACRO_EXECUTABLE xacro REQUIRED)

set(PLATO_NARITOUCH_URDF_OUTPUT
  "${CMAKE_CURRENT_BINARY_DIR}/urdf/plato_naritouch.urdf")
set(PLATO_GENERATE_URDF_SCRIPT
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_pinocchio_urdf.py")

file(GLOB_RECURSE PLATO_XACRO_DEPS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/urdf/*.xacro")
add_custom_command(
  OUTPUT ${PLATO_NARITOUCH_URDF_OUTPUT}
  COMMAND ${Python3_EXECUTABLE}
          ${PLATO_GENERATE_URDF_SCRIPT}
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${PLATO_NARITOUCH_URDF_OUTPUT}
          --xacro ${XACRO_EXECUTABLE}
  DEPENDS ${PLATO_XACRO_DEPS} ${PLATO_GENERATE_URDF_SCRIPT}
  COMMENT "Generating plato_naritouch.urdf for Pinocchio"
)
add_custom_target(generate_plato_naritouch_urdf ALL
  DEPENDS ${PLATO_NARITOUCH_URDF_OUTPUT})

foreach(_asset_dir IN ITEMS config launch meshes rviz urdf mjcf)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_asset_dir}")
    install(
      DIRECTORY ${_asset_dir}
      DESTINATION share/${PROJECT_NAME}
      PATTERN "plato_naritouch.urdf" EXCLUDE
    )
  endif()
endforeach()

install(FILES ${PLATO_NARITOUCH_URDF_OUTPUT}
  DESTINATION share/${PROJECT_NAME}/urdf
)

ament_package()

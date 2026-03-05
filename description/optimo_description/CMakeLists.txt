cmake_minimum_required(VERSION 3.8)
project(optimo_description)

find_package(ament_cmake REQUIRED)

foreach(_asset_dir IN ITEMS config launch meshes mjcf rviz urdf)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_asset_dir}")
    install(
      DIRECTORY ${_asset_dir}
      DESTINATION share/${PROJECT_NAME}
    )
  endif()
endforeach()

ament_package()

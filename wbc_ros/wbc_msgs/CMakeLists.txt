cmake_minimum_required(VERSION 3.8)
project(wbc_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotionTaskCommand.msg"
  "msg/ContactStatus.msg"
  "msg/ForceTaskCommand.msg"
  "msg/TaskState.msg"
  "msg/WbcCommand.msg"
  "msg/WbcState.msg"
  "msg/WbcTeleopCmd.msg"
  "srv/ConfigureTask.srv"
  "srv/ConfigureContact.srv"
  "srv/ListTasks.srv"
  "srv/TransitionState.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()

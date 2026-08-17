cmake_minimum_required(VERSION 3.16)
project(action_stream_msgs VERSION 0.1.0 LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(message_files
  "msg/Observation.msg"
  "msg/InferenceRequest.msg"
  "msg/TargetAction.msg"
  "msg/ActionChunk.msg"
  "msg/RobotCommand.msg"
  "msg/ExecutorDiagnostics.msg"
  "msg/RuntimeEvent.msg"
  "msg/EpisodeControl.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${message_files}
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

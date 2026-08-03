cmake_minimum_required(VERSION 3.16)
project(action_stream_executor VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
  add_compile_options(/W4 /permissive-)
endif()

find_package(ament_cmake REQUIRED)
find_package(action_stream_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)

add_library(executor_state_machine src/executor_state_machine.cpp)
target_include_directories(executor_state_machine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(action_stream_executor_node
  src/executor_node.cpp
  src/main.cpp
)
target_include_directories(action_stream_executor_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(action_stream_executor_node executor_state_machine)
ament_target_dependencies(action_stream_executor_node
  action_stream_msgs
  builtin_interfaces
  rclcpp
)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS executor_state_machine action_stream_executor_node
  EXPORT export_action_stream_executor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_executor_state_machine test/test_executor_state_machine.cpp)
  if(TARGET test_executor_state_machine)
    target_link_libraries(test_executor_state_machine executor_state_machine)
    target_include_directories(test_executor_state_machine
      PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
  endif()
endif()

ament_export_targets(export_action_stream_executor HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_package()

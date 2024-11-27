cmake_minimum_required(VERSION 3.8)
project(dynamixel_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

include_directories(include
  ${std_msgs_INCLUDE_DIRS}
)

# Build
add_executable(read_write_node src/read_write_node.cpp)
ament_target_dependencies(read_write_node
  dynamixel_sdk
  rclcpp
  std_msgs
  moveit_ros_planning_interface
)

# Install
install(TARGETS
  read_write_node
  DESTINATION lib/${PROJECT_NAME}
)

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

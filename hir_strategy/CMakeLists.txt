cmake_minimum_required(VERSION 2.8.3)
project(hir_strategy)

find_package(catkin REQUIRED COMPONENTS
  moveit_msgs
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)


cmake_minimum_required(VERSION 2.8.3)
project(hir_learning)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  roscpp
  rospy
  std_msgs
)

catkin_package(CATKIN_DEPENDS
  moveit_core
  moveit_ros_planning_interface
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#add_executable(demo src/demo.cpp)
#target_link_libraries(demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})

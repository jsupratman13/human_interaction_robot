cmake_minimum_required(VERSION 2.8.3)
project(hir_bringup)

find_package(catkin REQUIRED COMPONENTS
#  icart_mini_driver
#  kondo_driver
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)


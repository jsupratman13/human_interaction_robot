# Human Interaction Robot
Undergraduate thesis

## System
* Ubuntu 16.04 LTS
* ROS Kinetic
* Gazebo 7

## Dependency Repositories
* https://github.com/open-rdc/icart_mini
* https://github.com/jsupratman13/kondo_driver

## Dependent Package
* Keras
* Tensorflow

## Usage
### Real Robot
#### bring up real robot
```
roslaunch hir_bringup interaction_robot.launch
roslaunch kondo_driver multiple_check.launch
rosrun hir_learning set_fixed_pos_arm.py
```

#### train real robot
```
roscd hir_learning/scripts
python training_real_time.py
```

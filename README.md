# Human Interaction Robot
Undergraduate thesis

## System
* Ubuntu 16.04 LTS
* ROS Kinetic
* Gazebo 7

## Dependency Repositories
* https://github.com/open-rdc/icart_mini
* https://github.com/jsupratman13/kondo_driver

## Usage
### Simulator
#### bring up simulated robot
```
roslaunch hir_bringup interaction_robot_sim.launch
```

#### train robot
```
rosrun hir_learning ddqn.py
```

#### test trained result
```
roscd hir_learning/scripts
python test.py <modelname>.json <weightname>.hdf5
```

#### evaluate model and weights
```
roscd hir_learning/scripts
python diagnostic.py <modelname>.json *.hdf5
```
### Real Robot
#### bring up real robot
```
roslaunch hir_bringup interaction_robot.launch
```

#### test trained result
```
roscd hir_learning/scripts
python real.py <modelname>.json <weightname>.hdf5
```

# Human Interaction Robot
Undergraduate thesis

## System
* Ubuntu 14.04 LTS
* ROS indigio
* Gazebo 7

## Dependency Repositories
* https://github.com/open-rdc/icart_mini
* https://github.com/open-rdc/orne_navigation

## Usage
### bring up simulated robot
```
roslaunch hir_bringup interaction_robot_sim.launch
```

### train robot
```
rosrun hir_learning ddqn.py
```

### test robot
```
roscd hir_learning/scripts
python test.py <modelname>.json <weightname>.hdf5
```

### evaluate model and weights
```
roscd hir_learning/scripts
python diagnostic.py <modelname>.json *.hdf5
```


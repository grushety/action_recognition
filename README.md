## Action recognition and prediction in robotics. 
### Pepper experiment
#### Environment Requiremens
1. Ubuntu 16.04
2. Gazebo 7
3. ROS Kinetic
4. Pepper Package from https://github.com/michtesar/pepper_ros
```
sudo apt install ros-kinetic-moveit-ros-visualization ros-kinetic-moveit-planners-ompl 
```
Moveit from source to use moveit-commander in scripts
```
git clone https://github.com/ros-planning/moveit.git -b kinetic-devel
```

#### Run experiment
Run launch file to start experiment
```
roslaunch action_recognition reconstruction_test.launch
```
or 
```
roslaunch action_recognition prediction_test.launch
```
To change the experiment setting see comments in launch files
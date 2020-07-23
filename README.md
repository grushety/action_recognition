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
this part is still not ready ...

#### After installation

1. Please copy test.world file from the action_recognition/gazebo
to pepper-ros/pepper_gazebo_plugin/worlds folder and correct the world name in
pepper_gazebo_plugin_Y20.launch
2. please copy folder "pepper_camera" to local folder .gazebo/models

#### Run experiment

First please launch Gazebo, MoveIt to start experiment.
To start MoveIt plugin we should first make sure that simulation in Gazebo is running.

```
roslaunch pepper_gazebo_plugin pepper_gazebo_plugin_Y20.launch
roslaunch pepper_moveit_config moveit_planner.launch
```
After that we can launch tests
```
roslaunch action_recognition reconstruction_test.launch
```
or 
```
roslaunch action_recognition prediction_test.launch
```
To change the experiment setting see comments in launch files.

### Run MVAE test results

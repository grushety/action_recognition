# Project for Bachelor Thesis
##### "Vorhersage und Erkennung von Handlungskonsequenzen in Robotern"
#### Environment Requiremens
1. Ubuntu 16.04
2. Python 2.7
3. TensorFlow 1.3.0
4. Gazebo 7
5. ROS Kinetic
6. Pepper Package from https://github.com/michtesar/pepper_ros
```
git clone https://github.com/michtesar/pepper_ros.git
cd ..
rosdep check -y --from-paths . --ignore-src --rosdistro kinetic
rosdep install -y --from-paths . --ignore-src --rosdistro kinetic
catkin_make
```
7. Moveit from source to use moveit-commander in scripts
```
sudo apt install ros-kinetic-moveit-ros-visualization ros-kinetic-moveit-planners-ompl 
git clone https://github.com/ros-planning/moveit.git -b kinetic-devel
cd ..
rosdep check -y --from-paths . --ignore-src --rosdistro kinetic
rosdep install -y --from-paths . --ignore-src --rosdistro kinetic
catkin_make
```
8. Install from source : *ros-control*, *gazebo_ros_pkgs*, *roboticsgroup_gazebo_plugins*

```
git clone https://github.com/ros-controls/ros_control.git -b kinetic-devel
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b kinetic-devel
git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git
```

#### After installation

1. Copy *test.world* file from the *action_recognition/gazebo*
to *pepper-ros/pepper_gazebo_plugin/worlds* folder and correct the world name in
*pepper_gazebo_plugin_Y20.launch*
2. Copy folder *pepper_camera* to local folder *.gazebo/models* to use custom camera in simulation
3. Please change Pepper finger color.
go to *pepper_ros/pepper_meshes/meshes/1.0/LFinger13.dae* and *LFinger12.dae* and change:
- color sid="emission" from 0 0 0 1 to **1 0 0 1**
- color sid="ambient" from 0 0 0 1 to **1 0 0 1**
- color sid="diffuse" from 1 1 1 1 to **1 0 0 1**
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

Use *TestPerfomance.ipynb* to reproduce results presented in the work.
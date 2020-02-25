#!/usr/bin/env python

import math
import random
import sys

import moveit_commander
import rospy

NUM = 25  # arg for calculating the size of  valid joint configs to generate: size = (2*NUM+3)^2


class Motor(object):
    """
    Moves Robot's arm in fixed head bottom camera scope
    """
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.valid_joint_configs = generate_valid_joint_configs()
        self.number_of_samples = 10

    def start_motor(self):
        """
        Start movement in infinite loop if previous movement is ended
        """
        old_joints = self.group.get_current_joint_values()
        while True:
            cur_joints = self.group.get_current_joint_values()
            if cur_joints == old_joints:
                index = random.randint(0, len(self.valid_joint_configs) - 1)
                joint_goal = self.valid_joint_configs[index]
                self.group.go(joint_goal, wait=True)
            old_joints = cur_joints


def generate_valid_joint_configs():
    """
    @return: Array of joints config allowable to set it as moving goal
    """
    valid_joints_configs = []
    left = fill_vectors_between([-0.45, 0.52, 0, -0.15, 0], [-0.08, 0.65, 0, -0.1, 0])
    left_down = fill_vectors_between([-0.08, 0.65, 0, -0.1, 0], [0.5, 0.7, 0, -0.01, 0])
    middle = fill_vectors_between([-0.5, 0.5, 0, -0.85, 0], [-0.1, 0.55, 0, -0.7, 0])
    middle_down = fill_vectors_between([-0.1, 0.55, 0, -0.7, 0], [0.3, 0.6, 0, -0.85, 0])
    right = fill_vectors_between([-0.8, 0.5, 0, -1.5, 0], [-0.55, 0.5, 0, -1.55, 0])
    right_down = fill_vectors_between([-0.55, 0.5, 0, -1.55, 0], [-0.22, 0.5, 0, -1.55, 0])
    left.extend(left_down)
    middle.extend(middle_down)
    right.extend(right_down)
    for i in range(len(left)):
        line = fill_vectors_between(left[i], middle[i]) + fill_vectors_between(middle[i], right[i])
        valid_joints_configs.extend(line)
    return valid_joints_configs


def fill_vectors_between(v, w):
    """
    @param v, w:  joints vectors
    @return: array of vectors evenly distributed between v and w of size NUM
    """
    vectors = [v]
    v_0, v_1, v_2 = find_step(v[0], w[0]), find_step(v[1], w[1]), find_step(v[3], w[3])
    prev_vector = v
    for i in range(NUM):
        vec = [0, 0, 0, 0, 0]
        vec[0] = prev_vector[0] + v_0
        vec[1] = prev_vector[1] + v_1
        vec[2] = 0
        vec[3] = prev_vector[3] + v_2
        vec[4] = 0
        vectors.append(vec)
        prev_vector = vec
    return vectors


def find_step(a, b):
    """
    Help function for evenly distribution of values
    """
    if a == b:
        return 0
    step = math.fabs(a - b) / NUM
    if a > b:
        step *= -1
    return step

def main(args):
    ic = Motor()
    rospy.init_node('motor', anonymous=True)
    try:
        ic.start_motor()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS data_collector module")


if __name__ == '__main__':
    main(sys.argv)

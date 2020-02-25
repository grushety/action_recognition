#!/usr/bin/env python
import sys
import numpy as np
import math
import random
import time

import rospy
import moveit_commander
import moveit_msgs.msg

t = 15
class Motor:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")

    def start_motor(self):
        print "left up"
        joint_goal = [-0.45, 0.52, 0, -0.15, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "left middle"
        joint_goal = [-0.08, 0.65, 0, -0.1, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "left down"
        joint_goal = [0.5, 0.7, 0, -0.01, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "middle up"
        joint_goal = [-0.5, 0.5, 0, -0.85, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "center"
        joint_goal = [-0.1, 0.55, 0, -0.7, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "middle down"
        joint_goal = [0.3, 0.6, 0, -0.85, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "right up"
        joint_goal = [-0.8, 0.5, 0, -1.5, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "right middle"
        joint_goal = [-0.55, 0.5, 0, -1.55, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)
        print "right down"
        joint_goal = [-0.22, 0.5, 0, -1.55, 0]
        self.group.go(joint_goal, wait=True)
        time.sleep(t)


def main(args):
    ic = Motor()
    rospy.init_node('motor', anonymous=True)
    try:
        ic.start_motor()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS data_collector module"

if __name__ == '__main__':
    main(sys.argv)
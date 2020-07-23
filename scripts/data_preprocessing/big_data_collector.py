#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2
import imutils
from datetime import datetime, timedelta
import scipy.io as sio
from PIL import Image

import rospy
import rospkg
import moveit_commander
from sensor_msgs.msg import CompressedImage
import moveit_msgs.msg

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)
MILS = 500 #only 3-4 samples per second


class BigDataCollector:
    def __init__(self):
        self.image_pub = rospy.Subscriber("/pepper_robot/camera/bottom/image_raw/compressed",
                                          CompressedImage,
                                          self.get_arm_position,
                                          queue_size=20)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.groupL = moveit_commander.MoveGroupCommander("left_arm")
        self.groupR = moveit_commander.MoveGroupCommander("right_arm")
        self.image = []
        self.number_of_samples = 10
        rp = rospkg.RosPack()
        self.path = os.path.join(rp.get_path("action_recognition"), "scripts", "learning", "database")

    def get_arm_position(self, ros_data):
        # Convert  input image
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb = Image.fromarray(np_arr)
        # rgb.show()
        rgb = rgb.convert('L')
        img = rgb.resize((24, 24), Image.ANTIALIAS)


    def make_new_sample(self):
        current_joints = self.group.get_current_joint_values()
        current_joints = self.group1.get_current_joint_values()
        sp, sr, er = current_joints[0], current_joints[1], current_joints[3]
        x, y = self.coordinate[0], self.coordinate[1]
        sample = np.asarray([sp, sr, er, x, y])
        return sample

    def collector(self):
        samples = []
        counter = 0
        old_sample = self.make_new_sample()
        old_joints, old_time = self.group.get_current_joint_values(), datetime.now()
        while counter < self.number_of_samples:
            cur_joints = self.group.get_current_joint_values()
            if cur_joints != old_joints:
                if datetime.now() >= old_time + timedelta(milliseconds=MILS):
                    new_sample = self.make_new_sample()
                    args = (old_sample[0:3], new_sample[0:3], old_sample[3:], new_sample[3:])
                    sample = np.concatenate(args)
                    samples = np.append(samples, sample)
                    old_sample = new_sample
                    old_time = datetime.now() + timedelta(milliseconds=MILS)
                    counter += 1
                    print counter
            old_joints = cur_joints
        samples = np.asarray(samples, dtype=np.float32)
        samples = np.reshape(samples, (-1, 10))
        sio.savemat(self.path + '/large_train_set.mat', {'extra_data': samples})


def main(args):
    # Initializes and cleanup ros node
    ic = BigDataCollector()
    #ic.number_of_samples = args
    rospy.init_node('data_collector', anonymous=True)
    ic.collector()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

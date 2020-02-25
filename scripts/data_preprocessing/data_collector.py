#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2
import imutils
from datetime import datetime, timedelta
import scipy.io as sio

import rospy
import rospkg
import moveit_commander
from sensor_msgs.msg import CompressedImage
import moveit_msgs.msg

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)
MILS = 500 #only 3-4 samples per second


class DataCollector:
    def __init__(self):
        self.image_pub = rospy.Subscriber("/pepper_robot/camera/bottom/image_raw/compressed",
                                          CompressedImage,
                                          self.get_arm_position,
                                          queue_size=20)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.coordinate = (0, 0)
        self.number_of_samples = 1000
        rp = rospkg.RosPack()
        self.path = os.path.join(rp.get_path("action_recognition"), "scripts", "learning", "database")

    def get_arm_position(self, ros_data):
        # Convert  input image
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Find a center of red area
        mask = cv2.inRange(image, RED_LOW, RED_UP)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            self.coordinate = (x, y)
        else:
            self.coordinate = (0, 0)
        print self.coordinate

    def make_new_sample(self):
        current_joints = self.group.get_current_joint_values()
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
        sio.savemat(self.path + '/reconstruct_data.mat', {'data': samples})


def main(args):
    # Initializes and cleanup ros node
    ic = DataCollector()
    #ic.number_of_samples = args
    rospy.init_node('data_collector', anonymous=True)
    ic.collector()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python

import sys
import os
import numpy as np
import cv2
import imutils
from datetime import datetime, timedelta
from time import sleep
import scipy.io as sio

import rospy
import rospkg
from sensor_msgs.msg import CompressedImage

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)
MILS = 100 #only 3-4 samples per second


class LineGenerator:
    def __init__(self):
        self.image_pub = rospy.Subscriber("/iris/camera/image_raw/compressed",
                                          CompressedImage,
                                          self.get_arm_position,
                                          queue_size=20)
        self.coordinate = (0, 0)
        self.number_of_samples = 1000
        self.experiment_time = 7
        rp = rospkg.RosPack()
        self.path = os.path.join(rp.get_path("action_recognition"), "scripts", "data")
        #temp = np.load(self.path + '/lines.txt.npy')
        #print temp.tolist()

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
            if M["m10"]!= 0 and M["m01"]!=0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                self.coordinate = (x, y)
        else:
            self.coordinate = (0, 0)

    def collector(self):
        lines = []
        counter = 0
        while counter < self.number_of_samples:
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=self.experiment_time)
            line = []
            while datetime.now() < end_time:
                sleep(0.1)
                point = (self.coordinate)
                line.append(point)
            lines.append(line)
            counter+=1
            print counter
        sio.savemat(self.path + '/line.mat', {'data': lines})


def main(args):
    # Initializes and cleanup ros node
    ic = LineGenerator()
    #ic.number_of_samples = args
    rospy.init_node('line_generator', anonymous=True)
    ic.collector()

if __name__ == '__main__':
    main(sys.argv)

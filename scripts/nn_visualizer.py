#!/usr/bin/env python

import tensorflow as tf
import sys
import os
import numpy as np
import ast
import cv2

import rospy
import rospkg
from std_msgs.msg import String

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rp = rospkg.RosPack()
PATH = os.path.join(rp.get_path("action_recognition"), "scripts", "learning")
MILS = 50
missing_mod = [-2, -2, -2, -2]
global mode
mode = 0


class Visualizer(object):
    """
    Visualisation to compare predicted and reconstracted data to
    """
    def __init__(self):
        self.tracker_data = rospy.Subscriber('/action_recognition/pos_from_side_camera',
                                             String, self.update_data, queue_size=20)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.rec_data = rospy.Subscriber("/action_recognition/test_rec",
                                         String, self.rec_callback, queue_size=20)
        self.pred_data = rospy.Subscriber("/action_recognition/test_pred",
                                          String, self.pred_callback, queue_size=20)
        self.tracker_points = np.array([])
        self.reconstr_points = np.array([])
        self.predict_points = np.array([])
        self.status = ""

    def update_data(self, data):
        points = ast.literal_eval(data.data)
        self.tracker_points = np.asarray([points])

    def rec_callback(self, data):
        points = ast.literal_eval(data.data)
        if self.status == "start":
            self.reconstr_points = np.asarray([points], dtype=np.int32)

    def pred_callback(self, data):
        points = ast.literal_eval(data.data)
        if self.status == "start":
            self.predict_points = np.asarray([points], dtype=np.int32)

    def synchronize(self, data):
        msg = data.data.split(' ')
        self.status = msg[0]
        print
        self.status
        if self.status == "end":
            self.reconstr_points = np.array([])
            self.predict_points = np.array([])
            self.tracker_points = np.array([])


def main(args):
    if len(args) > 1:
        if args[1] == "1":
            global mode
            mode = 1
    rospy.init_node('nn_visualizer', anonymous=True)
    ic = Visualizer()
    try:
        while not rospy.is_shutdown():
            image = np.ones([480, 640, 3], dtype=np.uint8) * 255
            if ic.tracker_points.size > 4:
                cv2.polylines(image, [ic.tracker_points], False, (0, 0, 0), 2, lineType=4)
            if ic.reconstr_points.size > 4:
                cv2.polylines(image, [ic.reconstr_points], False, (255, 0, 255), 2, lineType=6)
            if ic.predict_points.size > 4:
                cv2.polylines(image, [ic.predict_points], False, (0, 255, 255), 2, lineType=8)
            cv2.imshow('monitor', image)
            cv2.waitKey(2)

    except KeyboardInterrupt:
        print "Shutting down ROS visualizer module"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python

import ast
import cv2
import sys
import os
from datetime import datetime, timedelta
from random import randint
import scipy.io as sio
import numpy as np

import rospy
import rospkg
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String

from service.service import create_layout_3D, get_random_color

N_LINES = 2
THICKNESS = 2
RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)

rp = rospkg.RosPack()
PATH = os.path.join(rp.get_path("action_recognition"), "scripts", "extra_data", "line.mat")
database = sio.loadmat(PATH)
db = database['data'].tolist()


class Monitor(object):
    """
    Monitor becomes the current trajectory of Robot's arm from Tracker node and
    noises it with other trajectories, pre-generated and recorded in /extra_data/lines.
    Monitor node sends the data to Comparator node and visualizes the extra_data
    """
    def __init__(self):
        self.publisher = rospy.Publisher("/action_recognition/monitor/data",
                                         Int32MultiArray,
                                         queue_size=20)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.tracker_data = rospy.Subscriber('/action_recognition/pos_from_side_camera',
                                             String, self.update_data, queue_size=20)

        self.cmd = ''
        self.color_array = [get_random_color() for i in range(N_LINES + 1)]
        self.start_time = datetime.now() - timedelta(seconds=7)
        self.lines = []
        self.monitor_data = np.array([])

    def synchronize(self, msg):
        """
        Callback
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        cmd = msg[0]
        if cmd == "prepare":
            self.set_fake_line_coordinates()
        if cmd == "end":
            self.color_array = []
            self.monitor_data = []

    def set_fake_line_coordinates(self):
        """
        Method picks N random trajectories from db and initialize color array for every trajectory
        """
        lines = []
        for i in range(N_LINES):
            index = randint(0, len(db) - 1)
            line = db[index]
            lines.append(line)
        self.lines = lines
        self.color_array = [get_random_color() for i in range(N_LINES + 1)]

    def update_data(self, data):
        """
        Callback
        Method creates a matrix with the current trajectory from the Tracker Node and
        the parts of noising trajectories, that have the same length as Tracker's trajectory.
        After that the matrix is sent to Comparator node.
        @param data: trajectory of Robot's arm from Tracker
        """
        points = ast.literal_eval(data.data)
        points = np.asarray([points])
        l = len(points[0])
        if 2 <= l <= 70 and len(self.lines) > 0:
            for line in self.lines:
                x = np.asarray([line[0:l]])
                points = np.concatenate((points, x), axis=0)
            self.monitor_data = np.array(points, dtype=np.uint32)

            # Prepare and send data
            one_dim_points = self.monitor_data.reshape(-1)
            msg = Int32MultiArray()
            msg.layout = create_layout_3D(l, N_LINES)
            msg.data = np.frombuffer(one_dim_points.tobytes(), 'int32')
            self.publisher.publish(msg)
        else:
            self.monitor_data = []


def main(args):
    """
    Visualize trajectories
    @param args: mode of the visualizing:
        1 - drawing trajectories as lines,
        0 - drawing only last point as circle
    """
    ic = Monitor()
    rospy.init_node('monitor', anonymous=True)
    mode = 1
    if len(args) > 1:
        mode = args[1]
    try:
        while not rospy.is_shutdown():
            image = np.ones([480, 640, 3], dtype=np.uint8) * 255
            if len(ic.monitor_data) > 1:
                colors = map(tuple, ic.color_array)
                for data, color in zip(ic.monitor_data, colors):
                    if mode:
                        cv2.polylines(image, np.int32([data]), False, color, 3)
                    else:
                        cv2.circle(image, tuple(data[-1]), 15, color, 10)
            cv2.imshow('monitor', image)
            cv2.waitKey(2)
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS tracker module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

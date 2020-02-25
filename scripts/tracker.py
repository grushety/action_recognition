#!/usr/bin/env python

import sys
import numpy as np
import cv2
import imutils
from datetime import datetime, timedelta

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)
# time difference with witch samples would be made (100 mils only 3-4 samples per second)
MILS = 100

global image_source
image_source = "/iris/camera/image_raw/compressed"


class Tracker(object):
    """
    Tracker node traces movements of Robot's finger and publish the data for Monitor node
    """
    def __init__(self):
        self.camera_subscriber = rospy.Subscriber(image_source,
                                                  CompressedImage, self.get_arm_position, queue_size=20)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.pub = rospy.Publisher('/action_recognition/pos_from_side_camera', String, queue_size=10)
        self.coordinate = [0, 0]
        self.status = ""

    def synchronize(self, msg):
        """
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        self.status = msg[0]

    def get_arm_position(self, ros_data):
        """
            Callback
            Method used to locate a position of Pepper's red finger on the image
        """
        # Convert  input image
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Find a contour of red area
        mask = cv2.inRange(image, RED_LOW, RED_UP)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Find a center of red area
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m10"] != 0 and M["m01"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                self.coordinate = [640 - x, y]  # mirror the coordinates

    def track(self):
        """
        The method with a certain frequency adds coordinates to others
        collected from the beginning of the test and publish them.
        At the end of the test the coordinates sequence is emptied.
        """
        coords = []
        t_old = datetime.now()
        while True:
            while self.status == "start":
                if datetime.now() > t_old + timedelta(milliseconds=MILS):
                    coords.append(self.coordinate)
                    self.pub.publish(str(coords))
                    t_old = datetime.now()
            coords = []


def main(args):
    """
    @param args: if mode set on 1, tracker will use bottom camera from Pepper's head instead of opposite camera
    """
    if len(args) > 1:
        if args[1] == "1":
            global image_source
            image_source = "/pepper_robot/camera/bottom/image_raw/compressed"
    ic = Tracker()
    rospy.init_node('tracker', anonymous=True)
    try:
        ic.track()
    except KeyboardInterrupt:
        print ("Shutting down ROS tracker module")


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python

import tensorflow as tf
import scipy.io
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import cv2
import imutils

from learning.mvae import VariationalAutoencoder
from learning.mvae import network_param
from service.service import denormalize_coord, prepare_data_to_send

import rospy
import rospkg
import moveit_commander
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import CompressedImage

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)

MILS = 10
missing_mod = [-2, -2, -2, -2]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rp = rospkg.RosPack()
PATH = os.path.join(rp.get_path("action_recognition"), "scripts", "learning")

class Reconstructor(object):
    """
    Uses MVAE to predict the position of Robot's finger
    with current joints configurations, periodically requested from Moveit tool.
    Publishes the results for Comparator node.
    """

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.pub_pred = rospy.Publisher('/action_recognition/pos_prediction', Int32MultiArray, queue_size=10)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.tracker_data = rospy.Subscriber("/pepper_robot/camera/bottom/image_raw/compressed",
                                             CompressedImage, self.get_position, queue_size=20)
        self.status = ""
        self.reconstructed_trajectory = np.array([])
        self.predicted_trajectory = np.array([])
        self.tracker_coords = []

    def synchronize(self, msg):
        """
        Callback
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        self.status = msg[0]
        if self.status == "end":
            self.reconstructed_trajectory = []

    def get_position(self, camera_data):
        """
            Callback
            Method used to locate a position of Pepper's red finger from head camera
        """
        # Convert  input image
        np_arr = np.fromstring(camera_data.data, np.uint8)
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
                self.tracker_coords = [x, y]

    def get_sample(self):
        """
        @return: current joints data formatted for MVAE input
        """
        sample = self.group.get_current_joint_values()
        return [sample[0], sample[1], sample[3]]

    def run(self):
        """
        The method with a certain frequency (defined by MILS var) requests joints configuration data
        and uses it as input in loaded network to make a reconstruction/prediction of visual output.
        """
        with tf.Graph().as_default() as g:
            with tf.Session() as sess:
                # Network parameters
                network_architecture = network_param()
                learning_rate = 0.00001
                batch_size = 1
                with tf.name_scope("model"):
                    model = VariationalAutoencoder(sess, network_architecture, batch_size=batch_size,
                                                    learning_rate=learning_rate,
                                                    vae_mode=False, vae_mode_modalities=False)

            with tf.Session() as sess:
                model_var = {v.name.lstrip("model/"): v
                                  for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="model1/")}

                new_saver = tf.train.Saver(var_list=model_var)
                new_saver.restore(sess, PATH + "/models/prediction_network.ckpt")
                # new_saver.restore(sess, PATH + "/models/mixed_network.ckpt")
                print("Models restored")

                old_time = datetime.now()
                old_joint = self.get_sample()
                while True:
                    if datetime.now() > old_time + timedelta(milliseconds=MILS) and self.status == "start":
                        joint = self.get_sample()
                        pos = self.tracker_coords

                        # prepare the input data for MVAE prediction
                        # First option : using reconstructed visual input for t-1
                        pred_input = [joint + [-2, -2, -2] + pos + [-2, -2]]

                        # Second option : using the tracked coordinates from Robot's point of view (head bottom camera)
                        # pred_input = [joint + [-2, -2, -2] + self.tracker_coords + [-2, -2]]

                        predict, _ = model.reconstruct(sess, pred_input)
                        pred_coord = denormalize_coord(predict[0][8:])



                        # handle time issue for next loop
                        old_joint = joint
                        old_time = datetime.now()
                        # publish predicted direction to Comparator node
                        msg_pred = prepare_data_to_send(pred_coord, self.predicted_trajectory)
                        self.pub_pred.publish(msg_pred)



def main(args):
    """
    @param args: MVAE model used in reconstruction or prediction
    """
    rospy.init_node('predictor', anonymous=True)
    ic = Reconstructor()
    try:
        ic.run()
    except KeyboardInterrupt:
        print ("Shutting down ROS predictor module")


if __name__ == '__main__':
    main(sys.argv)

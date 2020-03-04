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
from service.service import get_direction, denormalize_coord, prepare_data_to_send

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
    Uses MVAE to reconstruct and predict the position of Robot's finger
    with current joints configurations, periodically requested from Moveit tool.
    Publishes the results for Comparator node.
    """

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.pub_rec = rospy.Publisher('/action_recognition/pos_reconstruction', Int32MultiArray, queue_size=10)
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
                with tf.name_scope("model1"):
                    model1 = VariationalAutoencoder(sess, network_architecture, batch_size=batch_size,
                                                    learning_rate=learning_rate,
                                                    vae_mode=False, vae_mode_modalities=False)
                with tf.name_scope("model2"):
                    model2 = VariationalAutoencoder(sess, network_architecture, batch_size=batch_size,
                                                    learning_rate=learning_rate,
                                                    vae_mode=False, vae_mode_modalities=False)

            with tf.Session() as sess:
                model1_var = {v.name.lstrip("model1/"): v
                                  for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="model1/")}
                model2_var = {v.name.lstrip("model2/"): v
                                  for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="model2/")}
                new_saver = tf.train.Saver(var_list=model1_var)
                new_saver2 = tf.train.Saver(var_list=model2_var)
                new_saver.restore(sess, PATH + "/models/all_conf_network.ckpt")
                new_saver2.restore(sess, PATH + "/models/prediction_network.ckpt")
                # new_saver.restore(sess, PATH + "/models/mixed_network.ckpt")
                print("Models restored")

                old_time = datetime.now()
                old_joint = self.get_sample()
                while True:
                    if datetime.now() > old_time + timedelta(milliseconds=MILS) and self.status == "start":
                        joint = self.get_sample()

                        # prepare the input data for MVAE reconstruction
                        rec_input = [old_joint + joint + missing_mod]

                        # pass the data through nn and denormalize output coordinates
                        reconstruct, _ = model1.reconstruct(sess, rec_input)

                        # prepare the input data for MVAE prediction
                        # First option : using reconstructed visual input for t-1
                        pred_input = [joint + [-2, -2, -2] + list(reconstruct[0][8:]) + [-2, -2]]

                        # Second option : using the tracked coordinates from Robot's point of view (head bottom camera)
                        # pred_input = [joint + [-2, -2, -2] + self.tracker_coords + [-2, -2]]

                        predict, _ = model2.reconstruct(sess, pred_input)
                        rec_coord = denormalize_coord(reconstruct[0][8:])
                        pred_coord = denormalize_coord(predict[0][8:])

                        # code the prediction value in moving direction value like "up" or "down-left"
                        direction = get_direction(rec_coord, pred_coord)

                        # handle time issue for next loop
                        old_joint = joint
                        old_time = datetime.now()
                        # publish predicted direction to Comparator node
                        msg_pred = prepare_data_to_send(rec_coord, self.predicted_trajectory)
                        self.pub_pred.publish(direction)

                        # prepare reconstructed_trajectory array and publish it
                        msg_rec = prepare_data_to_send(rec_coord, self.reconstructed_trajectory)
                        self.pub_rec.publish(msg_rec)


def main(args):
    """
    @param args: MVAE model used in reconstruction or prediction
    """
    rospy.init_node('reconstructor', anonymous=True)
    ic = Reconstructor()
    try:
        ic.run()
    except KeyboardInterrupt:
        print ("Shutting down ROS reconstructor module")


if __name__ == '__main__':
    main(sys.argv)

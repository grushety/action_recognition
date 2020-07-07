#!/usr/bin/env python

import tensorflow as tf
import sys
import os
import random
from datetime import datetime, timedelta
import numpy as np

from learning.mvae import VariationalAutoencoder
from learning.mvae import network_param
from service.service import denormalize_coord, prepare_data_to_send

import rospy
import rospkg
import moveit_commander
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

RED_LOW = (0, 0, 150)
RED_UP = (10, 10, 255)

MILS = 90
missing_mod = [-2, -2, -2, -2]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rp = rospkg.RosPack()
PATH = os.path.join(rp.get_path("action_recognition"), "scripts", "learning")
#nn = "/models/mixed_network.ckpt"
nn = "models/reconstruction_network.ckpt"


class Reconstructor(object):
    """
    Uses MVAE to reconstruct the position of Robot's finger
    with current joints configurations, periodically requested from Moveit tool.
    Publishes the results for Comparator node.
    """

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.pub_rec = rospy.Publisher('/action_recognition/pos_reconstruction', String, queue_size=10)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.status = ""
        self.coords = []

    def synchronize(self, msg):
        """
        Callback
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        self.status = msg[0]
        if self.status == "end":
            self.coords = []

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

                model = VariationalAutoencoder(sess, network_architecture, batch_size=batch_size,
                                               learning_rate=learning_rate,
                                               vae_mode=False, vae_mode_modalities=False)
            with tf.Session() as sess:
                new_saver = tf.train.Saver()
                new_saver.restore(sess, PATH + nn)
                print("Model restored.")

                old_time = datetime.now()
                old_joint = self.get_sample()
                while True:
                    if datetime.now() > old_time + timedelta(milliseconds=MILS) and self.status == "start":
                        joint = self.get_sample()

                        # prepare the input data for MVAE reconstruction
                        rec_input = [old_joint + joint + missing_mod]

                        # pass the data through nn and denormalize output coordinates
                        reconstruct, _ = model.reconstruct(sess, rec_input)

                        rec_coord = denormalize_coord(reconstruct[0][8:]).tolist()
                        # handle time issue for next loop
                        old_joint = joint
                        old_time = datetime.now()
                        if self.status=="start":
                            self.coords.append(rec_coord)
                        # prepare reconstructed_trajectory array and publish it
                        #msg_rec = prepare_data_to_send(rec_coord, self.reconstructed_trajectory)

                            self.pub_rec.publish(str(self.coords))


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

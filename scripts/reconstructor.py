#!/usr/bin/env python

import tensorflow as tf
import scipy.io
import sys
import os
from datetime import datetime, timedelta
import numpy as np

from learning.mvae import VariationalAutoencoder
from learning.mvae import network_param
from service.service import get_direction, denormalize_coord, create_layout_2D

import rospy
import rospkg
import moveit_commander
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

MILS = 20
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
        self.pub_pred = rospy.Publisher('/action_recognition/pos_prediction', String, queue_size=10)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.status = ""
        self.reconstructed_trajectory = []

    def synchronize(self, msg):
        """
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        self.status = msg[0]
        if self.status == "end":
            self.reconstructed_trajectory = []

    def get_sample(self):
        """
        @return: current joints data formatted for MVAE
        """
        sample = self.group.get_current_joint_values()
        return [sample[0], sample[1], sample[3]]

    def run(self):
        """
        The method with a certain frequency (defined by MILS var) requests joints configuration data
        and uses it as input in loaded network to make a reconstruction of visual output.
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
                new_saver.restore(sess, PATH + "/models/all_conf_network.ckpt")
                # new_saver.restore(sess, PATH + "/models/prediction_network.ckpt")
                # new_saver.restore(sess, PATH + "/models/mixed_network.ckpt")
                print("Model restored")

                old_time = datetime.now()
                old_joint = self.get_sample()
                while True:
                    if datetime.now() > old_time + timedelta(milliseconds=MILS):

                        # prepare the input data for MVAE
                        joint = self.get_sample()
                        n = [old_joint + joint + missing_mod]

                        # handle time issue for next loop
                        old_joint = joint
                        old_time = datetime.now()

                        # pass the data through nn and denormalize output coordinates
                        reconstruct, reconstruct_log_sigma_sq = model.reconstruct(sess, n)
                        predict, predict_log_sigma_sq = model.reconstruct(sess, reconstruct)
                        rec_coord = denormalize_coord(reconstruct[0][8:])
                        pred_coord = denormalize_coord(predict[0][8:])

                        # code the prediction value in moving direction value like "up" or "down-left"
                        direction = get_direction(rec_coord, pred_coord)

                        if self.status == "start":

                            # if test is running, prepare reconstructed_trajectory array and publish it
                            self.reconstructed_trajectory.append(rec_coord)
                            length = len(self.reconstructed_trajectory)
                            ar = np.asarray(self.reconstructed_trajectory, dtype=np.uint32)
                            one_dim_points = ar.reshape(-1)
                            msg = Int32MultiArray()
                            msg.layout = create_layout_2D(length)
                            msg.data = np.frombuffer(one_dim_points.tobytes(), 'int32')
                            self.pub_rec.publish(msg)

                            # publish predicted direction to Comparator node
                            self.pub_pred.publish(direction)


def main(args):
    """
    @param args: MVAE model used
    """
    rospy.init_node('reconstructor', anonymous=True)
    ic = Reconstructor()
    try:
        ic.run()
    except KeyboardInterrupt:
        print ("Shutting down ROS reconstructor module")


if __name__ == '__main__':
    main(sys.argv)

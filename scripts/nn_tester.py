#!/usr/bin/env python

import tensorflow as tf
import sys
import os

from datetime import datetime, timedelta

from learning.mvae import VariationalAutoencoder
from learning.mvae import network_param
from service.service import denormalize_coord

import rospy
import rospkg
import moveit_commander
from std_msgs.msg import String

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rp = rospkg.RosPack()
PATH = os.path.join(rp.get_path("action_recognition"), "scripts", "learning")
MILS = 50
missing_mod = [-2, -2, -2, -2]

class NN_tester:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("left_arm")
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.rec_pub = rospy.Publisher("/action_recognition/test_rec",
                                         String, queue_size=20)
        self.pred_pub = rospy.Publisher("/action_recognition/test_pred",
                                          String, queue_size=20)
        self.reconstr_points = []
        self.predict_points = []
        self.status = ""

    def run(self):
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
                print("Model restored.")
                old_time = datetime.now()
                old_joint = self.get_sample()
                while True:
                    if datetime.now() > old_time + timedelta(milliseconds=MILS):
                        joint = self.get_sample()
                        n = [old_joint + joint + missing_mod]
                        reconstruct, reconstruct_log_sigma_sq = model.reconstruct(sess, n)
                        predict, predict_log_sigma_sq = model.reconstruct(sess, reconstruct)
                        old_joint = joint
                        old_time = datetime.now()
                        denorm_rec_coord = denormalize_coord(reconstruct[0][8:])
                        denorm_pred_coord = denormalize_coord(predict[0][6:8])
                        self.reconstr_points.append(denorm_rec_coord.tolist())
                        self.predict_points.append(denorm_pred_coord.tolist())
                        self.rec_pub.publish(str(self.reconstr_points))
                        self.pred_pub.publish(str(self.predict_points))

    def get_sample(self):
        sample = self.group.get_current_joint_values()
        return [sample[0], sample[1], sample[3]]

    def synchronize(self, data):
        msg = data.data.split(' ')
        self.status = msg[0]
        print self.status
        if self.status == "end":
            self.reconstr_points = []
            self.predict_points = []

def main(args):
    rospy.init_node('nn_tester', anonymous=True)
    ic = NN_tester()
    try:
        ic.run()
    except KeyboardInterrupt:
        print "Shutting down ROS visualizer module"


if __name__ == '__main__':
    main(sys.argv)
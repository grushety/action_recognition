#!/usr/bin/env python
import sys
import ast
import numpy as np
from scipy import spatial
from datetime import datetime, timedelta

from service.service import get_direction

import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray


class Comparator(object):
    """
    Proof the capability of MVAE to reconstruct or to predict a position of Robot's Finger,
    by finding out witch trajectory was generated by Robot between some noising trajectories
    """

    def __init__(self):
        self.sub_rec = rospy.Subscriber('/action_recognition/pos_reconstruction',
                                        Int32MultiArray, self.receive_reconstruction_data,
                                        queue_size=1)
        self.sub_pred = rospy.Subscriber('/action_recognition/pos_prediction',
                                         String, self.receive_prediction, queue_size=1)
        self.sub_monitor = rospy.Subscriber('/action_recognition/monitor/data',
                                            Int32MultiArray, self.receive_monitor_data,
                                            queue_size=1)
        self.pub = rospy.Publisher('/action_recognition/results', String, queue_size=10)
        self.status_subscriber = rospy.Subscriber("/action_recognition/test_status",
                                                  String, self.synchronize, queue_size=20)
        self.monitor_data = []
        self.pred_direction = "Zero"
        self.reconstr_data = np.array([])
        self.mode = 0
        self.status = ""
        self.last_directions = []
        self.results = []
        self.colors = []
        self.test = {
            "name": " ",
            "start_time": " ",
            "eval_2": " ",
            "pred_eval_2": {
                "is_right": "",
                "similarity_index": "",
                "pred_count": ""
            },
            "eval_3": " ",
            "pred_eval_3": {},
            "eval_5": " ",
            "pred_eval_5": {},
        }

        self.pred = {'counter': 0, 'dir': '', 'flag': True}
        self.rec_counter = 0
        self.monitor_counter = 0
        self.pred_counter = 0
        self.dir_sim = [0, 0, 0]
        self.right_dir = []

    def synchronize(self, msg):
        """
        Callback
        Method used to synchronize test process between ROS nodes
        @param msg: contains the info about test process
        """
        msg = msg.data.split(' ')
        self.status = msg[0]
        if self.status == "prepare":
            self.test['name'] = msg[1]
        if self.status == "start":
            time = msg[2] + ' ' + msg[3]
            date_time_obj = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
            self.test['start_time'] = date_time_obj.strftime("%H:%M:%S")
            self.timer(date_time_obj)
        if self.status == "end":
            rospy.loginfo(self.test)
            print "Monitor counter: ", self.monitor_counter
            print "Reconstr counter: ", self.rec_counter
            print "Predict counter: " , self.pred_counter
            self.results.append(self.test)
            self.test = {}
            self.pred["counter"] = 0
            self.rec_counter = 0
            self.monitor_counter = 0
            self.pred_counter = 0
            self.dir_sim = [0, 0, 0]
            self.right_dir = []

    def receive_monitor_data(self, data):
        """
        Callback
        @param data: 3d Array with N trajectories from Monitor node
        """
        # reformat data to np array
        self.last_directions = [0]
        l = data.layout.dim[1].size
        input_monitor = np.asarray(data.data)
        self.monitor_data = input_monitor.reshape(3, l, 2)
        self.monitor_counter+= 1
        i = 0
        self.right_dir.append(get_direction(self.monitor_data[0][-2], self.monitor_data[0][-1]))
        print self.right_dir
        for line in self.monitor_data:
            direction = get_direction(line[-2], line[1])
            #print direction, self.pred['dir']
            if self.pred['flag']:
                self.pred['flag'] = False
                if self.pred['dir'] == direction:
                    self.dir_sim[i] += 1
                    i +=1

    def receive_reconstruction_data(self, data):
        """
        Callback
        @param data: 2D-array of points
        """
        l = data.layout.dim[0].size
        input_reconstructor = np.asarray(data.data)
        self.reconstr_data = input_reconstructor.reshape(l, 2)
        self.rec_counter += 1

    def receive_prediction(self, data):
        """
        Callback
        @param data: Str , predicted direction
        """
        self.pred_counter += 1
        self.pred['dir'] = data.data
        self.pred['flag'] = True
        self.pred['counter'] += 1

    def find_min_ed(self):
        a = {}
        results = np.array([])
        if not self.reconstr_data.size == 0:
            tree = spatial.cKDTree(self.reconstr_data)
            for data in self.monitor_data:
                mindist, minid = tree.query(data)
                sum = np.sum(mindist)
                results = np.concatenate((results, [sum]))
            index = np.argmin(results)
            a['is_right'] = index == 0
            a['dist'] = results
        return a

    def get_pred_stats(self):
        """
        Gather statistics for predictions by evaluation
        @return:
        """
        a = {}
        index = np.argmax(self.dir_sim)
        if np.amax(self.dir_sim) != 0 or index == 0:
            a['is_right'] = True
        else:
            a['is_right'] = False
        a['sim_index'] = self.dir_sim
        a['pred_count'] = self.pred['counter']
        return a

    def timer(self, dt):
        """
        Evaluates the tests at some time points
        @param dt: Start time of running test
        """
        x, y, z = False, False, False
        while not x or not y or not z:
            if (datetime.now() - dt) >= timedelta(seconds=2) and not x:
                self.test['eval_2'] = self.find_min_ed()
                self.test['pred_eval_2'] = self.get_pred_stats()
                x = True
            if (datetime.now() - dt) >= timedelta(seconds=3) and not y:
                self.test['eval_3'] = self.find_min_ed()
                self.test['pred_eval_3'] = self.get_pred_stats()
                y = True
            if (datetime.now() - dt) >= timedelta(seconds=5) and not z:
                self.test['eval_5'] = self.find_min_ed()
                self.test['pred_eval_5'] = self.get_pred_stats()
                z = True


def main(args):
    ic = Comparator()
    rospy.init_node('comparator', anonymous=True)
    rospy.sleep(10)
    if args != "":
        Comparator.mode = args
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS comparator module")


if __name__ == '__main__':
    main(sys.argv)
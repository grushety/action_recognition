#!/usr/bin/env python

import sys
from time import sleep
from datetime import datetime

import rospy
from std_msgs.msg import String

global test_time
test_time = 7  # default test time
N_TEST = 1000


def test_controller():
    """
    Synchronize work of other nodes, by sending test status messages with additional info
    """
    commander = rospy.Publisher('/action_recognition/test_status',
                                String,
                                queue_size=10)
    for i in range(N_TEST):
        test_name = "test_" + str(i + 1)
        prepare_msg = "prepare " + test_name + " " + str(datetime.now())
        commander.publish(prepare_msg)
        sleep(2)
        start_msg = "start " + test_name + " " + str(datetime.now())
        commander.publish(start_msg)
        sleep(test_time)
        end_msg = "end " + test_name + " " + str(datetime.now())
        commander.publish(end_msg)
        sleep(3)


def main(args):
    """
    @param args: a duration of one test
    """
    if len(args) > 1:
        if args[1].isdigit():
            global test_time
            test_time = int(args[1])
    rospy.init_node('test_controller', anonymous=True)
    try:
        rospy.loginfo(args)
        test_controller()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv)

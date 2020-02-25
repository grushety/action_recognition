#!/usr/bin/env python

from random import randint
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout

def get_direction(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    if x == 0 and y == 0:
        return "zero"
    elif x == 0 and y > 0:
        return "up"
    elif x == 0 and y < 0:
        return "down"
    elif x < 0 and y == 0:
        return "left"
    elif x > 0 and y == 0:
        return "right"
    elif x < 0 and y > 0:
        return "upleft"
    elif x > 0 and y > 0:
        return "upright"
    elif x < 0 and y < 0:
        return "downleft"
    else:
        return "downright"


def denormalize_coord(coord):
    coord[0] = int(coord[0] * 1000)
    if coord[0] < 0:
        coord[0] = 0
    if coord[0] > 640:
        coord[0] = 640
    coord[1] = int(coord[1] * 1000)
    if coord[1] < 0:
        coord[1] = 0
    if coord[1] > 480:
        coord[1] = 480
    return coord


def get_random_color():
    return [randint(0, 255), randint(0, 255), randint(0, 255)]

def create_layout_3D(width, N_LINES):
    """
    @param width: Number of coordinate pairs in real trajectory
    @param N_LINES: Number of noising trajectories
    @return: Layout for 3D MultiArray message
    """
    layout = MultiArrayLayout()
    layout.dim.append(MultiArrayDimension())
    layout.dim.append(MultiArrayDimension())
    layout.dim.append(MultiArrayDimension())
    layout.dim[0].label = "trajectories"
    layout.dim[1].label = "points"
    layout.dim[2].label = "xy"
    layout.dim[0].size = N_LINES + 1
    layout.dim[1].size = width
    layout.dim[2].size = 2
    layout.dim[0].stride = N_LINES + 1
    layout.dim[1].stride = width
    layout.dim[2].stride = 2
    layout.data_offset = 0
    return layout

def create_layout_2D(length):
    layout = MultiArrayLayout()
    layout.dim.append(MultiArrayDimension())
    layout.dim.append(MultiArrayDimension())
    layout.dim[0].label = "trajectories"
    layout.dim[1].label = "points"
    layout.dim[0].size = length
    layout.dim[1].size = 2
    layout.dim[0].stride = length
    layout.dim[1].stride = 2
    layout.data_offset = 0
    return layout
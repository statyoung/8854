from math import atan2, cos, sin, sqrt, pi, tan
import pygame
import numpy as np


class Rot:
    def __init__(self, yaw):
        self.yaw = yaw

    def as_mat(self):
        return np.array([[cos(self.yaw), -sin(self.yaw)],
                         [sin(self.yaw), cos(self.yaw)]], dtype=np.float32)


class Vector2:
    def __init__(self, x, y):
        self.x_val = x
        self.y_val = y

    def as_vec(self):
        return np.array([[self.x_val], [self.y_val]], dtype=np.float32)


class Trans:
    def __init__(self, rot: Rot, disp: Vector2):
        self.rotation = rot
        self.position = disp

    def as_mat(self):
        return np.vstack((np.hstack((self.rotation.as_mat(), self.position.as_vec())), np.array([[0, 0, 1]])))


def trans_from_mat(trans_mat):
    rot_mat = trans_mat[0:2, 0:2]
    rot = Rot(yaw_from_mat(rot_mat))
    disp = Vector2(trans_mat[0, 2], trans_mat[1, 2])
    return Trans(rot, disp)


def yaw_from_mat(rot_mat):
    return atan2(rot_mat[1, 0], rot_mat[0, 0])


def rectangle_points(x, y, dx, dy, rotation=0):
    """Draw a rectangle, centered at x, y.

    Arguments:
      x (int/float):
        The x coordinate of the center of the shape.
      y (int/float):
        The y coordinate of the center of the shape.
      width (int/float):
        The width of the rectangle.
      height (int/float):
        The height of the rectangle.
      color (str):
        Name of the fill color, in HTML format.
      rotation (float):

    """
    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = sqrt((dy / 2.0) ** 2 + (dx / 2.0) ** 2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = atan2(dy / 2.0, dx / 2.0)

    # Transform that angle to reach each corner of the rectangle.
    angles = [angle, -angle + pi, angle + pi, -angle]

    # Convert rotation from degrees to radians.
    rot_radians = rotation

    # Calculate the coordinates of each point.
    for angle in angles:
        y_offset = radius * sin(angle + rot_radians)
        x_offset = radius * cos(angle + rot_radians)
        points.append([x + x_offset, y + y_offset])
    points = np.array(points)
    return points

def trailer_rectangle_points(x, y, dx, dy, rotation=0):
    """Draw a rectangle, centered at x, y.

    Arguments:
      x (int/float):
        The x coordinate of the center of the shape.
      y (int/float):
        The y coordinate of the center of the shape.
      width (int/float):
        The width of the rectangle.
      height (int/float):
        The height of the rectangle.
      color (str):
        Name of the fill color, in HTML format.
      rotation (float):

    """
    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = sqrt((dy / 2.0) ** 2 + dx ** 2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = atan2(dy / 2.0, -dx)

    # Transform that angle to reach each corner of the rectangle.
    angles = [np.pi/2, angle, -angle, -np.pi/2]

    # Convert rotation from degrees to radians.
    rot_radians = rotation

    # Calculate the coordinates of each point.
    points.append([x+(dy/2)*cos(angles[0]+rot_radians), y+(dy/2)*sin(angles[0]+rot_radians)])
    points.append([x+radius*cos(angles[1]+rot_radians), y+radius*sin(angles[1]+rot_radians)])
    points.append([x+radius*cos(angles[2]+rot_radians), y+radius*sin(angles[2]+rot_radians)])
    points.append([x+(dy/2)*cos(angles[3]+rot_radians), y+(dy/2)*sin(angles[3]+rot_radians)])
    points = np.array(points)
    return points


def gen_from_ackerman(r_ack, ang_vel_ack, L):
    x_vel = r_ack * ang_vel_ack
    y_vel = L * ang_vel_ack
    return x_vel, y_vel, ang_vel_ack


def gen_from_rc(v_r, phi, L):
    x_vel = v_r * cos(phi)
    y_vel = v_r * sin(phi)
    yaw_rate = y_vel / L
    return x_vel, y_vel, yaw_rate
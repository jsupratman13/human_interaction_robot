#!/usr/bin/env python
from __future__ import print_function
import roslib
import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt
import threading
import readchar
import copy

fig, ax = plt.subplots(1, 1)
robot_line, = ax.plot([0, 0], [0, 0], lw=5)
human_line, = ax.plot([0, 0], [0, 0], lw=5)
arm1_line, = ax.plot([0, 0], [0, 0], lw=2)
arm2_line, = ax.plot([0, 0], [0, 0], lw=2)
plt.axis([-5, 5, 0, 2])

def draw_graphics(self):
    robot_line.set_data([self.robot_x ,self.robot_x], [0, 1])
    human_line.set_data([self.human_x ,self.human_x], [0 ,1])
    wrist_angle_rad = self.state.position[0]
    elbow_pos = [self.robot_x + self.ARM_LENGTH * math.cos(wrist_angle_rad), 1 - self.ARM_LENGTH * math.sin(wrist_angle_rad)]
    arm1_line.set_data([self.robot_x, elbow_pos[0]], [1, elbow_pos[1]])
    arm2_line.set_data([self.human_x, elbow_pos[0]], [1, elbow_pos[1]])
    plt.pause(0.01)

class dummy_robot:
    DURATION = 0.1
    speed_up_factor = 3
    WIDTH = 0.3
    RANGE = 0.1
    ARM_LENGTH = 0.3

    def __init__(self):
        rospy.init_node('dummy_robot', anonymous=True)
        self.joint_state_pub = rospy.Publisher('/manipulator/joint_states', JointState, queue_size=1)
        self.action_sub = rospy.Subscriber('/icart_mini/cmd_vel', Twist, self.callback_action)
        self.timer = rospy.Timer(rospy.Duration(dummy_robot.DURATION), self.callback_joint_state_timer)
        self.getkey_thread = threading.Thread(target=self.getkey)
        self.getkey_thread.start()
        self.key = ''
        self.vel_x = 0
        self.robot_x = 0
        self.human_x = dummy_robot.WIDTH
        self.state = JointState()
        self.state.position = [0.0, 0.0, 0.0, 0.0]
        self.state.effort = [0.0, 0.0, 0.0, 0.0]
        self.init_pos = [0.0, 0.0, 0.0, 0.0]
        self.calc_inverse_kinematics(True)
        self.vel_x1 = self.vel_x2 = 0

    def callback_action(self, data):
        self.vel_x = data.linear.x

    def callback_joint_state_timer(self, data):
        self.robot_x += self.vel_x * dummy_robot.DURATION * dummy_robot.speed_up_factor
        self.vel_x2 = self.vel_x1
        self.vel_x1 = self.vel_x
        self.human_x += 0.05 if self.key == '\\' else 0.0
        self.human_x -= 0.05 if self.key == '/' else 0.0
        if self.key == 'r':
            rospy.loginfo('RESET\r')
            self.vel_x = 0
            self.robot_x = 0
            self.human_x = dummy_robot.WIDTH
        human_neutral_pos = self.robot_x + dummy_robot.WIDTH
        self.human_x = human_neutral_pos + max([min([(self.human_x - human_neutral_pos), dummy_robot.RANGE]), -dummy_robot.RANGE])
        self.calc_inverse_kinematics()
        if self.key == 'q':
            quit()
        self.key = ''
        self.joint_state_pub.publish(self.state)

    def calc_inverse_kinematics(self, is_first = False):
        distance = self.human_x - self.robot_x
        wrist_angle_rad = math.acos(distance/2/self.ARM_LENGTH)
        self.state.position[0] =  wrist_angle_rad
        self.state.position[1] = -wrist_angle_rad * 2
        self.state.position[3] =  wrist_angle_rad
        if is_first:
            self.init_pos = copy.deepcopy(self.state.position)
        self.state.effort[0] =  wrist_angle_rad - self.init_pos[0]
        self.state.effort[1] = -wrist_angle_rad * 2 - self.init_pos[1]
        self.state.effort[3] =  wrist_angle_rad - self.init_pos[3]

    def getkey(self):
        while True:
            self.key = readchar.readkey()

if __name__ == '__main__':
    dr = dummy_robot()
    r = rospy.Rate(1/dr.DURATION)
    while not rospy.is_shutdown():
        draw_graphics(dr)
        r.sleep()

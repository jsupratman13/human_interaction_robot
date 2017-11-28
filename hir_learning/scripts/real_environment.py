#!/usr/bin/python
import math,random
import sys,abc,time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

class Environment(object):
    __metaclass__ = abc.ABCMeta

    FORWARD = 0
    STOP = 1
    REVERSE = 2

    PUSH = 0
    NONE = 1
    PULL = 2

    def __init__(self):
        
        self.state = []
        self.vel_error = []
        self.pos_error = []
        self.joint_names = []
        self.initial_step_time = time.time()

        self.contact = Environment.NONE

        self.__observation_space = Environment.ObservationSpace()
        self.__action_space = Environment.ActionSpace()

        self.prev_action = Environment.STOP
        self.step_time = 0

        self.sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.__get_state)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)

    @property
    def action_space(self):
        return self.__action_space
    @property
    def observation_space(self):
        return self.__observation_space

    def __get_state(self, msg):
        self.state = [msg.effort[0],msg.effort[1], msg.effort[3]]
    
    def __move(self, action):
        vel = Twist()
        if action == Environment.STOP:
            vel.linear.x = 0
        elif action == Environment.REVERSE:
            vel.linear.x = -0.2
        elif action == Environment.FORWARD:
            vel.linear.x = 0.2
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

    def get_reward(self, action):
        #if self.contact == Environment.NONE and action == Environment.STOP:
        #    return 100
        #elif self.contact == Environment.PUSH and action == Environment.REVERSE:
        #    return 100
        #elif self.contact == Environment.PULL and action == Environment.FORWARD:
        #    return  100
        #return 0

        #reward =  -1 * sum([math.fabs(s) for s in self.state])
    
        return -1 * (math.fabs(self.state[0])+math.fabs(self.state[1]))

    def reset(self, test=0):
        self.contact = random.choice([Environment.NONE, Environment.PUSH, Environment.PULL])
        self.initial_step_time = time.time()
        self.step_time = 0
        
        return self.state


    def step(self, action):
        is_terminal = False
        reward = self.get_reward(action)
       
        self.step_time += 1
        self.__move(action)
        self.prev_action = action

        #if math.fabs(time.time()-self.initial_step_time) > 10:
        if self.step_time > 200:
            is_terminal = True
        
        if self.sub.get_num_connections() != 1:
            self.state = []
            print 'lost connection'

        return self.state, reward, is_terminal

    class ObservationSpace(object):
        def __init__(self):
            pass

        def get_size(self):
            return 3

    class ActionSpace(object):
        def __init__(self):
            self.action_list = [Environment.FORWARD,
                                Environment.STOP,
                                Environment.REVERSE]
        
        def sample(self):
            return random.choice(self.action_list)
    
        def get_size(self):
            return len(self.action_list)


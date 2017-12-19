#!/usr/bin/python
import math,random
import sys,abc,time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

class Environment(object):
    __metaclass__ = abc.ABCMeta

    FORWARD = 1
    STOP = 0
    REVERSE = 2

    PUSH = 0
    NONE = 1
    PULL = 2

    def __init__(self):
        
        self.state = []
        self.vel_error = [0, 0, 0]
        self.pos_error = []
        self.prev_pos_error = [0, 0, 0]
        self.joint_names = []
        self.initial_step_time = time.time()

        self.contact = Environment.NONE

        self.__observation_space = Environment.ObservationSpace()
        self.__action_space = Environment.ActionSpace()

        self.prev_action = Environment.STOP
        self.step_time = 0

        self.sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.__get_state)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)

        self.f = open('data.csv', 'w')
        self.initial_flag  = True

    @property
    def action_space(self):
        return self.__action_space
    @property
    def observation_space(self):
        return self.__observation_space

    def __get_state(self, msg):
        self.joint_names = list(msg.name)
        self.pos_error = list(msg.effort)

        self.state = [msg.effort[0], msg.effort[1], msg.effort[3]]
        for i in range(len(self.vel_error)):
            self.vel_error[i] = (self.state[i] - self.prev_pos_error[i])/0.2
        for j in range(len(self.prev_pos_error)):
            self.prev_pos_error[i] = self.state[i]

        self.state = [msg.effort[0],msg.effort[1], msg.effort[3], self.vel_error[0], self.vel_error[1], self.vel_error[2]]
    
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

    def collect(self,action):
        if self.initial_flag:
            self.initial_flag = False
            self.f.write('step,')
            for i in self.joint_names:
                self.f.write(str(i)+',')
            self.f.write('action\n')
        self.f.write(str(self.step_time)+',')
        for j in self.pos_error:
            self.f.write(str(j)+',')
        self.f.write(str(action)+'\n')

    def get_reward(self, action):
        reward = 0
        if self.contact == Environment.NONE and action == Environment.STOP:
           reward = 100
        elif self.contact == Environment.PUSH and action == Environment.REVERSE:
            reward = 100
        elif self.contact == Environment.PULL and action == Environment.FORWARD:
            reward = 100
        #reward =  -1 * sum([math.fabs(s) for s in self.state])
        #reward = -1 * (math.fabs(self.state[0])+math.fabs(self.state[1]))
        return reward

    def reset(self, test=0):
        self.contact = random.choice([Environment.NONE, Environment.PUSH, Environment.PULL])
        self.initial_step_time = time.time()
        self.step_time = 0
        
        return self.state


    def step(self, action, joy=None):
        if joy is not None:
            self.contact = joy

        is_terminal = False
        reward = self.get_reward(action)
       
        self.step_time += 1
        self.__move(action)
        self.prev_action = action

        #self.collect(action)

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
            return 6

    class ActionSpace(object):
        def __init__(self):
            self.action_list = [Environment.FORWARD,
                                Environment.STOP,
                                Environment.REVERSE]
        
        def sample(self):
            return random.choice(self.action_list)
    
        def get_size(self):
            return len(self.action_list)


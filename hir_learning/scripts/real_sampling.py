#!/usr/bin/python

import numpy as np
import rospy
import time, collections, random, sys
from keras.optimizers import Adam
from keras.models import model_from_json
import json
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from real_environment import Environment
import ConfigParser

class Robot(object):
    def __init__(self, model, weight, filename):
        self.sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.__get_state)
        rospy.Subscriber('/icart_mini/odom', Odometry, self.__get_odom)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)

        self.joint_names = []
        self.error = []
        self.odom = []
        self.initial_flag = True
        self.index = 0 
        self.f = open(filename, 'w')

        config = ConfigParser.RawConfigParser()
        config.read('real_parameters.cfg')
        self.epsilon = config.getfloat('epsilon_greedy', 'epsilon')
        self.epsilon_decay = config.getfloat('epsilon_greedy', 'epsilon_decay')
        self.min_epsilon = config.getfloat('epsilon_greedy', 'min_epsilon')
        self.alpha = config.getfloat('training', 'alpha')

        self.target_model = self.load_model(model)
        self.target_model.load_weights(weight)
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.nstates = self.target_model.get_input_shape_at(0)[1]

    def load_model(self, filename):
        json_file = open(filename, 'r')
        model = model_from_json(json_file.read())
        json_file.close()
        return model

    def step(self, action):
        vel = Twist()
        if action == Environment.FORWARD:
            vel.linear.x = 0.2
        elif action == Environment.REVERSE:
            vel.linear.x = -0.2
        else:
            vel.linear.x = 0.0
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

    def epsilon_greedy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([Environment.FORWARD, Environment.STOP, Environment.REVERSE])
        else:
            Q = self.target_model.predict(state)
            return np.argmax(Q[0])

    def __get_state(self, msg):
        self.joint_names = list(msg.name)
        self.error = list(msg.effort)

    def __get_odom(self, msg): 
        self.odom = []
        self.odom.append(msg.twist.twist.linear.x)
        self.odom.append(msg.twist.twist.angular.z)

    def save(self, s, a, r, s2, done):
        state = list(s[0])
        state2 = list(s2[0])
        for i in state:
            self.f.write(str(i)+',')
        self.f.write(str(a)+',')
        self.f.write(str(r)+',')
        for j in state2:
            self.f.write(str(j)+',')
        self.f.write(str(done)+'\n')

    def get_reward(self, s, s2, a):
        reward = -1 * sum([math.fabs(state) for state in s2])   
        return 100

    def sample(self, rate):
        step = 0
        try:
            while not self.error:
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'

        for i in self.joint_names:
            self.f.write(str(i)+'_1,')
        self.f.write('action,reward,')
        for j in self.joint_names:
            self.f.write(str(j)+'_2,')
        self.f.write('done\n')

        s = self.error
        self.nstates = 5#len(self.error)
        s = np.reshape(s, [1, self.nstates])
        while not rospy.is_shutdown():
            step += 1
            if not self.sub.get_num_connections():
                break

            a = self.epsilon_greedy(s)
            self.step(a)
            rate.sleep()
            s2 = np.reshape(self.error, [1,self.nstates])
            r = self.get_reward(s,s2,a)
            done = True if step > 500 else False
            self.save(s, a, r, s2, done)
            print 'step: ' + str(step) + ' action: ' + str(a) + ' reward: ' + str(r) + ' epsilon: ' + str(self.epsilon)
            s = s2

            if step > 500:
                raw_input('reset simulation')
                step = 0
                if self.epsilon > self.min_epsilon:
                    self.epsilon *= self.epsilon_decay
            

if __name__ == '__main__':
    rospy.init_node('collecting_data_node', disable_signals=True)
    rospy.loginfo('start')
    robot = Robot(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
    i = 0
    rate = rospy.Rate(50)
    try:
        robot.sample(rate)

    except KeyboardInterrupt:
        pass

    finally:
        robot.f.close()
        rospy.loginfo('finished')

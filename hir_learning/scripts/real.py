#!/usr/bin/python

import numpy as np
import json
import collections,random,sys
from keras.models import model_from_json
from keras.optimizers import Adam
import rospy, time, sys
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from environment import Environment

class Agent(object):
    def __init__(self, model, weight):
        self.alpha = 0.001
        self.ntrials = 6
        self.nstates = 6
        self.model = self.load_model(model)
        self.weight = weight
        rospy.Subscriber('/manipulator/joint_states', JointState, self.__get_state)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
        self.error = []

    def load_model(self,filename):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        json_file.close() 
        return model

    def epsilon_greedy(self,state):
        Q = self.model.predict(state)
        return np.argmax(Q[0])

    def __get_state(self, msg):
        self.error = list(msg.effort)

    def reset(self):
        vel = Twist()
        vel.linear.x = 0
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)
        return self.error

    def step(self, action):
        vel = Twist()
        if action == Environment.FORWARD:
            print 'forward'
            vel.linear.x = 0.5
        elif action == Environment.REVERSE:
            print 'reverse'
            vel.linear.x = -0.5
        else:
            print 'stop'
            vel.linear.x = 0.0

        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)
        return self.error, False

    def test(self):
        self.model.load_weights(self.weight)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.epsilon = 0.001
        self.error = [1]
        try:
            while not self.error:
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'
        
        s = self.reset()
        s = np.reshape(s,[1,self.nstates])
        done = False
        while not rospy.is_shutdown():
            if self.error:
                a = self.epsilon_greedy(s)
                s2, done = self.step(a)
                s = np.reshape(s2, [1,self.nstates])
            self.error = []

if __name__ == '__main__':
    if not len(sys.argv) > 2:
        assert False, 'missing model and/or weight'
    rospy.init_node('ddqn_test', disable_signals=True)
    rospy.loginfo('start testing')
    agent = Agent(str(sys.argv[1]),str(sys.argv[2]))
    try:
        agent.test()
    except (KeyboardInterrupt,SystemExit):
        sys.exit()
        agent.reset()
        rospy.loginf('finished testing')

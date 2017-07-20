#!/usr/bin/python

import numpy as np
import gym
import json
import collections,random,sys
from keras.models import model_from_json
from keras.optimizers import Adam
import rospy
from environment import Environment

class Agent(object):
    def __init__(self, env, model, weight):
        self.alpha = 0.001
        self.env = env
        self.ntrials = 5
        self.nstates = env.observation_space.get_size()
        self.model = self.load_model(model)
        self.weight = weight
        self.configure = {}

    def load_model(self,filename):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        json_file.close() 
        return model

    def epsilon_greedy(self,state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            Q = self.model.predict(state)
            return np.argmax(Q[0])

    def train(self):
        self.model.load_weights(self.weight)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.epsilon = 0.1
        for trial in range(self.ntrials):
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates])
            treward = 0
            while True:
                a = self.epsilon_greedy(s)
                s2, r, done = self.env.step(a)
                s = np.reshape(s2, [1,self.nstates])
                treward += r
                if done:
                    rospy.loginfo('test: ' + str(trial+1) + ' reward: ' + str(treward))
                    break

if __name__ == '__main__':
    if not len(sys.argv) > 2:
        assert False, 'missing model and/or weight'
    rospy.init_node('ddqn_test')
    rospy.loginfo('start testing')
    env = Environment()
    agent = Agent(env, str(sys.argv[1]),str(sys.argv[2]))
    agent.train()
    env.reset()
    rospy.loginfo('COMPLETE TESTING')
    rospy.spin()

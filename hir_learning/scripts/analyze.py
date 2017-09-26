#!/usr/bin/python

import numpy as np
import json
import collections,random,sys
from keras.models import model_from_json
from keras.optimizers import Adam
import rospy
from environment import Environment

class Agent(object):
    def __init__(self,env,model,num_weights):
        self.alpha = 0.001
        self.env = env
        self.ntrials = 10
        self.nstates = env.observation_space.get_size()
        self.model = self.load_model(model)
        self.num_weights = num_weights
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

    def test_weight(self,weightname):
        self.model.load_weights(weightname)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.epsilon = 0.1
        reward = []
        for trial in range(self.ntrials):
            s = self.env.reset(test=trial)
            s = np.reshape(s,[1,self.nstates])
            treward = []
            while True:
                a = self.epsilon_greedy(s)
                s2, r, done = self.env.step(a)
                s = np.reshape(s2, [1,self.nstates])
                treward.append(r)
                if done:
                    reward.append(sum(treward)/len(treward))
                    break
        return reward

    def run_diagnostic(self):
        for i in range(self.num_weights+1):
            print 'diagnose ' + str(i) + '/' + str(self.num_weights+1)
            if i == self.num_weights:
                weight = 'episodefinal.hdf5'
            else:
                weight = 'episode'+str(i)+'.hdf5'
            reward = self.test_weight(weight)
            avg_reward = sum(reward)/len(reward)
            self.configure[weight] = avg_reward
        f = open('result.csv', 'w')
        for key, value in sorted(self.configure.iteritems(), key=lambda(k,v): (v,k)):
            f.write(str(key)+','+str(value)+'\n')
        f.close()
     
if __name__ == '__main__':
    try:
        if not len(sys.argv) > 2:
            assert False, 'missing model and/or weight'
        rospy.init_node('ddqn_diagnose')
        rospy.loginfo('start diagnostic')
        env = Environment()
        weights = []
        for i in range(2,len(sys.argv)):
            weights.append(str(sys.argv[i]))
        agent = Agent(env, (str(sys.argv[1])),weights)
        agent.run_diagnostic()
        env.reset()
        rospy.loginfo('COMPLETE DIAGNOSE')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except (KeyboardInterrupt, SystemExit):
        pass

#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ddqn.py                             
#brief: double deep q-learning on neural network                  
#author: Joshua Supratman                    
#last modified: 2017.10.21. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import rospy
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections,random,sys,time
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from real_environment import Environment
import ConfigParser
import traceback

class Agent(object):
    def __init__(self,env):
        config = ConfigParser.RawConfigParser()
        config.read('real_parameters.cfg')
        self.gamma = config.getfloat('training','gamma')
        self.alpha = config.getfloat('training', 'alpha')
        self.nepisodes = config.getint('training', 'episodes')
        self.epsilon = config.getfloat('epsilon_greedy', 'epsilon')
        self.min_epsilon = config.getfloat('epsilon_greedy', 'min_epsilon')
        self.epsilon_decay = config.getfloat('epsilon_greedy', 'epsilon_decay')
        self.batch_size = config.getint('network', 'batch_size')
        self.updateQ = config.getint('network', 'update_network')
        self.weights_name = 'episodefinal.hdf5'
        np.random.seed(123)
        self.env = env
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        self.model = self.create_neural_network()
        self.memory = collections.deque(maxlen=config.getint('network', 'memory_size'))
        self.target_model = self.model
        self.loss_list = []
        self.reward_list = []

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(100,input_dim=self.nstates, activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(self.nactions,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        model_json = model.to_json()
        with open('hir_model.json','w') as json_file:
            json_file.write(model_json)
        return model

    def load_model(self,filename, weight, epsilon):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        json_file.close() 
        self.model = model
        self.model.load_weights(weight)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.epsilon = epsilon

    def epsilon_greedy(self,state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            Q = self.model.predict(state)
            return np.argmax(Q[0])

    def train(self):
        max_r = -1000000
        try:
            while not self.env.state:
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'

        max_r = -1000000
        rate = rospy.Rate(20)
        for episode in range(self.nepisodes):
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates]) 
            treward = []
            loss = 0
            raw_input('start new episode')
            while not rospy.is_shutdown():
                if self.env.state:
                    a = self.epsilon_greedy(s)
                    s2, r, done = self.env.step(a)
                    s2 = np.reshape(s2, [1,self.nstates])
                    self.memory.append((s,a,r,s2,done))
                    s = s2
                    treward.append(r)
                else:
                    done = False
                rate.sleep()
                if done:
                    break
            treward = sum(treward)/len(treward)

            #save checkpoint
            self.model.save_weights('episode'+str(episode)+'.hdf5')

            #replay experience
            if len(self.memory) > self.batch_size:
                loss = self.replay()
            self.loss_list.append(loss)
            self.reward_list.append(treward)
            
            print 'episode: ' + str(episode+1) + ' reward: ' + str(treward) + ' epsilon: ' + str(round(self.epsilon,2)) + ' loss: ' + str(round(loss,4))

            #Target Network
            if not episode % self.updateQ:
                #self.target_model = self.model
                self.target_model.set_weights(self.model.get_weights())

            #shift from explore to exploit
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            
        self.model.save_weights(self.weights_name)

    def replay(self):
        minibatch = random.sample(self.memory,self.batch_size)
        loss = 0.0
        for s,a,r,s2,done in minibatch:
            Q = r if done else r + self.gamma * self.target_model.predict(s2)[0][np.argmax(self.model.predict(s2)[0])]
            target = self.target_model.predict(s)
            target[0][a] = Q
            loss += self.model.train_on_batch(s,target)
        return loss/len(minibatch)

if __name__ == '__main__':
    rospy.init_node('online_ddqn', disable_signals=True)
    rospy.loginfo('START ONLINE TRAINING')
    env = Environment()
    agent = Agent(env)
    start_time = time.time()
    if len(sys.argv) > 3:
        agent.load_model(str(sys.argv[1]),str(sys.argv[2]),float(sys.argv[3]))

    try:
        agent.train()
        agent.plot()
        env.reset()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rospy.loginfo('EXIT TRAINING')

#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ddqn.py                             
#brief: double deep q-learning on neural network                  
#author: Joshua Supratman                    
#last modified: 2017.09.25. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import numpy as np
import rospy
import json
import matplotlib.pyplot as plt
import collections,random,sys
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from environment import Environment
import analyze

class Agent(object):
    def __init__(self,env):
        self.gamma = 0.95
        self.alpha = 0.01
        self.nepisodes = 4
        self.epsilon = 0.2
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.updateQ = 100
        self.weights_name = 'episodefinal.hdf5'
        self.env = env
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        self.model = self.create_neural_network()
        self.memory = collections.deque(maxlen=2000)
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
        max_r = -1000000
        for episode in range(self.nepisodes):
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates]) 
            treward = []
            loss = 0
            while True:
                a = self.epsilon_greedy(s)
                s2, r, done = self.env.step(a)
                s2 = np.reshape(s2, [1,self.nstates])
                self.memory.append((s,a,r,s2,done))
                s = s2
                treward.append(r)
                if done:
                    break
            treward = sum(treward)

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

    def plot(self):
        ep = np.arange(0,self.nepisodes, 1)
        plt.figure(1)
        plt.plot(ep, self.reward_list)
        plt.xlabel('episodes')
        plt.ylabel('reward')
        plt.savefig('reward.png')
        plt.figure(2)
        plt.plot(ep, self.loss_list)
        plt.xlabel('episodes')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.savefig('loss.png')
        plt.show()

if __name__ == '__main__':
    try:
        rospy.init_node('ddqn_learning')
        rospy.loginfo('start training')
        env = Environment()
        agent = Agent(env)
        diagnose = analyze.Agent(env, 'hir_model.json', agent.nepisodes)
        #agent.train()
        #agent.plot()
        rospy.loginfo('COMPLETE TRAINING')
        env.reset()
        rospy.loginfo('start diagnose')
        diagnose.run_diagnostic()
        env.reset()
        rospy.loginfo('COMPLETE DIAGNOSTIC')
        rospy.spin() 
    except rospy.ROSInterruptException:
        print 'kill node' 
        pass

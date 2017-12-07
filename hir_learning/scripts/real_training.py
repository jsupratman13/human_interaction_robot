#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: real_training.py                             
#brief: double deep q-learning on neural network                  
#author: Joshua Supratman                    
#last modified: 2017.12.07. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections,random,sys,time
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
import ConfigParser
import traceback

class Agent(object):
    def __init__(self, modelname, target_weight, target_name, sample_data):
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
        self.memory = collections.deque(maxlen=config.getint('network', 'memory_size'))
        
        self.weights_name = target_name
        np.random.seed(123)
        
        self.model = self.load_model(modelname)
        self.model.load_weights(target_weight)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        self.target_model = self.load_model(modelname)
        self.target_model.load_weights(target_weight)
        self.target_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        
        self.loss_list = []
        self.load_sample(sample_data)

    def load_model(self,filename):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        json_file.close() 
        return model

    def load_sample(self, filename):
        samples = open(filename, 'r')
        state_index = []
        state2_index = []
        line = samples.readline().strip('\n')
        line = line.split(',')
        for i in range(len(line)):
            if line[i].endswith('_1'):
                state_index.append(i)
            elif line[i] == 'action':
                action_index = i
            elif line[i] == 'reward':
                reward_index = i
            elif line[i].endswith('_2'):
                state2_index.append(i)
            elif line[i] == 'done':
                done_index = i
       
        for sample in samples:
            line = sample.split(',')
            s = []
            s2 = []
            for j in state_index:
                s.append(float(line[j]))
            s = np.reshape(s, [1, len(s)])
            a = int(line[action_index])
            r = float(line[reward_index])
            for k in state2_index:
                s2.append(float(line[k]))
            s2 = np.reshape(s2, [1, len(s2)])
            done = bool(line[done_index]=='True')
            
            self.memory.append((s,a,r,s2,done))

    def train(self):
        for episode in range(self.nepisodes):
            loss = 0

            #save checkpoint
            self.model.save_weights('episode'+str(episode)+'.hdf5')

            #replay experience
            if len(self.memory) > self.batch_size:
                loss = self.replay()
            
            print 'episode: ' + str(episode+1) + ' loss: ' + str(round(loss,4))
            self.loss_list.append(loss)

        # target network
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
        plt.plot(ep, self.loss_list)
        plt.xlabel('episodes')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.savefig('loss.png')
        #plt.show()

if __name__ == '__main__':
    if not len(sys.argv) > 3:
        nstate = int(sys.argv[1])
        naction = int(sys.argv[2])
        config = ConfigParser.RawConfigParser()
        config.read('real_parameters.cfg')
        alpha = config.getfloat('training', 'alpha')
        
        model = Sequential()
        model.add(Dense(100,input_dim=nstate, activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(naction,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=alpha))
        model_json = model.to_json()
        with open('hir_model.json','w') as json_file:
            json_file.write(model_json)
        model.save_weights('initial_weight.hdf5')
        
        print 'initialize model and weight with ' + str(nstate)+' states ' + str(naction) + ' actions '
        sys.exit()

    try:
        print 'start training'
        start_time = time.time()
        agent = Agent(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
        agent.train()
        agent.plot()
        print 'COMPLETE TRAINING'
        m,s = divmod(time.time()-start_time, 60)
        h,m = divmod(m, 60)
        print 'time took %d:%02d:%02d' %(h,m,s)
    except Exception, e:
        traceback.print_exc()
        sys.exit()

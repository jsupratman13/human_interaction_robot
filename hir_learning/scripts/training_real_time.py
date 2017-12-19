#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ddqn.py                             
#brief: double deep q-learning on neural network                  
#author: Joshua Supratman                    
#last modified: 2017.12.12. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import rospy
import numpy as np
import json
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import collections,random,sys,time
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from real_environment import Environment
import ConfigParser
import traceback
from sensor_msgs.msg import Joy

class Agent(object):
    def __init__(self,env):
        rospy.Subscriber('/joy',Joy,self.get_joy)
        self.joy = [Environment.NONE, 0]

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
        self.weights_name = 'episodefinal.hdf5'
        
        np.random.seed(123)
        self.env = env
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        
        self.model = self.create_neural_network()
        self.target_model = self.create_neural_network()
        self.target_model.set_weights(self.model.get_weights())
        
        self.loss_list = []
        self.reward_list = []
        self.episode = None

    def get_joy(self, msg):
        force = msg.axes[1]
        reset = msg.buttons[0]
        if force > 0.5:
            force = Environment.PUSH
        elif force < -0.5:
            force = Environment.PULL
        else:
            force = Environment.NONE

        self.joy = [force, reset]

    def create_neural_network(self):
        model = Sequential()
        model.add(Dense(100,input_dim=self.nstates, activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(self.nactions,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def save_model(self, model, filename):
        model_json = model.to_json()
        with open(filename,'w') as json_file:
            json_file.write(model_json)

    def load_model(self, filename, weight):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        json_file.close() 
        self.model = model
        self.model.load_weights(weight)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def epsilon_greedy(self,state):
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            Q = self.target_model.predict(state)
            action = np.argmax(Q[0])
        if action == Environment.STOP:
            name = 'STOP'
        elif action == Environment.FORWARD:
            name = 'FORWARD'
        else:
            name = 'REVERSE'
        return action, name

    def wait_keyboard_input(self):
        info = 'start new episode'
        if self.env.contact == Environment.NONE:
            info+=' no contact'
        elif self.env.contact == Environment.PUSH:
            info+=' push arm'
        elif self.env.contact == Environment.PULL:
            info+=' pull arm'
        #raw_input(info)
        while not self.joy[1]:
            pass

    def train(self):
        max_r = -1000000
        try:
            while not self.env.state:
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'

        max_r = -1000000
        rate = rospy.Rate(5)
        for episode in range(self.nepisodes):
            if self.episode and self.episode > episode:
                continue
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates]) 
            treward = []
            loss = 0
            step = 0
            self.wait_keyboard_input()
            while not rospy.is_shutdown():
                if self.env.state:
                    a, name = self.epsilon_greedy(s)
            
                    rate.sleep()
                    s2, r, done = self.env.step(a, joy=self.joy[0])
                    s2 = np.reshape(s2, [1,self.nstates])
                    self.memory.append((s,a,r,s2,done))
                    s = s2
                    treward.append(r)
                    print 'step: ' + str(step) + ' reward: ' + str(r) + ' action: ' + name + ' epsilon: ' + str(round(self.epsilon,2)) 
                    step += 1
                else:
                    done = False

                if done:
                    break
            treward = sum(treward)/len(treward)

            #replay experience
            if len(self.memory) > self.batch_size:
                loss = self.replay()
            self.loss_list.append(loss)
            self.reward_list.append(treward)
            
            print 'episode: ' + str(episode+1) + ' average reward: ' + str(treward) +' loss: ' + str(round(loss,4))

            #Target Network
            if not episode % self.updateQ:
                #self.target_model = self.model
                self.target_model.set_weights(self.model.get_weights())
                self.model.save_weights('target'+str(episode)+'.hdf5')
            else:
                #save checkpoint
                self.model.save_weights('episode'+str(episode)+'.hdf5')

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
        file1 = open('result.csv', 'a')
        file1.write('episode,reward,loss\n')
        if self.episode:
            episodes = range(self.episode, self.episode+len(self.reward_list),1)
        else:
            episodes = range(len(self.reward_list))
        for i in episodes:
            file1.write(str(i))
            file1.write(','+str(self.reward_list[i]))
            file1.write(','+str(self.loss_list[i]))
            file1.write('\n')
        file1.close()

if __name__ == '__main__':
    rospy.init_node('online_ddqn', disable_signals=True)
    rospy.loginfo('START ONLINE TRAINING')
    env = Environment()
    agent = Agent(env)
    start_time = time.time()
    if len(sys.argv) > 3:
        # modelname target weight, current weight, epsilon, nth episode
        agent.target_model = agent.load_model(str(sys.argv[1]),str(sys.argv[2]))
        agent.model = agent.load_model(str(sys.argv[1]), str(sys.argv[3]))
        agent.epsilon = float(sys.argv[4])
        agent.episode = int(sys.argv[5])
        
    else:
        agent.save_model(agent.model, 'initial_model.json')

    try:
        agent.train()
        env.reset()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        agent.plot()
        m,s = divmod(time.time()-start_time, 60)
        h,m = divmod(m,60)
        rospy.loginfo('time took %d:%02d:%02d' %(h,m,s))
        rospy.loginfo('EXIT TRAINING')

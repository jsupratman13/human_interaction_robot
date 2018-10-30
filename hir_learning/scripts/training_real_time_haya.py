#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ddqn.py                             
#brief: double deep q-learning on neural network                  
#author: Joshua Supratman                    
#last modified: 2017.12.12. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import chainerrl
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import os
from os.path import expanduser

import rospy
import collections,random,sys,time,math,copy


from real_environment import Environment
from sensor_msgs.msg import Joy
import pygame

class QFunction(chainer.Chain):
    def __init__(self, n_state=6, n_action=3):
        initializer = chainer.initializers.HeNormal()
        super(QFunction, self).__init__(
            fc1 = L.Linear(n_state, 10, initialW=initializer),
            fc2 = L.Linear(10, 10, initialW=initializer),
            fc3 = L.Linear(10, n_action, initialW=initializer)
        )

    def __call__(self, x, test=False):
        s = chainer.Variable(x.astype(np.float32))
        h1 = self.fc1(s)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        h = chainerrl.action_value.DiscreteActionValue(h3)
        return h

class reinforcement_learning:
    def __init__(self, n_state=6, n_action=3):
        self.q_func = QFunction(n_state, n_action)
        try:
            self.q_func.to_gpu()
        except:
            print("No GPU")
        self.optimizer = chainer.optimizers.Adam(eps=0.01)
        self.optimizer.setup(self.q_func)
        self.gamma = 0.0
        self.n_action = n_action
        self.explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.0, random_action_func=self.action_space_sample)
        self.replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 4)
#        self.phi = lambda x: x.astype(np.float32, copy=False)
#        self.phi = 0
        self.agent = chainerrl.agents.DoubleDQN(
            self.q_func, self.optimizer, self.replay_buffer, self.gamma, self.explorer,
            minibatch_size=4, replay_start_size=10, update_interval=1,
            target_update_interval=10)
        home = expanduser("~")
        if os.path.isdir(home + '/agent'):
            self.agent.load('agent')
            print('agent LOADED!!')

    def act_and_trains(self, obs, reward):
        self.action = self.agent.act_and_train(obs, reward)
        if (10 * np.random.rand()) < (- reward):
            self.action = self.action_space_sample()
        return self.action

    def stop_episode_and_train(self, obs, reward, done):
        self.agent.stop_episode_and_train(obs,reward,done)

    def act(self, obs):
        self.action = self.agent.act(obs)
        return self.action

    def save_agent(self, file_name):
        self.agent.save(file_name)
        print("agent SAVED!!")

    def action_space_sample(self):
        return np.random.randint(0,self.n_action)

class Agent(object):
    def __init__(self,env):
        pygame.init()

        rospy.Subscriber('/joy',Joy,self.get_joy)
        self.joy = [Environment.NONE, 0]
        self.rl = reinforcement_learning()
        self.nepisodes = 15

        self.env = env
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        self.action = Environment.STOP
        self.state = [0,0,0,0,0,0]
        self.reward = 0
        self.integral_reward = 0

        self.loss_list = []
        self.episode = None

        file2 = open('training2.csv', 'a')
        file2.write('episode,step,reward,action,state\n')
        file2.close()

    def get_joy(self, msg):
        force1 = msg.axes[0]
        force2 = msg.axes[1]
        reset = msg.buttons[0]
        if math.fabs(force1) < 0.7:
            if force2 > 0.5:
                force = Environment.PUSH
            elif force2 < -0.5:
                force = Environment.PULL
            else:
                force = Environment.NONE
        else:
            if force1 > 0.5:
                force = Environment.RIGHT
            elif force1 < -0.5:
                force = Environment.LEFT
            else:
                force = Environment.NONE

        self.joy = [force, reset]

    def save_model(self, model, filename):
        self.rl.save_agent()

    def act_and_trains(self, state, reward):
        self.action = self.rl.act_and_trains(state, reward)
        if self.action == Environment.FORWARD:
            name = 'FORWARD'
        elif self.action == Environment.REVERSE:
            name = 'REVERSE'
        elif self.action == Environment.TURN_LEFT:
            name = 'TURN LEFT'
        elif self.action == Environment.TURN_RIGHT:
            name = 'TURN RIGHT'
        elif self.action == Environment.LEFT_FORWARD:
            name = 'LEFT_FORWARD'
        elif self.action == Environment.RIGHT_FORWARD:
            name = 'RIGHT_FORWARD'
        elif self.action == Environment.LEFT_REVERSE:
            name = 'LEFT_REVERSE'
        elif self.action == Environment.RIGHT_REVERSE:
            name = 'RIGHT_REVERSE'
        else:
            name = 'STOP'
        return self.action, name

    def act(self, state):
        self.action = self.rl.act(state)
        return self.action

    def wait_keyboard_input(self):
        info = 'start new episode'
        if self.env.contact == Environment.NONE:
            info+=' no contact'
        elif self.env.contact == Environment.PUSH:
            info+=' push arm'
        elif self.env.contact == Environment.PULL:
            info+=' pull arm'
        raw_input(info)
        #while not self.joy[1]:
        self.env.base_reward = copy.deepcopy(self.env.pos)
        #    pass

    def train(self):
        try:
            while not self.env.state:
                print "PASS\r\n"
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'
        self.wait_keyboard_input()
        pygame.mixer.music.load('censor-beep-01.mp3')
        for episode in range(self.nepisodes):
            if self.episode and self.episode > episode:
                continue
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates]) 
            treward = []
            loss = 0
            step = 0
            pygame.mixer.music.load('censor-beep-01.mp3')
#            self.wait_keyboard_input()
            while not rospy.is_shutdown():
                if self.env.state:
                    self.state = self.env.state
                    self.state[3] = self.state[4] = self.state[5] = 0
                    a, name = self.act_and_trains(self.state, self.reward)
                    s2, self.reward, done, check = self.env.step(a, joy=self.joy[0])
                    print self.reward
#                    if self.reward < -0.1:
#                        self.reward = -100
#                        pygame.mixer.music.play(0)
#                    else:
#                        self.reward = 0

                    if check:
                        pygame.mixer.music.load('censor-beep-10.mp3')
                        pygame.mixer.music.play(0)
                        raw_input('press enter if connection is restored')
                        s = self.env.reset()

                    print 'epsode:' + str(episode) + 'step: ' + str(step) + ' reward: ' + str(self.reward) + ' action: ' + name + " " + str(self.state)
                    self.collect_state(episode,step, self.reward, name, self.state)
                    step += 1
                else:
                    done = False
                if done:
                    break
            self.rl.save_agent("model"+str(episode))
            self.rl.stop_episode_and_train(self.state, self.reward, done)

        pygame.mixer.music.load('censor-beep-02.mp3')
        pygame.mixer.music.play(0)
        while not rospy.is_shutdown():
            if self.env.state:
                self.state = self.env.state
                self.state[3] = self.state[4] = self.state[5] = 0
                a = self.act(self.state)
                s2, self.reward, done, check = self.env.step(a, joy=self.joy[0])
                print 'act ' + str(a) + ' ' + str(self.state)

    def collect_state(self,episode, step, reward, action,state):
        file2 = open('training2.csv', 'a')
        file2.write(str(episode)+',')
        file2.write(str(step)+',')
        file2.write(str(reward)+',')
        file2.write(str(action)+',')
        for pos in state:
            file2.write(str(pos)+',')
        file2.write('\n')
        file2.close()


if __name__ == '__main__':
    rospy.init_node('DDQN', disable_signals=True)
    rospy.loginfo('START ONLINE TRAINING')
    env = Environment()
    agent = Agent(env)
    start_time = time.time()

    try:
        agent.train()
        env.reset()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rospy.loginfo('EXIT TRAINING')

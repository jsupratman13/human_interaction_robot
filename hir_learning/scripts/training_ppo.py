#!/usr/bin/python

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ppo.py                             
#brief: proximal policy optimization
#author: Joshua Supratman                    
#last modified: 2018.07.31. 
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import chainerrl
import chainer
import chainer.functions as F
import chainerrl.links as L
import chainerrl.policies as P
import chainerrl.v_function as V
from chainerrl.agents import a3c, PPO


import numpy as np
import os
from os.path import expanduser

import rospy
import collections,random,sys,time,math,copy


from real_environment import Environment
from sensor_msgs.msg import Joy
import pygame

class A3CFF(chainer.ChainList, a3c.A3CModel):
    def __init__(self, n_actions):
        self.head = L.NIPSDQNHead()
        self.pi = P.FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = V.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)

class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    def __init__(self, nstates, nactions, hidden_sizes=(100,100)):
        self.pi = P.SoftmaxPolicy(model=L.MLP(nstates, nactions, hidden_sizes))
        self.v = L.MLP(nstates, 1, hidden_sizes=hidden_sizes)
        super(A3CFFSoftmax, self).__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    def __init__(self, n_observations, action_space, n_layers=2, n_nodes=50, bound_mean=True, normalize_obs=True):
        assert bound_mean in [False, True]
        assert normalize_obs in [False, True]
        super(A3CFFGaussian, self).__init__()
        self.normalize_obs = normalize_obs
        hidden_sizes = (n_nodes,)*n_layers
        with self.init_scope():
            self.pi = P.FCGaussianPolicyWithStateIndependentCovariance(n_observations, action_space.get_size(), n_layers, n_nodes, var_type='diagonal', nonlinearity=F.tanh, bound_mean=bound_mean, min_action=action_space.low(), max_action=action_space.high(), mean_wscale=1e-2)
            self.v = L.MLP(n_observations, 1, hidden_sizes=hidden_sizes)
            if normalize_obs:
                self.obs_filter = L.EmpiricalNormalization(shape=n_observations)

    def pi_and_v(self, state):
        if self.normalize_obs:
            state = F.clip(self.obs_filter(state, update=False), -5.0, 5.0)
        return self.pi(state), self.v(state)
    pass

class reinforcement_learning:
    def __init__(self, n_state=6, n_action=3, action_space=None):
        #model = AA3CFF(n_action) #for image
        model = A3CFFGaussian(n_state, action_space)
        #opt = chainer.optimizers.Adam(alpha=2.5e-4, eps=1e-5)
        opt = chainer.optimizers.Adam(alpha=0.01, eps=0.01)
        opt.setup(model)
        self.n_action = n_action
        phi = lambda x: np.array(x, dtype=np.float32)

        #self.agent = PPO(model, opt, phi=phi, update_interval=50, minibatch_size=16, epochs=2, clips_eps=0.1, clip_eps_vf=None, standardize_advantages=True)
        self.agent = PPO(model, opt, phi=phi, update_interval=50, minibatch_size=16, epochs=2, clip_eps_vf=None, entropy_coef=0.0, standardize_advantages=True)

    def act_and_trains(self, obs, reward):
        self.action = self.agent.act_and_train(obs, reward)
        return self.action

    def stop_episode_and_train(self, obs, reward, done):
        self.agent.stop_episode_and_train(obs,reward,done)

    def act(self, obs):
        self.action = self.agent.act(obs)
        return self.action

    def action_space_sample(self):
        return np.random.randint(1,self.n_action)

class Agent(object):
    def __init__(self,env):
        pygame.init()

        self.nepisodes = 15

        self.env = env
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        self.action = Environment.STOP
        self.state = [0,0,0,0,0,0]
        self.reward = 0
        self.integral_reward = 0

        self.rl = reinforcement_learning(self.nstates, self.nactions, env.action_space)

        self.loss_list = []
        self.episode = None

        file2 = open('training2.csv', 'a')
        file2.write('episode,step,reward,action,state\n')
        file2.close()

    def save_model(self, model, filename):
        self.rl.save_agent()

    def act_and_trains(self, state, reward):
        self.action = self.rl.act_and_trains(state, reward)
        return self.action

    def act(self, state):
        self.action = self.rl.act(state)
        return self.action

    def wait_keyboard_input(self):
        info = 'start new episode'
        raw_input(info)
        #while not self.joy[1]:
        self.env.base_reward = copy.deepcopy(self.env.pos)
        #    pass

    def train(self):
        try:
            while not self.env.state:
                print("PASS\r\n")
                pass
        except KeyboardInterrupt:
            assert False, 'failed to get joint state'
        self.wait_keyboard_input()
        pygame.mixer.music.load('censor-beep-01.mp3')
        epsilon = 0.2
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
                    if epsilon > np.random.rand():
                        a = self.env.action_space.sample()
                        name = 'RANDOM ' + str(a) + ' '
                    else:
                        a = self.act_and_trains(self.state, self.reward)
                        print(a)
                        a = np.clip(a,self.env.action_space.low(), self.env.action_space.high())
                        name = str(a)

                    if math.fabs(a) < 0.03: a = 0

                    s2, self.reward, done, check = self.env.step(a)
                    if self.reward < -1:
                        self.reward = -100
#                        pygame.mixer.music.play(0)
#                    else:
#                        self.reward = 0

                    if check:
                        pygame.mixer.music.load('censor-beep-10.mp3')
                        pygame.mixer.music.play(0)
                        raw_input('press enter if connection is restored')
                        s = self.env.reset()

                    print('epsode: ' + str(episode) + ' step: ' + str(step) + ' reward: ' + str(self.reward) + ' action: ' + name + " " + str(self.state))
                    self.collect_state(episode,step, self.reward, name, self.state)
                    step += 1
                else:
                    done = False
                if done:
                    break
            #self.rl.save_agent("model"+str(episode))
            self.rl.stop_episode_and_train(self.state, self.reward, done)
            epsilon *= 0.95

        pygame.mixer.music.load('censor-beep-02.mp3')
        pygame.mixer.music.play(0)
        while not rospy.is_shutdown():
            if self.env.state:
                self.state = self.env.state
                self.state[3] = self.state[4] = self.state[5] = 0
                a = self.act(self.state)
                a = np.clip(a,self.env.action_space.low(), self.env.action_space.high())
                s2, self.reward, done, check = self.env.step(a)
                print('act ' + str(a) + ' ' + str(self.state))

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

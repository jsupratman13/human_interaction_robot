#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#filename: ddpg.py                             
#brief: deep deterministic policy gradient                  
#author: Joshua Supratman                    
#last modified: 2018.06.28
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
import numpy as np
import time, math
import json
import matplotlib
import matplotlib.pyplot as plt
import collections,random,sys
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import ConfigParser
import copy

import rospy
from real_environment import Environment
#from sensor_msgs.msg import Joy

class Agent(object):
    def __init__(self,env):
        config = ConfigParser.RawConfigParser()
        config.read('real_parameters.cfg')
        self.env = env
        sess = tf.Session()
        K.set_session(sess)
        np.random.seed(123)

        self.gamma = config.getfloat('training','gamma')
        alpha_actor = config.getfloat('training','alpha_actor')
        alpha_critic = config.getfloat('training','alpha_critic')
        tau = config.getfloat('network','tau')
        self.nepisodes = config.getint('training','episodes')
        self.weights_name = 'episodefinal.hdf5'
        
        self.nstates = env.observation_space.get_size()
        self.nactions = env.action_space.get_size()
        
        self.batch_size = config.getint('network','batch_size')
        self.memory = collections.deque(maxlen=config.getint('network','memory_size'))

        self.actor = ActorNetwork(sess, self.nstates, self.nactions, tau, alpha_actor)
        self.critic = CriticNetwork(sess, self.nstates, self.nactions, tau, alpha_critic)
        self.noise = Ornstein_Uhlenbeck(np.zeros(self.nactions))
        
        self.loss_list = []
        self.reward_list = []

    def load_model(self,filename):
        json_file = open(filename,'r')
        model = model_from_json(json_file.read())
        model.summary()
        json_file.close() 
        return model

    def train(self):
        max_r = -10000
        self.actor.update_target_network()
        self.critic.update_target_network()
        raw_input('ready')
        self.env.base_reward = copy.deepcopy(self.env.pos)
        for episode in range(self.nepisodes):
            s = self.env.reset()
            #s = np.reshape(s,[1,self.nstates])
            s_t = np.hstack((s[0], s[1], s[2], s[3], s[4], s[5]))
            treward = []
            step = 0
            while True:
                loss = 0.0
                step += 1
                if np.random.rand() < 0.0:
                    a = [[self.env.action_space.sample()]]
                    print 'RANDOM: '+str(a)
                else:
                    a = self.actor.model.predict(s_t.reshape(1, s_t.shape[0])) + self.noise()
                    print 'ACTUAL: ' + str(a)
                a[0][0] = max(self.env.action_space.low(), min(self.env.action_space.high(), a[0][0]))
                if math.fabs(a[0][0]) < 0.03 : a[0][0] = 0

                s2, r, done, lost = self.env.step(a[0][0])
                s2_t = np.hstack((s2[0], s2[1], s2[2], s2[3], s2[4], s2[5]))

                if r < -2:
                    r = -100

                self.memory.append((s,a[0],r, s2_t,done))

                if len(self.memory) >= self.batch_size:
                    minibatch = random.sample(self.memory,self.batch_size)
                    states = np.asarray([e[0] for e in minibatch])
                    actions = np.asarray([e[1] for e in minibatch])
                    rewards = np.asarray([e[2] for e in minibatch])
                    next_states = np.asarray([e[3] for e in minibatch])
                    dones = np.asarray([e[4] for e in minibatch])
                    target = np.zeros(len(minibatch))

                    target_estimate = self.critic.target_model.predict([next_states, self.actor.target_model.predict(next_states)])

                    for i in range(len(minibatch)):
                        if dones[i]:
                            target[i] = rewards[i]
                        else:
                            target[i] = rewards[i] + self.gamma * target_estimate[i]

                    loss += self.critic.model.train_on_batch([states, actions], target)

                    a_for_grad = self.actor.model.predict(states)
                    grads = self.critic.gradients(states, a_for_grad)
                    self.actor.update_network(states, grads)

                    self.actor.update_target_network()
                    self.critic.update_target_network()

                s = s2
                treward.append(r)

                print 'episode: ' + str(episode) + ' step: ' + str(step) + ' reward: ' + str(r) + ' action: ' + str(a) + ' ' + str(s) 

                if done:
                    treward = sum(treward)/len(treward)
                    
                    #save checkpoint
                    #if treward >= max_r:
                    #    max_r = treward
                    #    self.actor.model.save_weights('episode'+str(episode)+'.hdf5', overwrite=True)

                    #print 'episode: ' + str(episode+1) + ' reward: ' + str(treward)
                    break
            
            self.reward_list.append(treward)
            self.loss_list.append(loss)
            
        self.actor.model.save_weights(self.weights_name)

    def test(self,modelname,weightname,ntrials=5):
        model = self.load_model(modelname)
        model.load_weights(weightname)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        for trial in range(ntrials):
            s = self.env.reset()
            s = np.reshape(s,[1,self.nstates])
            treward = 0
            while True:
                #self.env.render()
                a = model.predict(s)
                s2, r, done = self.env.step(np.array(a[0]))
                s = np.reshape(s2, [1,self.nstates])
                treward += r
                if done:
                    print 'trial: '+str(trial+1) + ' reward: ' + str(treward)
                    break

    def test_single(self,modelname,weightname,s1, s2):
        model = self.load_model(modelname)
        model.load_weights(weightname)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        env.reset()
        s = [s1, s2]
        s = np.reshape(s,[1,self.nstates])
        treward = 0
        for i in range(5):
            a = model.predict(s)
            s2, r, done = self.env.step(np.array(a[0]))
            s = np.reshape(s2, [1,self.nstates])
            treward += r
            print s2

    def record(self):
        file1 = open('result_ddpg.csv', 'a')
        if self.nepisodes:
            episodes = range(self.nepisodes, self.nepisodes+len(self.reward_list),1)
        else:
            file1.write('episode,reward,loss\n')
            episodes = range(len(self.reward_list))
        for i in range(len(episodes)):
            file1.write(str(episodes[i]))
            file1.write(','+str(self.reward_list[i]))
            file1.write(','+str(self.loss_list[i]))
            file1.write('\n')
        file1.close()

    def save_model(self):
        actor_model = self.actor.model.to_json()
        critic_model = self.critic.model.to_json()
        with open('actor_model.json', 'w') as json_file:
            json_file.write(actor_model)
        #with open('initial_critic', 'w') as json_file:
        #    json_file.write(critic_model)

class Ornstein_Uhlenbeck(object):
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.prev_x = self.x0 if self.x0 is not None else np.zeros(self.mu.size)

    def __call__(self):
        x = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.size)
        self.prev_x = x
        return x

class ActorNetwork(object):
    def __init__(self, sess, num_state, num_action, tau, learning_rate):
        self.tau = tau
        self.sess = sess

        self.model, self.weights, self.state = self.create_network(num_state, num_action)
        self.target_model, self.target_weights, self.target_state = self.create_network(num_state, num_action)
        self.action_gradient = tf.placeholder(tf.float32, [None, num_action])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def update_network(self, states, action_gradients):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.action_gradient: action_gradients})

    def update_target_network(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.tau*actor_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_network(self, num_state, num_action):
        #action_input = Input(shape=(1,)+num_state)
        state_input = Input(shape=[num_state])
        x = Dense(50, activation='relu')(state_input)
        x = Dense(10, activation='relu')(x)
        x = Dense(num_action, activation='linear')(x)
        model = Model(inputs=state_input, outputs=x)
        model_json = model.to_json()
        with open('actor_model.json','w') as json_file:
            json_file.write(model_json)
        return model, model.trainable_weights, state_input

class CriticNetwork(object):
    def __init__(self, sess, num_state, num_action, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate

        self.model, self.action, self.state = self.create_network(num_state, num_action)
        self.target_model, self.target_action, self.target_state = self.create_network(num_state, num_action)
        self.action_gradient = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_gradient, feed_dict={self.state: states, self.action: actions})[0]

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.tau*critic_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_network(self, num_state, num_action):
        #action_input = Input(shape=(num_action,))
        #state_input = Input(shape(1,)+num_state)
        #, Inputflattened_state = Flatten()(state_input)
        #x = concatenate([action_input, flattened_state])
        action_input = Input(shape=[num_action])
        state_input = Input(shape=[num_state])
        x = concatenate([state_input, action_input])
        x = Dense(10, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[state_input, action_input], outputs=x)
        model_json = model.to_json()
        with open('critic_model.json','w') as json_file:
            json_file.write(model_json)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model, action_input, state_input

if __name__ == '__main__':
    rospy.init_node('ddpg_learning', disable_signals=True)
    rospy.loginfo('START ONLINE TRAINING DDPG')
    env = Environment()
    agent = Agent(env)
    start_time = time.time()
    if len(sys.argv) > 2:
        agent.test_single(str(sys.argv[1]), str(sys.argv[2]), float(sys.argv[3]),float(sys.argv[4]))
        assert False, 'finished test'

    elif len(sys.argv) > 1:
        agent.test(str(sys.argv[1]), str(sys.argv[2]))
        assert False, 'finished test'

    else:
        agent.save_model()

    try:
        agent.train()
        rospy.loginfo('COMPLETE TRAINING')
        env.reset()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        agent.record()
        m,s = divmod(time.time()-start_time,60)
        h,m = divmod(m,60)
        rospy.loginfo('time took %d:%02d:%02d' %(h,m,s))
        print 'time took %d:%02d:%02d' %(h,m,s)
        rospy.loginfo('exit time')

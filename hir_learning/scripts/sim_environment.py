#!/usr/bin/python
import math,random
import sys,abc,time

class Environment(object):
    __metaclass__ = abc.ABCMeta

    # MOVE
    STOP = 0
    FORWARD = 1
    REVERSE = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    LEFT_FORWARD = 5
    RIGHT_FORWARD = 6
    LEFT_REVERSE = 7
    RIGHT_REVERSE = 8

    # FORCE
    PUSH = 0
    NONE = 1
    PULL = 2
    LEFT = 3
    RIGHT = 4

    def __init__(self):
        
        self.state = []
        self.initial_step_time = time.time()

        self.contact = Environment.NONE

        self.__observation_space = Environment.ObservationSpace()
        self.__action_space = Environment.ActionSpace()

        self.prev_action = Environment.STOP
        self.step_time = 0


        #self.f = open('data.csv', 'w')
        self.initial_flag  = True
        self.base_reward = []

    @property
    def action_space(self):
        return self.__action_space
    @property
    def observation_space(self):
        return self.__observation_space

    def __move(self, action):
        linear, angular = action
        self.state[0] = self.state[0] + linear
        self.state[1] = self.state[1] + angular

    def get_reward(self, action):
        stimulus = sum([math.pow(0 - state,2)for state in self.state])
        reward = -1 * stimulus * 100
        return reward

    def reset(self, test=0):
        self.initial_step_time = time.time()
        self.step_time = 0
        self.state = [random.uniform(-2,2),random.uniform(-2,2)]
        
        return self.state


    def step(self, action, joy=None):
        if joy is not None:
            self.contact = joy

        is_terminal = False
       
        self.step_time += 1
        
        self.__move(action)
        
        self.prev_action = action

       # self.collect(action)

        #if math.fabs(time.time()-self.initial_step_time) > 10:
        if self.step_time > 50:
            is_terminal = True
        
        reward = self.get_reward(action)
        
        return self.state, reward, is_terminal

    class ObservationSpace(object):
        def __init__(self):
            pass

        def get_size(self):
            return 2
            #return 5*2
            #return 3*2

    class ActionSpace(object):
        def __init__(self):
            self.low = [-0.2, -0.4]
            self.high = [0.2, 0.4]
            self.action_list = [
                                Environment.FORWARD,
                                Environment.STOP,
                                Environment.REVERSE,
                                Environment.TURN_LEFT,
                                Environment.TURN_RIGHT,
                                #Environment.LEFT_FORWARD,
                                #Environment.RIGHT_FORWARD,
                                #Environment.LEFT_REVERSE,
                                #Environment.RIGHT_REVERSE
                                ]
        
        def sample(self):
            return [random.uniform(self.low[0],self.high[0]),random.uniform(self.low[1],self.high[1])]
            #return random.choice(self.action_list)
    
        def get_size(self):
            #return len(self.action_list)
            return 2

        def low(self):
            return self.low
        
        def high(self):
            return self.high

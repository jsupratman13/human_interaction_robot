#!/usr/bin/python
import math,random
import sys,abc,time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

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
        #self.vel_error = [0 for i in range(5)]
        self.vel_error = [0 for i in range(3)]
        self.prev_action = [Environment.NONE for i in range(4)]
        self.pos_error = []
        self.pos = []
        #self.prev_pos_error = [0 for i in range(5)]
        self.prev_pos_error = [0 for i in range(3)]
        self.joint_names = []
        self.initial_step_time = time.time()

        self.contact = Environment.NONE

        self.__observation_space = Environment.ObservationSpace()
        self.__action_space = Environment.ActionSpace()

        self.step_time = 0

        self.sub = rospy.Subscriber('/manipulator/joint_states', JointState, self.__get_state)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)

        #self.f = open('data.csv', 'w')
        self.initial_flag  = True
        self.sleep_rate = 1 #in seconds
        self.rate = rospy.Rate(1/self.sleep_rate) #in Hz
        self.base_reward = []

    @property
    def action_space(self):
        return self.__action_space
    @property
    def observation_space(self):
        return self.__observation_space

    def __get_state(self, msg):
        self.joint_names = list(msg.name)
#        self.pos = [msg.position[0],msg.position[1],msg.position[2]*3,msg.position[3],msg.position[5]*3]
 #       self.pos_error = [msg.effort[0],msg.effort[1],msg.effort[2],msg.effort[3],msg.effort[5]]
        self.pos = [msg.position[0],msg.position[1],msg.position[3]]
        self.pos_error = [msg.effort[0],msg.effort[1],msg.effort[3]]
        for i in range(len(self.vel_error)):
            self.vel_error[i] = (self.pos_error[i] - self.prev_pos_error[i])/self.sleep_rate
        for j in range(len(self.prev_pos_error)):
            self.prev_pos_error[j] = self.pos_error[j]
        
        #self.state = self.pos_error + self.vel_error + self.prev_action
        self.state = self.pos_error + self.vel_error
        #self.state = self.pos_error
    
    def __move(self, action):
        vel = Twist()
        if action == Environment.STOP:
            vel.linear.x = 0
            vel.angular.z = 0
        elif action == Environment.REVERSE:
            vel.linear.x = -0.2
            vel.angular.z = 0
        elif action == Environment.FORWARD:
            vel.linear.x = 0.2
            vel.angular.z = 0
        elif action == Environment.TURN_LEFT:
            vel.linear.x = 0
            vel.angular.z = 0.4
        elif action == Environment.TURN_RIGHT:
            vel.linear.x = 0
            vel.angular.z = -0.4            
        elif action == Environment.LEFT_FORWARD:
            vel.linear.x = 0.2
            vel.angular.z = 0.4            
        elif action == Environment.RIGHT_FORWARD:
            vel.linear.x = 0.2
            vel.angular.z = -0.4            
        elif action == Environment.LEFT_REVERSE:
            vel.linear.x = -0.2
            vel.angular.z = 0.4            
        elif action == Environment.RIGHT_REVERSE:
            vel.linear.x = -0.2
            vel.angular.z = -0.4            
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        self.pub.publish(vel)

    def __vel_move(self, action):
        translate, rotate = action
        vel = Twist()
        vel.linear.x = translate
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.z = rotate
        vel.angular.x = 0
        vel.angular.y = 0
        self.pub.publish(vel)

    def collect(self,action):
        if self.initial_flag:
            self.initial_flag = False
            self.f.write('step,')
            for i in self.joint_names:
                self.f.write(str(i)+',')
            self.f.write('action\n')
        self.f.write(str(self.step_time)+',')
        for j in self.pos_error:
            self.f.write(str(j)+',')
        self.f.write(str(action)+'\n')

    def get_reward(self, action):
        #reward = 0
        #if self.contact == Environment.NONE and action == Environment.STOP:
        #   reward = 100
        #elif self.contact == Environment.PUSH and action == Environment.REVERSE:
        #    reward = 100
        #elif self.contact == Environment.PULL and action == Environment.FORWARD:
        #    reward = 100
        #elif self.contact == Environment.LEFT and action == Environment.TURN_LEFT:
        #    reward = 100
        #elif self.contact == Environment.RIGHT and action == Environment.TURN_RIGHT:
        #    reward = 100
        stimulus = sum([math.pow(self.base_reward[i] - self.pos[i],2)for i in range(len(self.pos))])
        reward = -1 * stimulus * 100
        return reward

    def reset(self, cont=False):
        self.contact = random.choice([Environment.NONE, Environment.PUSH, Environment.PULL])
        self.initial_step_time = time.time()
        if not cont:
            self.step_time = 0
        self.prev_action = [Environment.NONE for i in range(4)]
        
        return self.state


    def step(self, action, joy=None):
        if joy is not None:
            self.contact = joy

        is_terminal = False
        lost = False
       
        self.step_time += 1
        self.__move(action)

        self.prev_action.pop()
        self.prev_action.insert(0,action)

       # self.collect(action)

        #if math.fabs(time.time()-self.initial_step_time) > 10:
        if self.step_time > 50:
            is_terminal = True
        
        if self.sub.get_num_connections() != 1:
            print 'lost connection'
            lost = True

        self.rate.sleep()        
        reward = self.get_reward(action)
        
        return self.state, reward, is_terminal, lost

    class ObservationSpace(object):
        def __init__(self):
            pass

        def get_size(self):
            #return 5*2
            return 3*2
            #return 3
            #return 3*2+4

    class ActionSpace(object):
        def __init__(self):
            self.action_list = [
                                Environment.FORWARD,
                                Environment.STOP,
                                Environment.REVERSE,
                                #Environment.TURN_LEFT,
                                #Environment.TURN_RIGHT,
                                #Environment.LEFT_FORWARD,
                                #Environment.RIGHT_FORWARD,
                                #Environment.LEFT_REVERSE,
                                #Environment.RIGHT_REVERSE
                                ]
        
        def sample(self):
            return random.choice(self.action_list)
    
        def get_size(self):
            return len(self.action_list)
            #return 2

        def low(self):
            return [-0.4,-0.2]
        
        def high(self):
            return [0.4,0.2]

if __name__ == '__main__':
    try:
        rospy.init_node('collect_data', disable_signals=True)
        env = Environment()

    except (KeyboardInterrupt, SystemExit):
        pass

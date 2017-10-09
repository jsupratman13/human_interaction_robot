#!/usr/bin/python
import rospy
import math,random
import sys,abc,time
from gazebo_msgs.srv import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Point, Wrench, Twist
from std_srvs.srv import Empty
import initialize

class Environment(object):
    __metaclass__ = abc.ABCMeta

    FORWARD = 0
    STOP = 1
    REVERSE = 2

    PUSH = 0
    NONE = 1
    PULL = 2

    def __init__(self):
        initialize.move_group()
        self.sub = rospy.Subscriber('/manipulator/left_arm_controller/state', JointTrajectoryControllerState, self.__get_state)
        self.pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
        self.__apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.__sim_reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.__clear_force = rospy.ServiceProxy('/gazebo/clear_body_wrenches',BodyRequest)

        self.state = []
        self.vel_error = []
        self.pos_error = []
        self.joint_names = []
        self.initial_step_time = 0

        self.contact = Environment.NONE

        self.__observation_space = Environment.ObservationSpace()
        self.__action_space = Environment.ActionSpace()

        self.force = 0

    @property
    def action_space(self):
        return self.__action_space
    @property
    def observation_space(self):
        return self.__observation_space

    def recording(self, force, state):
        f = open('state_force.csv','a')
        f.write(str(force)+','+str(state[0])+','+str(self.vel_error[2])+'\n')
        f.close()

    def __get_state(self, msg):
        self.joint_names =  msg.joint_names
        self.pos_error = list(msg.error.positions)
        self.vel_error = list(msg.error.velocities)
        self.state = self.pos_error + self.vel_error
        #self.state = [self.pos_error[2], self.vel_error[2]]
        #self.state = [self.pos_error[2]]

    def move(self, action):
        vel = Twist()
        if action == Environment.FORWARD:
            vel.linear.x = 0.5
        elif action == Environment.STOP:
            vel.linear.x = 0.0
        elif action == Environment.REVERSE:
            vel.linear.x = -0.5
        else:
            assert False, 'unkonwn action'
        vel.linear.y = 0
        vel.linear.z = 0
        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = 0
        self.pub.publish(vel)

    def apply_force(self, force):
        self.force = force
        try:
            body_name = 'robot1::wrist_roll_link'
            reference_frame = 'robot1::wrist_roll_link'
            point = Point()
            point.x = 0
            point.y = 0
            point.z = 0
            wrench = Wrench()
            wrench.force.x = self.force
            wrench.force.y = 0
            wrench.force.z = 0
            wrench.torque.x = 0
            wrench.torque.y = 0
            wrench.torque.z = 0
            start_time = rospy.Time.now()
            duration = rospy.Duration(10)

            self.__apply_force(body_name, reference_frame, point, wrench, start_time, duration)

        except rospy.ServiceException as e:
            rospy.loginfo('apply force failed %s', e)

    def clear_force(self):
        self.force = 0
        try:
            body_name = 'robot1::wrist_roll_link'
            self.__clear_force(body_name)
        except rospy.ServiceException as e:
            rospy.loginfo('clear force failed %s',e) 

    def reset_sim(self):
        try:
            self.__sim_reset()
        except rospy.ServiceException as e:
            rospy.loginfo('reset simulation failed %s', e)

    def get_reward(self, action):
        if self.contact == Environment.NONE and action == Environment.STOP:
            return 100
        elif self.contact == Environment.PUSH and action == Environment.REVERSE:
            return 100
        elif self.contact == Environment.PULL and action == Environment.FORWARD:
            return  100
	return 0
        #return -1 * sum(self.state)

    def reset(self, test=0):
        self.move(Environment.STOP)
        self.clear_force()
        self.reset_sim() 
        time.sleep(10)
       
        if 0 < test <= 3:
            print 'pull'
            self.contact = Environment.PULL
            self.apply_force(random.randint(10,30))
        elif 3 < test <= 6:
            print 'push'
            self.contact = Environment.PUSH
            self.apply_force(random.randint(-30,-10))
        elif 6 < test <= 9:
            print 'none'
            self.contact = Environment.NONE
            self.apply_force(0)
        else: 
            prob = random.random()*100
            if prob < 33:
                print 'pull'
                self.contact = Environment.PULL
                self.apply_force(random.randint(10,30))
            elif 33 <= prob < 66:
                print 'push'
                self.contact = Environment.PUSH
                self.apply_force(random.randint(-30,-10))
            else:
                print 'none'
                self.contact = Environment.NONE
                self.apply_force(0)

        self.initial_step_time = rospy.Time.now().secs
        return self.state

    def step(self, action):
        is_terminal = False
        self.move(action)
        reward = self.get_reward(action)
        
        if math.fabs(rospy.Time.now().secs-self.initial_step_time) > 10:
            is_terminal = True

        #self.recording(self.force, self.state)

        #if reward > -0.08:
        #    is_terminal = True

        return self.state, reward, is_terminal

    class ObservationSpace(object):
        def __init__(self):
            pass

        def get_size(self):
            return 12
            #return 2

    class ActionSpace(object):
        def __init__(self):
            self.action_list = [Environment.FORWARD,
                                Environment.STOP,
                                Environment.REVERSE]
        
        def sample(self):
            return random.choice(self.action_list)
    
        def get_size(self):
            return len(self.action_list)


if __name__ == '__main__':
    rospy.init_node('enivornment')
    rospy.loginfo('test')
    env = Environment()
    env.reset()
    action = Environment.FORWARD
    try:
        env.apply_force_pull()
        #env.move(action)
        #print env.get_reward(action)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

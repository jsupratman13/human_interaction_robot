#!/usr/bin/python
import sys
import rospy
from gazebo_msgs.srv import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Point, Wrench
from std_srvs.srv import Empty

def push():
    try:
        service = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        
        joint_name = 'wrist_roll_joint'
        effort = 10000.0
        start_time = rospy.Time.now()
        duration = rospy.Duration(10)

        success = service(joint_name, effort, start_time, duration)

        print 'service call %s', success
    except rospy.ServiceException as e:
        rospy.loginfo('service call failed %s', e)

def push2():
    try:
        service = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        
        body_name = 'robot1::wrist_roll_link'
        reference_frame = 'robot1::wrist_roll_link'
        point = Point()
        point.x = 0
        point.y = 0
        point.z = 0
        wrench = Wrench()
        wrench.force.x = 35
        wrench.force.y = 0
        wrench.force.z = 0
        wrench.torque.x = 0
        wrench.torque.y = 0
        wrench.torque.z = 0
        start_time = rospy.Time.now()
        duration = rospy.Duration(10)

        success = service(body_name, reference_frame, point, wrench, start_time, duration)

        print 'service call %s', success 
    except rospy.ServiceException as e:
        rospy.loginfo('service call failed %s', e)

def read_state(msg):
    print msg.joint_names
    print msg.error.positions

def reset():
    try:
        service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        service()
    except rospy.ServiceException as e:
        rospy.loginfo('service call failed %s', e)

if __name__ == '__main__':
    rospy.init_node("test", anonymous=True)
    sub = rospy.Subscriber('/manipulator/left_arm_controller/state', JointTrajectoryControllerState, read_state)
    try:
        push2()
        rospy.sleep(5);
        reset();
    except rospy.ROSInterruptException:
        pass
    rospy.spin()

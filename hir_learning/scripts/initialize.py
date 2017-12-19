#!/usr/bin/python
import sys
import rospy
import numpy as np
import copy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

def move_group():
    rospy.loginfo("initialize ar")
    moveit_commander.roscpp_initialize(sys.argv)
    
    pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
    vel = Twist()

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("manipulator")

    rospy.loginfo("--------- generating plan based on predefined pos")
    group.clear_pose_targets()
    group.set_named_target('initial')
    plan2 = group.plan()
    group.go(wait=True)
    rospy.loginfo("--------- waiting while rviz display plan2")
    rospy.sleep(5)

    vel.linear.x = 0
    pub.publish(vel)
    #moveit_commander.roscpp_shutdown()
    rospy.loginfo("--------- FINISHED")

def move_pos():
    rospy.loginfo('initalize arm')
    elbow_pub = rospy.Publisher('/elbow_pitch_controller/command', Float64, queue_size=10)
    shoulder_pub = rospy.Publisher('/shoulder_pitch_controller/command', Float64, queue_size=10)
    wrist_pub = rospy.Publisher('/wrist_pitch_controller/command', Float64, queue_size=10)
    joint = Float64()
    
    shoulder_traj = [-0.2]
    elbow_traj = np.arange(0, 3.0, 0.2)
    wrist_traj = np.arange(0, 1.5 , 0.2)

    for e in elbow_traj:
        joint.data = e
        elbow_pub.publish(joint)
        rospy.sleep(0.5)
    for w in wrist_traj:
        joint.data = w
        wrist_pub.publish(joint)
        rospy.sleep(0.5)
    for s in shoulder_traj:
        joint.data = s
        shoulder_pub.publish(joint)
        rospy.sleep(0.5)

    rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('initialize')
        move_pos()
    except rospy.ROSInterruptException:
        pass
            

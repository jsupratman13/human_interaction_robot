#!/usr/bin/python
import sys
import rospy
import copy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Twist

def move_group():
    rospy.loginfo("initialize ar")
    moveit_commander.roscpp_initialize(sys.argv)
    
    pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
    vel = Twist()

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("arm")

    rospy.loginfo("--------- generating plan based on predefined pos")
    group.clear_pose_targets()
    group.set_named_target('initial_pose')
    plan2 = group.plan()
    group.go(wait=True)
    rospy.loginfo("--------- waiting while rviz display plan2")
    rospy.sleep(5)

    vel.linear.x = 0
    pub.publish(vel)
    #moveit_commander.roscpp_shutdown()
    rospy.loginfo("--------- FINISHED")


if __name__ == '__main__':
    try:
        move_group()
    except rospy.ROSInterruptException:
        pass
            

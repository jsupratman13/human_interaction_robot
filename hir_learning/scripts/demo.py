#!/usr/bin/python
import sys
import rospy
import copy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Twist

def move_group():
    rospy.loginfo("start demo")
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("demo", anonymous=True)
    
    pub = rospy.Publisher('/icart_mini/cmd_vel', Twist, queue_size=10)
    vel = Twist()

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander("arm")

    rospy.loginfo("--------- reference frame: %s", group.get_planning_frame())
    rospy.loginfo("--------- end effector: %s", group.get_end_effector_link())
    print "---------------- rospy.loginfo robot groups:"
    print robot.get_group_names()
    print "---------------- rospy.loginfo robot state" 
    print robot.get_current_state()
    print "-----------------"

    rospy.loginfo("-------- generating plan based on joint")
    group.clear_pose_targets()
    group_variable_values = group.get_current_joint_values()
    group_variable_values[2] = 0.5
    print group_variable_values
    group.set_joint_value_target(group_variable_values)
    plan1 = group.plan()
    group.go(wait=True)
    rospy.loginfo("--------- waiting while rviz display plan")
    rospy.sleep(5)

    rospy.loginfo("--------- generating plan based on predefined pos")
    group.clear_pose_targets()
    group.set_named_target('initial_pose')
    plan2 = group.plan()
    group.go(wait=True)
    rospy.loginfo("--------- waiting while rviz display plan2")
    rospy.sleep(5)


    rospy.loginfo("--------- generate plan while robot is moving")
    group.clear_pose_targets()
    group.set_named_target('pulled_pose')
    plan3 = group.plan()
    group.go(wait=True)
    vel.linear.x = 0.5
    pub.publish(vel)
    rospy.loginfo("--------- waiting while rviz display plan3")
    rospy.sleep(5)

    vel.linear.x = 0
    pub.publish(vel)
    moveit_commander.roscpp_shutdown()
    rospy.loginfo("--------- FINISHED")


if __name__ == '__main__':
    try:
        move_group()
    except rospy.ROSInterruptException:
        pass
            

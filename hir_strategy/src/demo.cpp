#include <ros/ros.h>
#include <moveit/move_group_interface/move_group.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/Twist.h>


int main(int argc, char **argv){
	ros::init(argc, argv, "demo");
	ros::NodeHandle nh;
	ros::AsyncSpinner spinner(1);
	spinner.start();

	ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("icart_mini/cmd_vel", 1000);
	geometry_msgs::Twist vel;

	moveit::planning_interface::MoveGroup group("arm");
	moveit::planning_interface::PlanningSceneInterface scene;
	moveit::planning_interface::MoveGroup::Plan my_plan;

	std::vector<double> group_variable_values;
	bool success;

	ROS_INFO("-------- Reference Frame: %s", group.getPlanningFrame().c_str());
	ROS_INFO("-------- End Effector: %s", group.getEndEffectorLink().c_str());

	ROS_INFO("-------- Generate Plan Based On Joint");
	//group.clearPoseTargets();
	group.getCurrentState()->copyJointGroupPositions(group.getCurrentState()->getRobotModel()->getJointModelGroup(group.getName()), group_variable_values);
	group_variable_values[2] = 0.5;
	group.setJointValueTarget(group_variable_values);
	success = group.plan(my_plan);
	group.move();
	ROS_INFO("-------- Plan %s ", success?"SUCCESS":"FAILED");
	sleep(5.0);

	ROS_INFO("-------- Generate Plan Based ON Predefined Pos");
	//group.clearPoseTargets();
	group.setNamedTarget("initial_pose");
	success = group.plan(my_plan);
	group.move();
	ROS_INFO("-------- Plan %s ", success?"SUCCESS":"FAILED");
	sleep(5.0);

	ROS_INFO("-------- Generate Plan While Robot Is Moving");
	group.setNamedTarget("pulled_pose");
	success = group.plan(my_plan);
	group.move();
	vel.linear.x = 0.5;
	pub.publish(vel);
	ROS_INFO("-------- Plan %s ", success?"SUCCESS":"FAILED");
	sleep(5.0);

	vel.linear.x = 0.0;
	pub.publish(vel);
	ROS_INFO("-------- Finshed");
	ros::shutdown();
	return 0;
}

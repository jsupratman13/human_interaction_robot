#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <gazebo_msgs/ApplyBodyWrench.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <control_msgs/JointTrajectoryControllerState.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Wrench.h>
#include <cstdlib>

//get state
void getState(const control_msgs::JointTrajectoryControllerState& msg){
	for(int i = 0; i < msg.joint_names.size(); i++){
		ROS_INFO_STREAM(msg.joint_names[i] << ' ' << msg.error.positions[i]);
	}
}

int main(int argc, char **argv){
	ros::init(argc, argv, "test");
	ros::NodeHandle nh;
	ros::Subscriber sub = nh.subscribe("/manipulator/left_arm_controller/state", 1000, getState);
	bool success;

	//push simulation
	ros::ServiceClient client = nh.serviceClient<gazebo_msgs::ApplyBodyWrench>("/gazebo/apply_body_wrench");
	gazebo_msgs::ApplyBodyWrench abw;
	std::string body_name = "robot1::wrist_roll_link";
	std::string reference_frame = "robot1::wrist_roll_link";
	geometry_msgs::Point point;
	geometry_msgs::Wrench wrench;
	point.x = 0;
	point.y = 0;
	point.z = 0;
	wrench.force.x = 40;
	wrench.force.y = 0;
	wrench.force.z = 0;
	wrench.torque.x = 0;
	wrench.torque.y = 0;
	wrench.torque.z = 0;
	ros::Time start_time = ros::Time::now();
	ros::Duration duration(10);
	
	abw.request.body_name = body_name;
	abw.request.reference_frame = reference_frame;
	abw.request.reference_point = point;
	abw.request.wrench = wrench;
	abw.request.start_time = start_time;
	abw.request.duration = duration;
	
	success = client.call(abw);

	if (success){
		ROS_INFO_STREAM("PUSH SUCCESS");
	}
	sleep(5);
	
	//reset simulation
	ros::ServiceClient client2 = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_simulation");
	std_srvs::Empty empty;
	success = client2.call(empty);
	if(success){
		ROS_INFO_STREAM("RESET SUCCESS");
	}
	sleep(5);
	
	ros::spin();
	return 0;
}

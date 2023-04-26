# Functions for handling the ROS callbacks
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry


def orientation2angular(orientation: PoseStamped) -> Vector3:
    quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    angular = Vector3(euler[0], euler[1], euler[2])
    return angular


def tOdometry_callback(odometry: Odometry, state_traj, ego_callback) -> None:
    now = rospy.get_rostime()
    time = now.secs + now.nsecs * pow(10, -9)
    p = odometry.pose.pose.position
    angular = orientation2angular(odometry.pose.pose.orientation)
    state = [p.x, p.y, angular.z]
    ego_callback(state_traj, state, time)


def agent_callback(agentPose: PoseStamped, agentname: str, cbf_list, agent_callback) -> None:
    now = rospy.get_rostime()
    time = now.secs + now.nsecs * pow(10, -9)
    p = agentPose.pose.pose.position
    angular = orientation2angular(agentPose.pose.pose.orientation)
    state = [p.x, p.y, angular.z]
    for i in range(len(cbf_list)):
        if cbf_list[i].agent.name == agentname:
            agent_callback(cbf_list[i].agent.state_traj, state, time)


# Function to publish control signals
def publish(ego_control_traj, vw_publisher, u) -> None:
    now = rospy.get_rostime()
    time = now.secs + now.nsecs * pow(10, -9)
    vel_msg = Twist()
    vel_msg.linear.x = u[0]
    vel_msg.angular.z = u[1]
    new_control_traj = add_control_traj(ego_control_traj, u, time)
    vw_publisher.publish(vel_msg)

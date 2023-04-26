#! /usr/bin/env python3

import argparse

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, SetModelState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, String

freq = 10  # Hz
speed = -0.2  # m/s


class Agent(object):
    def __init__(self, model_name):
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_model_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.pub = rospy.Publisher("/" + model_name + "pose", PoseStamped, queue_size=10)

        self.model = GetModelStateRequest()
        self.model.model_name = model_name

    def control_callback(self, event):
        result = self.get_model_srv(self.model)
        rospy.loginfo(result)

        state_msg = ModelState()
        state_msg.model_name = self.model.model_name
        state_msg.pose = result.pose
        state_msg.twist = result.twist
        state_msg.pose.position.x = result.pose.position.x + 0.6 * speed / freq
        resp = self.set_model_srv(state_msg)
        pose = PoseStamped()
        pose.pose = state_msg.pose
        self.pub.publish(pose)


if __name__ == "__main__":
    # Process arguments
    p = argparse.ArgumentParser(description="agent node")
    p.add_argument(
        "--model_name",
        nargs=1,
        type=str,
        required=True,
        help="the taregt model name on gazebo",
    )
    args = p.parse_args(rospy.myargv()[1:])

    try:
        rospy.init_node(args.model_name[0] + "_controller")
        agent = Agent(args.model_name[0])
        rospy.Timer(rospy.Duration(1.0 / freq), agent.control_callback)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

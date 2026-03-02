"""ROS2 runtime helpers for sensor/controller execution loops."""

from typing import Callable

import rclpy
from std_msgs.msg import Float64

from cbfkit.utils.user_types import ControllerCallable, SensorCallable

NODE_NAME = "ros2_node"  # Replace with an appropriate node name
execution_time_publisher = None  # Global publisher variable


def spin(node_name: str, callback: Callable, frequency: float = 0.01):
    """Create a ROS2 spinner that invokes `callback` at `frequency` Hz."""

    def spinner():
        nonlocal node_name, callback, frequency

        rclpy.init()

        # Set up callback
        node = rclpy.create_node(node_name)
        timer_period = 1.0 / frequency
        node.create_timer(timer_period, callback)

        rclpy.spin(node)

    return spinner


def stepper(
    sensor: SensorCallable,
    controller: ControllerCallable,
) -> Callable[[None], None]:
    """Step function to take the simulation forward one timestep. Designed to work generically with
    broad classes of dynamics, controllers, and estimators.

    Args:
        sensor (Callable): function handle to generate new state sensor
        controller (Callable): function handle to compute control input

    Returns
    -------
        step (Callable): function handle for computing one step in ROS2 sim/experiment
    """

    def step(_something):
        nonlocal sensor, controller

        start_time = rclpy.clock.Clock().now().to_msg().sec

        # Get sensor measurement (subscribers inside) -- assume perfect measurements
        t, y = sensor()

        # Compute control input using ROS2-wrapped controller (takes care of publishing)
        _u, _data = controller(t, y)

        # Compute execution time
        execution_time_ms = 1e3 * (rclpy.clock.Clock().now().to_msg().sec - start_time)

        # Execution time message
        global execution_time_publisher
        if execution_time_publisher is not None:
            execution_time_msg = Float64()
            execution_time_msg.data = execution_time_ms
            execution_time_publisher.publish(execution_time_msg)

    return step

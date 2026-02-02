# Code-generated
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path to allow importing tutorials package
root_path = Path(__file__).resolve().parents[4]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
except ImportError:
    # Mock rclpy for testing/simulation environments
    print("rclpy not found, using mocks")
    rclpy = MagicMock()
    # Configure spin to not block so script exits
    rclpy.spin = lambda node: None

    class Node:
        def __init__(self, name): pass
        def create_publisher(self, *args): return MagicMock()
        def create_subscription(self, *args): return MagicMock()
        def create_timer(self, *args): return MagicMock()
        def destroy_node(self): pass
        def get_logger(self): return MagicMock()

    class Float32MultiArray:
        def __init__(self): self.data = []

import jax.numpy as jnp

# Handle missing config module (config.py is missing in repo)
try:
    import config
except ImportError:
    class Config:
        QUEUE_SIZE = 10
        CONTROLLER_NAME = "controller_1"
        CONTROLLER_PARAMS = {"k_p": 1.0}
        TIMER_INTERVAL = 0.1
        MODULE_NAME = "tutorials.tutorials.mppi_cbf_si"
    config = Config()


class MppiCbfSiController(Node):
    def __init__(self):
        super().__init__("mppi_cbf_si_controller")
        self.publisher = self.create_publisher(
            Float32MultiArray, "control_command", config.QUEUE_SIZE
        )
        self.subscription = self.create_subscription(
            Float32MultiArray, "state_estimate", self.listener_callback, config.QUEUE_SIZE
        )
        self.controller = self.get_controller_by_name(
            config.CONTROLLER_NAME, **config.CONTROLLER_PARAMS
        )
        self.reset_control_msg()
        self.timer = self.create_timer(config.TIMER_INTERVAL, self.publish_control)

    @staticmethod
    def get_controller_by_name(name, **kwargs):
        """
        Dynamically import and return an instance of the specified controller.
        """
        module = __import__(config.MODULE_NAME + "." + "controllers", fromlist=[name])
        controller_class = getattr(module, name)
        return controller_class(**kwargs)

    def reset_control_msg(self):
        control_msg = Float32MultiArray()
        control_msg.data = [0.0]
        self.control_msg = control_msg

    def publish_control(self):
        self.publisher.publish(self.control_msg)

    def listener_callback(self, msg):
        estimate = jnp.array(msg.data)
        t = 0.0  # USER_DEFINED
        # Dummy u_nom and key for ROS node usage
        u_nom = jnp.zeros(1)
        key = 0
        from cbfkit.utils.user_types import ControllerData

        data = ControllerData()

        control_command, _ = self.controller(t, estimate, u_nom, key, data)
        self.reset_control_msg()
        if control_command.ndim > 0:
            self.control_msg.data = [float(c) for c in control_command]
        else:
            self.control_msg.data = [float(control_command)]


def main(args=None):
    rclpy.init(args=args)
    node = MppiCbfSiController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
